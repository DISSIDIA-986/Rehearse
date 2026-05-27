"""LocalVocal entry point: startup, native smoke test, and the live voice loop.

Startup order (Codex contract): load decks -> warmup LLM -> think_probe (fails
loud on thinking/slow) -> load ASR/TTS/VAD models (warm). Then either:
  --smoke : health checks + a TTS->ASR round-trip + device list, then exit (no mic)
  (default): the continuous half-duplex conversation loop until Ctrl-C / "stop"

The live loop needs a microphone, so it is validated manually; --smoke validates
everything that does not need audio hardware.
"""

from __future__ import annotations

import argparse
import glob
import sys
import threading
import time

import numpy as np

from localvocal import audio_io
from localvocal.anki_loader import load_sentences
from localvocal.duplex import HalfDuplexGate
from localvocal.llm_client import DEFAULT_MODEL, think_probe, warmup
from localvocal.loop_core import UtteranceAssembler, is_stop
from localvocal.pipeline import respond
from localvocal.prompt_builder import build_system_prompt
from localvocal.session_seeder import PracticeStat, select_targets
from localvocal.vad import EndpointConfig, EndpointDetector, SileroVad

PREROLL_MS = 200
FRAME = 512  # samples @16k (Silero window, 32ms)


def _load_models():
    from localvocal.asr import WhisperASR
    from localvocal.tts import KokoroTTS

    return WhisperASR(), KokoroTTS()


def smoke(decks: list[str], model: str) -> int:
    """Native smoke test: everything that does not need a microphone."""
    print("== LocalVocal smoke ==")
    ok = True

    try:
        sentences = load_sentences(decks)
        print(f"[ok] decks: {len(sentences)} sentences from {len(decks)} file(s)")
    except Exception as e:
        print(f"[FAIL] decks: {e}"); ok = False; sentences = []

    try:
        warmup(model=model)
        r = think_probe(model=model)
        print(f"[ok] LLM {model}: non-thinking, ttft={r.ttft_s:.2f}s, reply={r.text[:40]!r}")
    except Exception as e:
        print(f"[FAIL] LLM probe: {e}"); ok = False

    try:
        asr, tts = _load_models()
        phrase = "The weather is really nice today."
        audio = tts.synth(phrase)
        a16 = audio_io.resample(audio, tts.sr, audio_io.ASR_SR)
        heard = asr.transcribe(a16)
        match = set(phrase.lower().replace(".", "").split()) <= set(
            heard.lower().replace(".", "").split()
        )
        print(f"[{'ok' if match else 'FAIL'}] TTS->ASR round-trip: {heard!r}")
        ok = ok and match  # gate on words surviving, not mere non-empty
    except Exception as e:
        print(f"[FAIL] audio round-trip: {e}"); ok = False

    try:
        from localvocal.practiced_scorer import ollama_embed

        ollama_embed(["ping"])
        print("[ok] nomic-embed reachable (D3 practiced scoring)")
    except Exception as e:
        print(f"[WARN] nomic-embed: {e} (practiced tracking will be off)")

    try:
        SileroVad()
        print("[ok] Silero VAD loaded")
    except Exception as e:
        print(f"[WARN] Silero VAD: {e} (install --extra vad)")

    try:
        import sounddevice as sd

        ins = [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
        outs = [d["name"] for d in sd.query_devices() if d["max_output_channels"] > 0]
        print(f"[ok] audio devices: {len(ins)} input, {len(outs)} output")
    except Exception as e:
        print(f"[WARN] audio devices: {e}")

    print("== smoke PASS ==" if ok else "== smoke FAIL ==")
    return 0 if ok else 1


def run_loop(decks: list[str], model: str, voice: str, n_targets: int,
             full_duplex: bool) -> int:  # pragma: no cover - needs a microphone
    import queue

    try:
        import sounddevice as sd
    except Exception as e:
        print(f"sounddevice unavailable ({e}). Install: uv sync --extra audio",
              file=sys.stderr)
        return 1
    from localvocal.tts import KokoroTTS

    sentences = load_sentences(decks)
    if not sentences:
        print("No sentences loaded — check --decks paths.", file=sys.stderr)
        return 1
    print(f"Loaded {len(sentences)} sentences. Warming up {model}...")
    try:
        warmup(model=model)
        think_probe(model=model)
        asr, tts = _load_models()
        if voice:
            tts = KokoroTTS(voice=voice)
        vad = SileroVad()
    except Exception as e:
        print(f"Startup failed: {e}", file=sys.stderr)
        return 1

    # Capture at the device's NATIVE rate (PortAudio won't always accept 16k),
    # resample each block to 16k once for VAD/ASR. Keeps the input device open
    # for the whole session (Codex blind-spot #2: never reopen per turn).
    try:
        in_sr = int(sd.query_devices(kind="input")["default_samplerate"])
    except Exception:
        in_sr = audio_io.DEFAULT_DEVICE_SR
    block = max(1, round(FRAME * in_sr / audio_io.ASR_SR))

    # D3 practiced scoring needs nomic-embed; probe once, warn (not fatal) if down.
    from localvocal.practiced_scorer import ollama_embed
    practiced_on = True
    try:
        ollama_embed(["ping"])
    except Exception as e:
        practiced_on = False
        print(f"  (nomic-embed unreachable: {e} — practiced tracking off)")

    try:
        out_sr = int(sd.query_devices(kind="output")["default_samplerate"])
    except Exception:
        out_sr = audio_io.DEFAULT_DEVICE_SR

    gate = HalfDuplexGate(full_duplex=full_duplex)
    endpoint = EndpointDetector(EndpointConfig())
    preroll = audio_io.RingBuffer(int(PREROLL_MS / 1000 * audio_io.ASR_SR))
    vad_mon = SileroVad() if full_duplex else None  # separate VAD for barge-in
    assembler = UtteranceAssembler(vad, endpoint, preroll)
    stats: dict[str, PracticeStat] = {}
    history: list[dict[str, str]] = []
    audio_q: queue.Queue = queue.Queue(maxsize=400)
    latencies: list[float] = []
    attempts = hits = 0
    practiced_keys: set[str] = set()

    def _cb(indata, frames, time_info, status):  # PortAudio thread
        try:
            audio_q.put_nowait(audio_io.to_mono(indata).copy())
        except queue.Full:
            pass

    def _flush():
        try:
            while True:
                audio_q.get_nowait()
        except queue.Empty:
            pass

    def _frame16(blk: np.ndarray) -> np.ndarray:
        return audio_io.resample(blk, in_sr, audio_io.ASR_SR)

    out_stream = sd.OutputStream(samplerate=out_sr, channels=1, dtype="float32")
    out_stream.start()

    def play(reply24k: np.ndarray) -> None:
        """Resample 24k->device and play on the persistent stream. Half-duplex
        blocks then flushes stale capture; full-duplex writes in chunks and barges
        in on 3 voiced frames. try/finally guarantees the gate reopens."""
        data = audio_io.resample(reply24k, tts.sr, out_sr)
        gate.begin_playback()
        try:
            if full_duplex:
                _flush()
                vad_mon.reset()
                chunk = max(1, int(0.1 * out_sr))
                voiced = 0
                for i in range(0, len(data), chunk):
                    out_stream.write(data[i:i + chunk])
                    while not audio_q.empty():
                        voiced = voiced + 1 if vad_mon.prob(_frame16(audio_q.get_nowait())) > 0.6 else 0
                    if voiced >= 3:  # user barged in
                        out_stream.abort(); out_stream.start()
                        break
            else:
                out_stream.write(data)  # blocks until played
        finally:
            gate.end_playback(time.monotonic())
            if not full_duplex:
                _flush()  # drop echo / backlog captured while we were speaking

    # background: press Enter to cut off the assistant mid-reply (both modes)
    def _watch_keys():
        for _ in sys.stdin:
            out_stream.abort(); out_stream.start()
    threading.Thread(target=_watch_keys, daemon=True).start()

    def _summary():
        if latencies:
            print(f"\nlatency vad_end->first audio: p50={np.percentile(latencies,50):.2f}s "
                  f"p95={np.percentile(latencies,95):.2f}s over {len(latencies)} turns")
        if attempts:
            print(f"practiced: {hits} hits / {attempts} target-exposures, "
                  f"{len(practiced_keys)} distinct sentences")

    mode = "full-duplex (voice barge-in)" if full_duplex else \
        "half-duplex (press Enter to interrupt)"
    try:
        with sd.InputStream(samplerate=in_sr, channels=1, blocksize=block,
                            dtype="float32", callback=_cb):
            print(f"\nReady. Just start talking. [{mode}]  Say 'stop' or Ctrl-C to quit.\n")
            while True:
                now = time.monotonic()
                blk = audio_q.get()
                if not gate.capture_enabled(now):
                    continue
                utterance = assembler.push(_frame16(blk))
                if utterance is None:
                    continue
                t_end = time.monotonic()
                targets = select_targets(sentences, stats, n=n_targets)
                r = respond(utterance, history, targets, asr=asr, tts=tts,
                            embed=ollama_embed, system_prompt=build_system_prompt(targets))
                if not r.user_text:
                    continue
                print(f"you: {r.user_text}")
                if is_stop(r.user_text):
                    print("ai : Bye! Keep practicing.")
                    _summary(); return 0
                print(f"ai : {r.reply_text}")
                if practiced_on and r.practiced_error:
                    print(f"  (practiced-scoring error: {r.practiced_error})")
                attempts += len(targets)
                for h in r.practiced:
                    hits += 1
                    k = _key(sentences, h.target)
                    practiced_keys.add(k)
                    st = stats.setdefault(k, PracticeStat())
                    st.count += 1; st.last_ts = time.time()
                history += [{"role": "user", "content": r.user_text},
                            {"role": "assistant", "content": r.reply_text}]
                history = history[-12:]  # keep context short
                if r.reply_audio.size:
                    latencies.append(time.monotonic() - t_end)  # vad_end -> first audio
                    play(r.reply_audio)
    except KeyboardInterrupt:
        print("\nBye! Keep practicing.")
        _summary(); return 0
    finally:
        try:
            out_stream.stop(); out_stream.close()
        except Exception:
            pass


def _key(sentences, target_text):
    for s in sentences:
        if s.text == target_text:
            return s.key
    return target_text.lower()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="localvocal", description=__doc__)
    ap.add_argument("--decks", nargs="*", default=None,
                    help="AnkiApp XML deck files (default: data/*.xml)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--voice", default="")
    ap.add_argument("--n-targets", type=int, default=3)
    ap.add_argument("--full-duplex", action="store_true",
                    help="enable voice barge-in (use with headphones/AirPods)")
    ap.add_argument("--smoke", action="store_true",
                    help="run health checks + TTS->ASR round-trip, then exit (no mic)")
    args = ap.parse_args(argv)

    decks = args.decks or sorted(glob.glob("data/*.xml"))
    if not decks:
        print("No decks found. Pass --decks or put AnkiApp XML in data/.", file=sys.stderr)
        return 1

    if args.smoke:
        return smoke(decks, args.model)
    return run_loop(decks, args.model, args.voice, args.n_targets, args.full_duplex)


if __name__ == "__main__":
    raise SystemExit(main())
