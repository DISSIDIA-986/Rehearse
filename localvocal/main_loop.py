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
from localvocal.pipeline import respond
from localvocal.prompt_builder import build_system_prompt
from localvocal.session_seeder import PracticeStat, select_targets
from localvocal.vad import EndpointConfig, EndpointDetector, SileroVad, VadState

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
        print(f"[{'ok' if match else 'WARN'}] TTS->ASR round-trip: {heard!r}")
        ok = ok and bool(heard)
    except Exception as e:
        print(f"[FAIL] audio round-trip: {e}"); ok = False

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

    gate = HalfDuplexGate(full_duplex=full_duplex)
    endpoint = EndpointDetector(EndpointConfig())
    preroll = audio_io.RingBuffer(int(PREROLL_MS / 1000 * audio_io.ASR_SR))
    stats: dict[str, PracticeStat] = {}
    history: list[dict[str, str]] = []
    audio_q: queue.Queue = queue.Queue(maxsize=400)

    def _cb(indata, frames, time_info, status):  # runs on PortAudio thread
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

    def play(reply: np.ndarray) -> None:
        """Play the reply. Half-duplex: block + flush stale capture afterward.
        Full-duplex: non-blocking + watch for voice barge-in. try/finally so the
        capture gate is ALWAYS reopened, even on interrupt/device error."""
        gate.begin_playback()
        try:
            sd.play(reply, tts.sr)
            if full_duplex:
                voiced = 0
                while sd.get_stream().active:
                    try:
                        blk = audio_q.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    voiced = voiced + 1 if vad.prob(_frame16(blk)) > 0.6 else 0
                    if voiced >= 3:  # ~3 voiced frames -> user is talking, barge in
                        sd.stop()
                        break
            else:
                sd.wait()
        finally:
            gate.end_playback(time.monotonic())
            if not full_duplex:
                _flush()  # drop echo / backlog captured while we were speaking

    # background: press Enter to cut off the assistant mid-reply (both modes)
    def _watch_keys():
        for _ in sys.stdin:
            sd.stop()
    threading.Thread(target=_watch_keys, daemon=True).start()

    mode = "full-duplex (voice barge-in)" if full_duplex else \
        "half-duplex (press Enter to interrupt)"
    collected: list[np.ndarray] = []
    try:
        with sd.InputStream(samplerate=in_sr, channels=1, blocksize=block,
                            dtype="float32", callback=_cb):
            print(f"\nReady. Just start talking. [{mode}]  Ctrl-C to quit.\n")
            while True:
                blk = audio_q.get()
                now = time.monotonic()
                if not gate.capture_enabled(now):
                    continue
                frame = _frame16(blk)
                preroll.push(frame)
                ev = endpoint.update(vad.prob(frame))
                if ev == "start":
                    collected = [preroll.get()]  # prepend pre-roll (no onset clip)
                elif endpoint.state is VadState.SPEAKING:
                    collected.append(frame)
                elif ev == "end":
                    collected.append(frame)
                    utterance = np.concatenate(collected)
                    collected = []
                    endpoint.reset(); vad.reset()
                    targets = select_targets(sentences, stats, n=n_targets)
                    r = respond(utterance, history, targets, asr=asr, tts=tts,
                                system_prompt=build_system_prompt(targets))
                    if not r.user_text:
                        continue
                    print(f"you: {r.user_text}")
                    print(f"ai : {r.reply_text}")
                    if r.practiced_error:
                        print(f"  (practiced-scoring unavailable: {r.practiced_error})")
                    for h in r.practiced:
                        st = stats.setdefault(_key(sentences, h.target), PracticeStat())
                        st.count += 1; st.last_ts = time.time()
                    history += [{"role": "user", "content": r.user_text},
                                {"role": "assistant", "content": r.reply_text}]
                    history = history[-12:]  # keep context short
                    if r.reply_audio.size:
                        play(r.reply_audio)
    except KeyboardInterrupt:
        print("\nBye! Keep practicing.")
        return 0


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
