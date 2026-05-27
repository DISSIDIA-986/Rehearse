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
    import sounddevice as sd

    from localvocal.tts import KokoroTTS

    sentences = load_sentences(decks)
    if not sentences:
        print("No sentences loaded — check --decks paths.", file=sys.stderr)
        return 1
    print(f"Loaded {len(sentences)} sentences. Warming up {model}...")
    warmup(model=model)
    think_probe(model=model)
    asr, tts = _load_models()
    if voice:
        tts = KokoroTTS(voice=voice)
    vad = SileroVad()
    gate = HalfDuplexGate(full_duplex=full_duplex)
    endpoint = EndpointDetector(EndpointConfig())
    preroll = audio_io.RingBuffer(int(PREROLL_MS / 1000 * audio_io.ASR_SR))
    stats: dict[str, PracticeStat] = {}
    history: list[dict[str, str]] = []

    # background: press Enter to cut off the assistant mid-reply (works both modes)
    interrupt = threading.Event()
    def _watch_keys():
        for _ in sys.stdin:
            interrupt.set(); sd.stop()
    threading.Thread(target=_watch_keys, daemon=True).start()

    print(f"\nReady. Just start talking. {'(full-duplex)' if full_duplex else '(half-duplex; press Enter to interrupt)'}")
    print("Ctrl-C to quit.\n")

    def play(audio: np.ndarray):
        gate.begin_playback()
        sd.play(audio, tts.sr)
        sd.wait()
        gate.end_playback(time.monotonic())

    collected: list[np.ndarray] = []
    try:
        with sd.InputStream(samplerate=audio_io.ASR_SR, channels=1,
                            blocksize=FRAME, dtype="float32") as stream:
            while True:
                block, _ = stream.read(FRAME)
                frame = audio_io.to_mono(block)
                now = time.monotonic()
                if not gate.capture_enabled(now):
                    continue
                preroll.push(frame)
                ev = endpoint.update(vad.prob(frame))
                if ev == "start":
                    collected = [preroll.get()]  # prepend pre-roll (no onset clip)
                elif endpoint.state.value == "speaking":
                    collected.append(frame)
                elif ev == "end":
                    collected.append(frame)
                    utterance = np.concatenate(collected)
                    collected = []
                    endpoint.reset(); vad.reset()
                    targets = select_targets(sentences, stats, n=n_targets)
                    sys_prompt = build_system_prompt(targets)
                    r = respond(utterance, history, targets, asr=asr, tts=tts,
                                system_prompt=sys_prompt)
                    if not r.user_text:
                        continue
                    print(f"you: {r.user_text}")
                    print(f"ai : {r.reply_text}")
                    for h in r.practiced:
                        st = stats.setdefault(_key(sentences, h.target), PracticeStat())
                        st.count += 1; st.last_ts = time.time()
                    history += [{"role": "user", "content": r.user_text},
                                {"role": "assistant", "content": r.reply_text}]
                    history = history[-12:]  # keep context short
                    interrupt.clear()
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
