"""Real-mic capture probe — validates the live audio path against actual hardware.

NOT shipped (experiment). Captures N seconds from the default input (DJI Mic Mini),
runs it through the real loop_core helpers (io_rates -> capture -> drain_utterance)
and the real ASR. With --round-trip it also runs the full pipeline.speak_turn
(ASR -> MLX coach -> Kokoro TTS) and plays the reply, printing the latency trace.

Use a continuously-playing podcast as the "speech" source:
    uv run python experiments/mic_probe.py --seconds 6
    uv run python experiments/mic_probe.py --round-trip
"""

from __future__ import annotations

import argparse
import queue
import time

import numpy as np
import sounddevice as sd

from rehearse import audio_io
from rehearse.asr import WhisperASR
from rehearse.loop_core import drain_utterance, io_rates


def _capture(seconds: float, in_sr: int, block: int) -> np.ndarray | None:
    q: queue.Queue = queue.Queue(maxsize=8000)

    def cb(indata, frames, t, status):  # PortAudio thread (same shape as the loop)
        try:
            q.put_nowait(audio_io.to_mono(indata).copy())
        except queue.Full:
            pass

    print(f"\n>>> Capturing {seconds:.0f}s from the mic NOW (podcast should be playing)...")
    with sd.InputStream(samplerate=in_sr, channels=1, blocksize=block,
                        dtype="float32", callback=cb):
        time.sleep(seconds)
    return drain_utterance(q, in_sr)  # the real loop helper


def _run_vad(utt: np.ndarray, in_sr: int) -> None:
    """Validate auto-mode endpointing: feed REAL captured speech (then appended
    silence) through real Silero VAD + the EndpointDetector state machine and
    confirm it fires 'start' on speech and 'end' on the speech->silence edge."""
    from rehearse.vad import EndpointConfig, EndpointDetector, SileroVad

    a = utt if in_sr == audio_io.ASR_SR else audio_io.resample(utt, in_sr, audio_io.ASR_SR)
    silence = np.zeros(int(1.5 * audio_io.ASR_SR), dtype=np.float32)  # > end_silence_ms
    stream = np.concatenate([a, silence])

    vad = SileroVad()
    vad.prob(np.zeros(SileroVad.FRAME, dtype=np.float32)); vad.reset()  # warm
    det = EndpointDetector(EndpointConfig())  # default: start 64ms, end 1000ms silence
    F = SileroVad.FRAME
    n_speech_frames = len(a) // F
    probs: list[float] = []
    events: list[tuple] = []
    for i in range(0, len(stream) - F + 1, F):
        p = vad.prob(stream[i:i + F])
        probs.append(p)
        ev = det.update(p)
        if ev:
            events.append((i // F, ev, round(p, 2)))

    sp = probs[:n_speech_frames] or [0.0]
    voiced = sum(1 for p in sp if p >= 0.5) / len(sp)
    print(f"VAD on REAL speech: {len(sp)} frames  voiced={voiced:.0%}  "
          f"mean={float(np.mean(sp)):.2f} max={max(sp):.2f}")
    print(f"VAD on appended silence: mean={float(np.mean(probs[n_speech_frames:] or [0])):.2f}")
    print(f"endpoint events (frame, kind, prob): {events}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--asr-model", default="small.en")
    ap.add_argument("--round-trip", action="store_true",
                    help="also run the full ASR->coach->TTS pipeline and play the reply")
    ap.add_argument("--vad", action="store_true",
                    help="run real Silero VAD + EndpointDetector over the capture "
                         "(+ appended silence) to validate auto-mode endpointing")
    args = ap.parse_args()

    in_sr, out_sr, block = io_rates(sd)
    di = sd.query_devices(kind="input")
    print(f"input: {di['name']}  in_sr={in_sr} out_sr={out_sr} block={block}")

    print(f"Loading ASR ({args.asr_model})...")
    asr = WhisperASR(model_size=args.asr_model)
    asr.transcribe(np.zeros(audio_io.ASR_SR, dtype=np.float32))  # warm

    utt = _capture(args.seconds, in_sr, block)
    if utt is None:
        print("!! captured NOTHING — the mic delivered no audio.")
        return 1
    rms = float(np.sqrt(np.mean(utt ** 2)))
    peak = float(np.max(np.abs(utt))) if utt.size else 0.0
    print(f"captured {len(utt) / audio_io.ASR_SR:.1f}s @16k  rms={rms:.4f} peak={peak:.3f}")
    if rms < 1e-4:
        print("!! near-silence — is the DJI mic actually picking up the podcast?")

    if args.vad:
        _run_vad(utt, in_sr)

    t0 = time.monotonic()
    text = asr.transcribe(utt)
    print(f"ASR [{time.monotonic() - t0:.2f}s]: {text!r}")

    if args.round_trip and text:
        from rehearse.mlx_llm import mlx_warm_and_probe, resolve_coach_chat
        from rehearse.pipeline import speak_turn
        from rehearse.prompt_builder import build_system_prompt
        from rehearse.tts import KokoroTTS

        chat_fn, mid, backend = resolve_coach_chat("auto")
        print(f"\ncoach: {backend} ({mid}) — warming...")
        if backend == "mlx":
            mlx_warm_and_probe(mid)
        tts = KokoroTTS()
        tts.synth("warm")  # warm the TTS pipeline
        st = speak_turn(utt, [], asr=asr, tts=tts,
                        system_prompt=build_system_prompt([], brief=True), chat_fn=chat_fn)
        print(f"you : {st.user_text!r}")
        print(f"ai  : {st.reply_text!r}")
        print(f"trace: asr={st.asr_s:.2f}s ttft={st.ttft_s}s tts_ttfa={st.tts_ttfa_s:.2f}s "
              f"tts_total={st.tts_s:.2f}s")
        if st.reply_audio.size:
            print(">>> playing reply...")
            with sd.OutputStream(samplerate=out_sr, channels=1, dtype="float32") as out:
                out.write(audio_io.resample(st.reply_audio, tts.sr, out_sr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
