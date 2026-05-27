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
from pathlib import Path

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


def _load_models(asr_model: str = "small.en", voice: str = ""):
    from localvocal.asr import WhisperASR
    from localvocal.tts import KokoroTTS

    tts = KokoroTTS(voice=voice) if voice else KokoroTTS()
    return WhisperASR(model_size=asr_model), tts


def _startup(decks: list[str], model: str, voice: str, asr_model: str):
    """Shared startup for both turn modes: load decks, warm LLM/ASR/TTS, probe
    embeddings. Returns (sentences, asr, tts, practiced_on) or None on failure."""
    sentences = load_sentences(decks)
    if not sentences:
        print("No sentences loaded — check --decks paths.", file=sys.stderr)
        return None
    print(f"Loaded {len(sentences)} sentences. Warming up {model} + ASR/TTS "
          f"(first run / cold model can take 30-60s)...")
    try:
        warmup(model=model)
        think_probe(model=model)
        asr, tts = _load_models(asr_model, voice)
        # Warm ASR + TTS first-call (whisper init + Kokoro pipeline build) so the
        # user's FIRST turn isn't an 8s cold path.
        _warm = audio_io.resample(tts.synth("Let's practice English."),
                                  tts.sr, audio_io.ASR_SR)
        asr.transcribe(_warm)
    except Exception as e:
        print(f"Startup failed: {e}", file=sys.stderr)
        return None
    from localvocal.practiced_scorer import ollama_embed
    practiced_on = True
    try:
        ollama_embed(["ping"])
    except Exception as e:
        practiced_on = False
        print(f"  (nomic-embed unreachable: {e} — practiced tracking off)")
    return sentences, asr, tts, practiced_on


def _save_debug(debug_dir, utterance16k, user_text, reply_text, info):
    """Save a turn's raw utterance WAV + transcript for diagnosing accent/noise."""
    import wave
    from pathlib import Path

    d = Path(debug_dir)
    d.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    pcm = (np.clip(utterance16k, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(d / f"{ts}.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(audio_io.ASR_SR)
        w.writeframes(pcm.tobytes())
    (d / f"{ts}.txt").write_text(f"you: {user_text}\nai : {reply_text}\n{info}\n")


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
             full_duplex: bool, end_silence_ms: int, asr_model: str,
             debug: bool, brief: bool) -> int:  # pragma: no cover - needs a microphone
    import queue

    try:
        import sounddevice as sd
    except Exception as e:
        print(f"sounddevice unavailable ({e}). Install: uv sync --extra audio",
              file=sys.stderr)
        return 1
    from localvocal.practiced_scorer import ollama_embed

    started = _startup(decks, model, voice, asr_model)
    if started is None:
        return 1
    sentences, asr, tts, practiced_on = started
    try:
        vad = SileroVad()
        vad.prob(np.zeros(SileroVad.FRAME, dtype=np.float32)); vad.reset()  # warm
    except Exception as e:
        print(f"VAD load failed: {e} (install: uv sync --extra vad)", file=sys.stderr)
        return 1

    # Capture at the device's NATIVE rate (PortAudio won't always accept 16k),
    # resample each block to 16k once for VAD/ASR. Input device stays open all
    # session (Codex blind-spot #2: never reopen per turn).
    try:
        in_sr = int(sd.query_devices(kind="input")["default_samplerate"])
    except Exception:
        in_sr = audio_io.DEFAULT_DEVICE_SR
    block = max(1, round(FRAME * in_sr / audio_io.ASR_SR))
    try:
        out_sr = int(sd.query_devices(kind="output")["default_samplerate"])
    except Exception:
        out_sr = audio_io.DEFAULT_DEVICE_SR

    debug_dir = "debug" if debug else None
    gate = HalfDuplexGate(full_duplex=full_duplex)
    endpoint = EndpointDetector(EndpointConfig(end_silence_ms=end_silence_ms))
    preroll = audio_io.RingBuffer(int(PREROLL_MS / 1000 * audio_io.ASR_SR))
    vad_mon = SileroVad() if full_duplex else None  # separate VAD for barge-in
    assembler = UtteranceAssembler(vad, endpoint, preroll)
    stats: dict[str, PracticeStat] = {}
    history: list[dict[str, str]] = []
    audio_q: queue.Queue = queue.Queue(maxsize=400)
    latencies: list[float] = []
    attempts = hits = 0
    practiced_keys: set[str] = set()
    perr_shown = False

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

    # out_stream is touched ONLY from the main thread (here). The Enter watcher
    # just sets this event; play() checks it between chunks. No cross-thread
    # stream access -> no race.
    interrupt = threading.Event()

    def play(reply24k: np.ndarray) -> None:
        """Resample 24k->device, play in 100ms chunks. Interruptible between
        chunks by the Enter key (both modes) or voice barge-in (full-duplex).
        try/finally guarantees the capture gate reopens."""
        data = audio_io.resample(reply24k, tts.sr, out_sr)
        chunk = max(1, int(0.1 * out_sr))
        gate.begin_playback()
        interrupt.clear()
        if full_duplex:
            _flush()
            vad_mon.reset()
        voiced = 0
        try:
            for i in range(0, len(data), chunk):
                if interrupt.is_set():
                    break
                out_stream.write(data[i:i + chunk])
                if full_duplex:  # watch the mic for a barge-in while we speak
                    while not audio_q.empty():
                        f16 = _frame16(audio_q.get_nowait())
                        preroll.push(f16)  # keep onset so barge-in isn't clipped
                        voiced = voiced + 1 if vad_mon.prob(f16) > 0.6 else 0
                    if voiced >= 3:
                        interrupt.set()
        finally:
            if interrupt.is_set():
                out_stream.abort(); out_stream.start()  # flush buffered tail (main thread only)
            gate.end_playback(time.monotonic())
            if not full_duplex:
                _flush()  # drop echo / backlog captured while we were speaking

    # background: press Enter to cut off the assistant mid-reply (both modes).
    # Only sets the event; never touches out_stream (avoids a cross-thread race).
    def _watch_keys():
        for _ in sys.stdin:
            interrupt.set()
    threading.Thread(target=_watch_keys, daemon=True).start()

    def _summary():
        if latencies:
            print(f"\nlatency vad_end->first audio: p50={np.percentile(latencies,50):.2f}s "
                  f"p95={np.percentile(latencies,95):.2f}s over {len(latencies)} turns")
        if not practiced_on:
            print("practiced: tracking was unavailable (nomic-embed down) this session")
        elif attempts:
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
                            embed=ollama_embed if practiced_on else None,
                            system_prompt=build_system_prompt(targets, brief=brief))
                if not r.user_text:
                    continue
                print(f"you: {r.user_text}")
                if is_stop(r.user_text):
                    print("ai : Bye! Keep practicing.")
                    _summary(); return 0
                print(f"ai : {r.reply_text}")
                if debug_dir:
                    _save_debug(debug_dir, utterance, r.user_text, r.reply_text,
                                f"asr={r.asr_s:.2f}s ttft={r.ttft_s}s tts={r.tts_s:.2f}s "
                                f"practiced={[round(h.similarity, 2) for h in r.practiced]}")
                if r.practiced_error:  # latch tracking off so we don't block every turn
                    if not perr_shown:
                        print(f"  (practiced-scoring error: {r.practiced_error}; tracking off)")
                        perr_shown = True
                    practiced_on = False
                if practiced_on:
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


def run_manual(decks: list[str], model: str, voice: str, n_targets: int,
               asr_model: str, debug: bool, brief: bool) -> int:  # pragma: no cover - needs a microphone
    """Manual turns: you press Enter to start, speak as long as you want (pause to
    think freely — no VAD time pressure), press Enter to send. Best for a thinking
    non-native speaker. Quit with 's'+Enter or Ctrl-C."""
    import queue

    try:
        import sounddevice as sd
    except Exception as e:
        print(f"sounddevice unavailable ({e}). Install: uv sync --extra audio", file=sys.stderr)
        return 1
    from localvocal.practiced_scorer import ollama_embed

    started = _startup(decks, model, voice, asr_model)
    if started is None:
        return 1
    sentences, asr, tts, practiced_on = started
    try:
        in_sr = int(sd.query_devices(kind="input")["default_samplerate"])
        out_sr = int(sd.query_devices(kind="output")["default_samplerate"])
    except Exception:
        in_sr = out_sr = audio_io.DEFAULT_DEVICE_SR
    block = max(1, round(FRAME * in_sr / audio_io.ASR_SR))
    debug_dir = "debug" if debug else None

    audio_q: queue.Queue = queue.Queue(maxsize=8000)
    recording = threading.Event()

    def _cb(indata, frames, time_info, status):
        if recording.is_set():  # only enqueue during a recording window -> clean boundary
            try:
                audio_q.put_nowait(audio_io.to_mono(indata).copy())
            except queue.Full:
                pass

    out_stream = sd.OutputStream(samplerate=out_sr, channels=1, dtype="float32")
    out_stream.start()
    stats: dict[str, PracticeStat] = {}
    history: list[dict[str, str]] = []
    attempts = hits = 0
    pkeys: set[str] = set()
    print("\nManual turns: Enter to speak, Enter again to send. Pause to think freely. "
          "Say or type 's'+Enter to quit.\n")
    try:
        with sd.InputStream(samplerate=in_sr, channels=1, blocksize=block,
                            dtype="float32", callback=_cb):
            while True:
                if input("[Enter to start speaking] ").strip().lower() == "s":
                    break
                while not audio_q.empty():  # clear any leftover
                    audio_q.get_nowait()
                recording.set()
                input("[recording... Enter when you're done] ")
                recording.clear()
                blocks = []
                try:
                    while True:
                        blocks.append(audio_q.get_nowait())
                except queue.Empty:
                    pass
                if not blocks:
                    print("(heard nothing — try again)\n"); continue
                # resample the whole utterance ONCE (no per-block boundary artifacts)
                utterance = audio_io.resample(np.concatenate(blocks), in_sr, audio_io.ASR_SR)
                targets = select_targets(sentences, stats, n=n_targets)
                r = respond(utterance, history, targets, asr=asr, tts=tts,
                            embed=ollama_embed if practiced_on else None,
                            system_prompt=build_system_prompt(targets, brief=brief))
                if r.practiced_error:
                    practiced_on = False  # latch off; don't block future turns
                if not r.user_text:
                    print("(heard nothing — try again)\n"); continue
                print(f"you: {r.user_text}")
                if is_stop(r.user_text):  # spoken "stop"/"goodbye" ends the session
                    print("ai : Bye! Keep practicing.")
                    break
                print(f"ai : {r.reply_text}")
                if debug_dir:
                    _save_debug(debug_dir, utterance, r.user_text, r.reply_text,
                                f"asr={r.asr_s:.2f}s tts={r.tts_s:.2f}s")
                if practiced_on:
                    attempts += len(targets)
                    for h in r.practiced:
                        hits += 1; k = _key(sentences, h.target); pkeys.add(k)
                        st = stats.setdefault(k, PracticeStat()); st.count += 1; st.last_ts = time.time()
                history += [{"role": "user", "content": r.user_text},
                            {"role": "assistant", "content": r.reply_text}]
                history = history[-12:]
                if r.reply_audio.size:
                    out_stream.write(audio_io.resample(r.reply_audio, tts.sr, out_sr))
                print()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        try:
            out_stream.stop(); out_stream.close()
        except Exception:
            pass
    print("\nBye! Keep practicing.")
    if practiced_on and attempts:
        print(f"practiced: {hits} hits / {attempts} target-exposures, {len(pkeys)} distinct")
    return 0


RECALL_NUM_PREDICT = 120  # coach replies stay short (their turn to talk, not the AI's)


def _startup_recall(model: str, voice: str, asr_model: str):
    """Startup for markdown-recall mode: warm LLM/ASR/TTS + require embeddings
    (coverage scoring is the whole point here). Returns (asr, tts, embed) or None."""
    print(f"Warming up {model} + ASR/TTS (first run / cold model can take 30-60s)...")
    try:
        warmup(model=model)
        think_probe(model=model)
        asr, tts = _load_models(asr_model, voice)
        _warm = audio_io.resample(tts.synth("Let's begin."), tts.sr, audio_io.ASR_SR)
        asr.transcribe(_warm)
    except Exception as e:
        print(f"Startup failed: {e}", file=sys.stderr)
        return None
    from localvocal.embeddings import ollama_embed
    try:
        ollama_embed(["ping"])
    except Exception as e:
        print(f"nomic-embed unreachable: {e}. Recall scoring needs it — "
              f"start Ollama / `ollama pull nomic-embed-text`.", file=sys.stderr)
        return None
    return asr, tts, ollama_embed


def run_recall(path: str, model: str, voice: str, asr_model: str,
               extract_model: str, debug: bool) -> int:  # pragma: no cover - needs a microphone
    """Markdown-recall mode: walk an agenda extracted from a doc, interviewing the
    user to recall each item FROM MEMORY (manual turns — recall needs thinking time,
    no VAD pressure). Separate from the English path; shares only speak_turn()."""
    import functools
    import queue

    try:
        import sounddevice as sd
    except Exception as e:
        print(f"sounddevice unavailable ({e}). Install: uv sync --extra audio", file=sys.stderr)
        return 1
    from localvocal.coverage import CoverageTracker, has_substance
    from localvocal.markdown_extractor import load_markdown
    from localvocal.pipeline import speak_turn
    from localvocal.recall_session import RecallSession

    started = _startup_recall(model, voice, asr_model)
    if started is None:
        return 1
    asr, tts, embed = started

    print(f"Extracting recall items from {path} (one-time, {extract_model})...")
    try:
        items = load_markdown(path, model=extract_model)
    except Exception as e:
        print(f"Could not read/extract {path}: {e}", file=sys.stderr)
        return 1
    items = [it for it in items if any(has_substance(p) for p in it.expected_points)]
    if not items:
        print("No recallable content found in that document.", file=sys.stderr)
        return 1
    agenda = Path(path).name + ".recall.json"
    print(f"Loaded {len(items)} recall items. Agenda written to {agenda} (C9: "
          f"open it to see exactly what will be drilled).")  # transparency

    session = RecallSession(items, tracker=CoverageTracker(items, embed=embed),
                            source_title=items[0].source_title or Path(path).stem)
    coach_chat = functools.partial(chat, model=model)

    try:
        in_sr = int(sd.query_devices(kind="input")["default_samplerate"])
        out_sr = int(sd.query_devices(kind="output")["default_samplerate"])
    except Exception:
        in_sr = out_sr = audio_io.DEFAULT_DEVICE_SR
    block = max(1, round(FRAME * in_sr / audio_io.ASR_SR))
    debug_dir = "recall-debug" if debug else None

    audio_q: queue.Queue = queue.Queue(maxsize=8000)
    recording = threading.Event()

    def _cb(indata, frames, time_info, status):
        if recording.is_set():
            try:
                audio_q.put_nowait(audio_io.to_mono(indata).copy())
            except queue.Full:
                pass

    out_stream = sd.OutputStream(samplerate=out_sr, channels=1, dtype="float32")
    out_stream.start()

    def say(text: str) -> None:
        from localvocal.sentence_chunker import chunk_sentences
        from localvocal.text_sanitize import sanitize_for_tts
        for c in chunk_sentences(sanitize_for_tts(text)):
            out_stream.write(audio_io.resample(tts.synth(c), tts.sr, out_sr))

    def _record_utterance() -> np.ndarray | None:
        while not audio_q.empty():
            audio_q.get_nowait()
        recording.set()
        input("[recording... Enter when you're done] ")
        recording.clear()
        blocks = []
        try:
            while True:
                blocks.append(audio_q.get_nowait())
        except queue.Empty:
            pass
        if not blocks:
            return None
        return audio_io.resample(np.concatenate(blocks), in_sr, audio_io.ASR_SR)

    history: list[dict[str, str]] = []
    print("\nRecall mode: Enter to speak your answer, Enter again to send. "
          "Pause to think freely. Say or type 's'+Enter to quit.\n")
    opening = session.opening_line()
    print(f"ai : {opening}")
    say(opening)
    try:
        with sd.InputStream(samplerate=in_sr, channels=1, blocksize=block,
                            dtype="float32", callback=_cb):
            while not session.done:
                if input("[Enter to answer] ").strip().lower() == "s":
                    break
                utterance = _record_utterance()
                if utterance is None:
                    print("(heard nothing — try again)\n"); continue
                prompt = session.coach_prompt()
                st = speak_turn(utterance, history, asr=asr, tts=tts,
                                system_prompt=prompt, chat_fn=coach_chat,
                                num_predict=RECALL_NUM_PREDICT)
                if not st.user_text:
                    print("(heard nothing — try again)\n"); continue
                print(f"you: {st.user_text}")
                if is_stop(st.user_text):
                    break
                out = session.record(st.user_text)
                cov = out.coverage
                hit = sum(1 for b in cov.bullets if b.status == "hit") if cov else 0
                tot = len(cov.bullets) if cov else 0
                n, total = session.progress()
                tag = "✓ complete" if (cov and cov.complete) else f"{hit}/{tot} points"
                flags = (" [hint]" if out.gave_hint else "") + (" →next" if out.advanced else "")
                print(f"ai : {st.reply_text}")
                print(f"     [{n}/{total}] {tag}{flags}")
                if debug_dir:
                    _save_debug(debug_dir, utterance, st.user_text, st.reply_text,
                                f"item={cov.item_key if cov else '-'} {tag}{flags} "
                                f"asr={st.asr_s:.2f}s tts={st.tts_s:.2f}s")
                history += [{"role": "user", "content": st.user_text},
                            {"role": "assistant", "content": st.reply_text}]
                history = history[-12:]
                if st.reply_audio.size:
                    out_stream.write(audio_io.resample(st.reply_audio, tts.sr, out_sr))
                print()
            closing = "That's the whole agenda — nicely done." if session.done \
                else "Let's stop there."
            print(f"ai : {closing}")
            say(closing)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        try:
            out_stream.stop(); out_stream.close()
        except Exception:
            pass
    print(f"\nRecall summary: {session.summary()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="localvocal", description=__doc__)
    ap.add_argument("--content", choices=["english", "markdown"], default="english",
                    help="english = Anki conversation practice; markdown = recall a doc from memory")
    ap.add_argument("--path", default=None,
                    help="absolute path to the markdown file to recall (with --content markdown)")
    ap.add_argument("--extract-model", default=None,
                    help="LLM for one-time markdown->agenda extraction (default: accuracy model)")
    ap.add_argument("--decks", nargs="*", default=None,
                    help="AnkiApp XML deck files (default: data/*.xml)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--voice", default="")
    ap.add_argument("--n-targets", type=int, default=3)
    ap.add_argument("--full-duplex", action="store_true",
                    help="enable voice barge-in (use with headphones/AirPods)")
    ap.add_argument("--manual-turns", action="store_true",
                    help="press Enter to start/stop each turn — no VAD time pressure")
    ap.add_argument("--end-silence-ms", type=int, default=1000,
                    help="silence (ms) that ends your turn in auto mode; raise for more thinking time")
    ap.add_argument("--asr-model", default="small.en",
                    help="ASR model: small.en (fast) | medium.en | distil-large-v3 (accurate, slower)")
    ap.add_argument("--brief", action="store_true",
                    help="one-sentence replies — snappier (~1s less) and you talk more")
    ap.add_argument("--debug", action="store_true",
                    help="save each turn's audio + transcript to debug/ for diagnosis")
    ap.add_argument("--smoke", action="store_true",
                    help="run health checks + TTS->ASR round-trip, then exit (no mic)")
    args = ap.parse_args(argv)

    if args.content == "markdown":
        if not args.path:
            print("--content markdown needs --path /abs/file.md", file=sys.stderr)
            return 1
        if not Path(args.path).is_file():
            print(f"No such file: {args.path}", file=sys.stderr)
            return 1
        from localvocal.markdown_extractor import EXTRACT_MODEL
        return run_recall(args.path, args.model, args.voice, args.asr_model,
                          args.extract_model or EXTRACT_MODEL, args.debug)

    decks = args.decks or sorted(glob.glob("data/*.xml"))
    if not decks:
        print("No decks found. Pass --decks or put AnkiApp XML in data/.", file=sys.stderr)
        return 1

    if args.smoke:
        return smoke(decks, args.model)
    if args.manual_turns:
        return run_manual(decks, args.model, args.voice, args.n_targets,
                          args.asr_model, args.debug, args.brief)
    return run_loop(decks, args.model, args.voice, args.n_targets, args.full_duplex,
                    args.end_silence_ms, args.asr_model, args.debug, args.brief)


if __name__ == "__main__":
    raise SystemExit(main())
