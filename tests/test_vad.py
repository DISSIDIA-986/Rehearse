from rehearse.vad import EndpointConfig, EndpointDetector, VadState


def _cfg():
    # frame=32ms, start after 90ms (3 frames), end after 300ms silence (10 frames)
    return EndpointConfig(
        threshold=0.5, frame_ms=32, start_voiced_ms=90,
        end_silence_ms=300, max_utterance_ms=8000,
    )


def test_starts_after_enough_voiced_frames():
    d = EndpointDetector(_cfg())
    assert d.update(0.9) is None      # 32ms
    assert d.update(0.9) is None      # 64ms
    assert d.update(0.9) == "start"   # 96ms >= 90
    assert d.state is VadState.SPEAKING


def test_brief_noise_does_not_start():
    d = EndpointDetector(_cfg())
    assert d.update(0.9) is None      # one voiced frame
    assert d.update(0.1) is None      # silence resets the voiced run
    assert d.update(0.9) is None
    assert d.update(0.1) is None
    assert d.state is VadState.IDLE


def test_ends_after_trailing_silence():
    d = EndpointDetector(_cfg())
    for _ in range(3):
        d.update(0.9)  # -> start
    assert d.state is VadState.SPEAKING
    ev = None
    for _ in range(10):  # 10 * 32 = 320ms >= 300
        ev = d.update(0.0)
    assert ev == "end"
    assert d.state is VadState.ENDED


def test_silence_resets_within_speech():
    d = EndpointDetector(_cfg())
    for _ in range(3):
        d.update(0.9)  # start
    for _ in range(5):
        d.update(0.0)  # 160ms silence (< 300)
    assert d.state is VadState.SPEAKING
    d.update(0.9)  # voiced resets silence counter
    for _ in range(9):
        assert d.update(0.0) != "end"  # 288ms < 300 after reset


def test_max_utterance_cap_forces_end():
    cfg = EndpointConfig(frame_ms=32, start_voiced_ms=32, end_silence_ms=10_000,
                         max_utterance_ms=320)
    d = EndpointDetector(cfg)
    d.update(0.9)  # start immediately (start_voiced_ms=32)
    ev = None
    for _ in range(20):
        ev = d.update(0.9) or ev  # keep talking; never silent
    assert ev == "end"  # capped


def test_default_endpoint_is_patient():
    # Deliberately patient defaults (raised from 300ms/8s after live feedback that
    # 300ms cut off a non-native speaker mid-thought). Pinned so it's not silent drift.
    c = EndpointConfig()
    assert c.end_silence_ms == 1000
    assert c.max_utterance_ms == 20_000


def test_reset():
    d = EndpointDetector(_cfg())
    for _ in range(3):
        d.update(0.9)
    d.reset()
    assert d.state is VadState.IDLE
