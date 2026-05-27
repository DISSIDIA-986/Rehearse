from localvocal.duplex import HalfDuplexGate


def test_half_duplex_mutes_during_playback_and_guard():
    g = HalfDuplexGate(full_duplex=False, guard_ms=150)
    assert g.capture_enabled(0.0)          # idle -> listening
    g.begin_playback()
    assert not g.capture_enabled(1.0)      # assistant speaking -> mic muted
    g.end_playback(1.0)
    assert not g.capture_enabled(1.05)     # within 150ms guard
    assert g.capture_enabled(1.20)         # guard elapsed -> listening again


def test_full_duplex_always_listens():
    g = HalfDuplexGate(full_duplex=True)
    g.begin_playback()
    assert g.capture_enabled(0.0)          # AirPods: barge-in allowed
    g.end_playback(0.0)
    assert g.capture_enabled(0.0)
