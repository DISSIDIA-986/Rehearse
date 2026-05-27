"""Duplex policy gate (D2).

Half-duplex (default, for the DJI mic + Mac speaker): mute mic capture while the
assistant is speaking, plus a short guard after playback so the speaker's tail /
room reverb doesn't trip the VAD. This is what kills the open-mic echo BLOCKER
without AEC.

Full-duplex (--full-duplex, for AirPods where the in-ear output never bleeds into
the mic): always listen, enabling voice barge-in.

Keyboard interrupt is orthogonal and handled by the loop (works in both modes).
This class is pure timing logic so it can be unit-tested without audio hardware.
"""

from __future__ import annotations

DEFAULT_GUARD_MS = 150


class HalfDuplexGate:
    def __init__(self, full_duplex: bool = False, guard_ms: int = DEFAULT_GUARD_MS):
        self.full_duplex = full_duplex
        self.guard_ms = guard_ms
        self._speaking = False
        self._guard_until = 0.0  # monotonic seconds

    def begin_playback(self) -> None:
        self._speaking = True

    def end_playback(self, now: float) -> None:
        self._speaking = False
        self._guard_until = now + self.guard_ms / 1000.0

    def capture_enabled(self, now: float) -> bool:
        """Should the mic be feeding the VAD right now?"""
        if self.full_duplex:
            return True  # AirPods: listen always (barge-in)
        if self._speaking:
            return False  # assistant talking -> mic muted (no echo)
        return now >= self._guard_until  # post-playback guard window
