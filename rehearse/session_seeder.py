"""Pick the next 2-4 target sentences to weave into a session (D3 / spaced rep).

Least-practiced first: order by (practice_count asc, last_practiced asc). When
everything has been practiced, the longest-untouched still come first. Pure
function over a stats mapping so it is trivially testable; the persistent store
(SQLite) is wired in a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass

from rehearse.anki_loader import Sentence


@dataclass
class PracticeStat:
    count: int = 0
    last_ts: float = 0.0  # epoch seconds of last practice
    unaided_count: int = 0  # T-P2-2b shadow signal — NOT used by select_targets (yet)


def select_targets(
    sentences: list[Sentence],
    stats: dict[str, PracticeStat] | None = None,
    n: int = 3,
) -> list[Sentence]:
    """Return up to n least-practiced sentences (2-4 recommended per turn)."""
    if not sentences or n <= 0:
        return []
    stats = stats or {}
    n = min(n, len(sentences))
    default = PracticeStat()
    ordered = sorted(
        sentences,
        key=lambda s: (
            stats.get(s.key, default).count,
            stats.get(s.key, default).last_ts,
        ),
    )
    return ordered[:n]
