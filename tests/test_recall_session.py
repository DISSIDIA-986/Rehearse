"""RecallSession state-machine tests. A FakeTracker isolates progression logic
from embeddings; the key invariant is that expected_points NEVER leak to the coach."""

from collections import defaultdict

from rehearse.coverage import BulletScore, ItemCoverage, Summary
from rehearse.practice_item import PracticeItem
from rehearse.recall_session import RecallSession


class FakeTracker:
    """Deterministic, transcript-driven coverage: 'DONE' -> all bullets hit,
    'MORE' -> one more bullet hit (progress), anything else -> no progress."""

    def __init__(self, items):
        self.items = list(items)
        self._hits = defaultdict(int)

    def score(self, item, text):
        n = len(item.expected_points)
        if "DONE" in text:
            h = n
        elif "MORE" in text:
            h = min(n, self._hits[item.key] + 1)
        else:
            h = self._hits[item.key]
        self._hits[item.key] = h
        bullets = [BulletScore(p, 1.0, 0, 0, "hit" if i < h else "miss")
                   for i, p in enumerate(item.expected_points)]
        return ItemCoverage(item.key, item.section, text, bullets)

    def summary(self):
        tot = sum(len(it.expected_points) for it in self.items)
        hit = sum(self._hits[it.key] for it in self.items)
        return Summary(tot, hit, len(self.items), len(self.items), [])


def _items():
    return [
        PracticeItem(id="q0", prompt="Tell me about the pricing engine",
                     expected_points=["SECRETALPHA", "SECRETBETA"],
                     support_snippets=["Think NPV and ACPI"], section="S0",
                     source_title="resume.md"),
        PracticeItem(id="q1", prompt="How did you do risk control?",
                     expected_points=["SECRETGAMMA"], section="S1", source_title="resume.md"),
    ]


def _sess():
    items = _items()
    return RecallSession(items, tracker=FakeTracker(items), source_title="resume.md")


def test_opening_introduces_first_question():
    s = _sess()
    line = s.opening_line()
    assert "pricing engine" in line and "resume.md" in line
    assert 0 in s._asked


def test_coach_prompt_never_leaks_expected_points():
    s = _sess()
    s.opening_line()
    seen = [s.coach_prompt()]
    s.record("MORE")
    seen.append(s.coach_prompt())
    s.record("DONE")            # complete item0 -> advance to item1
    seen.append(s.coach_prompt())  # move_on / ask item1
    s.record("DONE")
    blob = "\n".join(seen)
    for secret in ("SECRETALPHA", "SECRETBETA", "SECRETGAMMA"):
        assert secret not in blob


def test_first_item_probes_not_reasks_after_opening():
    s = _sess()
    s.opening_line()
    p = s.coach_prompt()
    assert "deeper" in p or "what ELSE" in p  # probe wording, not the ask template


def test_complete_advances_to_next_item():
    s = _sess()
    s.opening_line()
    s.coach_prompt()
    out = s.record("DONE")
    assert out.advanced and out.coverage.complete
    assert s.current.id == "q1"


def test_stall_offers_support_then_gives_up():
    s = _sess()
    s.opening_line()
    s.coach_prompt()
    assert not s.record("nope").gave_hint          # stall=1
    out = s.record("still nothing")                # stall=2 -> hint queued
    assert out.gave_hint and not out.advanced
    assert "Think NPV and ACPI" in s.coach_prompt()  # hint surfaced as data
    out = s.record("blank")                         # stall=3 -> move on
    assert out.advanced and s.current.id == "q1"


def test_item_without_support_moves_on_at_stall():
    items = [PracticeItem(id="x", prompt="q?", expected_points=["SECRET"], section="S")]
    s = RecallSession(items, tracker=FakeTracker(items))
    s.coach_prompt()
    s.record("nope")                  # stall=1
    out = s.record("nope")            # stall=2, no support -> give up
    assert out.advanced and s.done


def test_progress_and_done():
    s = _sess()
    assert s.progress() == (1, 2)
    s.opening_line(); s.coach_prompt(); s.record("DONE")
    assert s.progress() == (2, 2) and not s.done
    s.coach_prompt(); s.record("DONE")
    assert s.done
    summ = s.summary()
    assert summ.hit_bullets == 3 and summ.total_bullets == 3


def test_empty_agenda_is_done():
    s = RecallSession([], tracker=FakeTracker([]))
    assert s.done
    assert "nothing to recall" in s.opening_line()
