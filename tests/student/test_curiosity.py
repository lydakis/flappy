"""Learning-progress signal and practice selection."""

import random

from student.curiosity import ProgressTracker


def test_progress_is_recent_minus_older_success_and_needs_a_full_window():
    tracker = ProgressTracker(("a", "b"), window=8)
    for success in [0, 0, 0, 0, 1, 1, 1, 0]:
        tracker.record("a", 1, bool(success))
    assert tracker.progress("a") == 0.75
    assert tracker.progress("b") is None
    for _ in range(8):
        tracker.record("b", 1, True)
    assert tracker.progress("b") == 0 and tracker.plateaued("b")


def test_practice_prefers_changing_skills_over_mastered_ones():
    tracker = ProgressTracker(("learning", "mastered"), window=8)
    for success in [0, 0, 0, 0, 1, 1, 1, 1]:
        tracker.record("learning", 1, bool(success))
    for _ in range(8):
        tracker.record("mastered", 1, True)
    rng = random.Random(0)
    picks = [tracker.choose_practice_skill(rng) for _ in range(500)]
    assert picks.count("learning") > 0.9 * len(picks)


def test_frontier_moves_up_as_lower_levels_are_mastered():
    tracker = ProgressTracker(("a",), window=4)
    assert tracker.frontier_difficulty("a") == 1
    for _ in range(10):
        tracker.record("a", 1, True)
    assert tracker.frontier_difficulty("a") == 2
    # Unseen high-pay jobs carry an optimism bonus over a known-mediocre cell.
    assert tracker.job_value("a", 5, 12.0) > tracker.job_value("a", 1, 1.0)
