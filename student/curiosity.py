"""Per-skill learning progress: the curiosity signal that replaces RND novelty."""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque

from world.tasks import MAX_DIFFICULTY, MIN_DIFFICULTY


class ProgressTracker:
    """Track graded outcomes per skill and per (skill, difficulty).

    Learning progress (LP) for a skill is the recent success rate minus the rate
    in the preceding window, over the last ``window`` outcomes. Practice favours
    skills with large |LP| (still changing), so mastered and hopeless skills both
    fade once their success rate stops moving.
    """

    def __init__(
        self, skills: tuple[str, ...], window: int = 16, frontier: float = 0.7
    ):
        if window < 4 or window % 2:
            raise ValueError("window must be an even number >= 4")
        self.skills = skills
        self.window = window
        self.frontier = frontier
        self.history = {s: deque(maxlen=window) for s in skills}
        self.counts: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])

    def record(self, skill: str, difficulty: int, success: bool) -> None:
        self.history[skill].append(bool(success))
        cell = self.counts[(skill, difficulty)]
        cell[0] += bool(success)
        cell[1] += 1

    def success_rate(self, skill: str) -> float:
        """Smoothed recent success rate (0.5 before any evidence)."""
        h = self.history[skill]
        return (sum(h) + 1) / (len(h) + 2)

    def estimate(self, skill: str, difficulty: int) -> tuple[float, int]:
        """Smoothed success estimate and attempt count for one cell."""
        wins, n = self.counts[(skill, difficulty)]
        return (wins + 1) / (n + 2), n

    def progress(self, skill: str) -> float | None:
        """Recent-minus-older success rate, or None until a full window exists."""
        h = list(self.history[skill])
        if len(h) < self.window:
            return None
        half = self.window // 2
        return sum(h[half:]) / half - sum(h[:half]) / half

    def practice_weights(self, floor: float = 0.05) -> dict[str, float]:
        """Unmeasured skills get maximal weight so every skill is visited."""
        weights = {}
        for skill in self.skills:
            lp = self.progress(skill)
            weights[skill] = 1.0 if lp is None else abs(lp) + floor
        return weights

    def choose_practice_skill(self, rng: random.Random) -> str:
        weights = self.practice_weights()
        return rng.choices(list(weights), weights=list(weights.values()))[0]

    def frontier_difficulty(self, skill: str) -> int:
        """Lowest difficulty whose estimated success is below the frontier."""
        for difficulty in range(MIN_DIFFICULTY, MAX_DIFFICULTY + 1):
            if self.estimate(skill, difficulty)[0] < self.frontier:
                return difficulty
        return MAX_DIFFICULTY

    def job_value(
        self, skill: str, difficulty: int, pay: float, bonus: float = 0.5
    ) -> float:
        """Optimistic expected pay used to pick jobs from the board."""
        p, n = self.estimate(skill, difficulty)
        return min(1.0, p + bonus / math.sqrt(n + 1)) * pay

    def plateaued(self, skill: str, eps: float = 0.1) -> bool:
        lp = self.progress(skill)
        return lp is not None and abs(lp) < eps
