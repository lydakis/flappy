"""Evaluation metrics for FLAPPY."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class EpisodeStats:
    success: bool
    reward: float
    steps: int


def success_rate(episodes: Iterable[EpisodeStats]) -> float:
    stats = list(episodes)
    if not stats:
        return 0.0
    return sum(1 for ep in stats if ep.success) / len(stats)


def average_reward(episodes: Iterable[EpisodeStats]) -> float:
    stats = list(episodes)
    if not stats:
        return 0.0
    return sum(ep.reward for ep in stats) / len(stats)


def steps_to_success(episodes: Iterable[EpisodeStats]) -> float:
    successes = [ep.steps for ep in episodes if ep.success]
    if not successes:
        return 0.0
    return sum(successes) / len(successes)


def normalized_return(episodes: Iterable[EpisodeStats], max_steps: int) -> float:
    stats = list(episodes)
    if not stats or max_steps == 0:
        return 0.0
    total = sum(ep.reward for ep in stats)
    return total / (len(stats) * max_steps)
