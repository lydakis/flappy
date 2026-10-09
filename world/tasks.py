"""Shared task, grade and family types for the text-native world."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Protocol

MIN_DIFFICULTY = 1
MAX_DIFFICULTY = 5


@dataclass(frozen=True)
class Task:
    """One gradable unit of work. ``hidden`` is grader-only and never shown."""

    task_id: str
    family: str
    skill: str
    difficulty: int
    prompt: str
    hidden: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class Grade:
    """Outcome of grading one answer: pass/fail, partial score and feedback."""

    passed: bool
    score: float
    feedback: str


class TaskFamily(Protocol):
    """A family owns some skills, samples tasks for them and grades answers."""

    name: str
    skills: tuple[str, ...]

    def sample(self, skill: str, difficulty: int, rng: random.Random) -> Task: ...

    def grade(self, task: Task, answer: str) -> Grade: ...


def check_difficulty(difficulty: int) -> None:
    if not MIN_DIFFICULTY <= difficulty <= MAX_DIFFICULTY:
        raise ValueError(f"difficulty must be in 1..5, got {difficulty}")


def new_task_id(rng: random.Random, skill: str, difficulty: int) -> str:
    return f"{skill}-d{difficulty}-{rng.getrandbits(32):08x}"
