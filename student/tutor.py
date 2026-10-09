"""Tutor services the student can buy with credits: hint, worked example, explanation.

The tutor sees only what the student sees (task prompt, its own answer and the
grader's short feedback), never hidden tests or reference answers. Every request
goes through a backend; the real one is ``llm.tutor_ledger.LedgerTutorClient``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from world.tasks import Task

PRICES = {"hint": 1.0, "explanation": 2.0, "worked_example": 3.0}
MAX_ANSWER_CHARS = 1200


class TutorBackend(Protocol):
    def request(self, prompt: str, purpose: str) -> str | None: ...


@dataclass(frozen=True)
class TutorReply:
    kind: str
    text: str | None
    example_task: Task | None = None  # worked examples are on a sibling task
    answer: str | None = None  # the tutor's proposed final answer, if any


def _ascii(text: str, limit: int = MAX_ANSWER_CHARS) -> str:
    return text.encode("ascii", "replace").decode()[:limit]


def parse_answer(text: str) -> str | None:
    """Return the text after the last ``ANSWER:`` marker, else the whole reply.

    Tasks that say "reply with only ..." often make the tutor drop the marker.
    Unmarked replies are safe to try because only graded-correct answers are used.
    """
    marker = "ANSWER:"
    answer = text.rsplit(marker, 1)[1] if marker in text else text
    return answer.strip() or None


class Tutor:
    """Builds tutor prompts, calls the backend and parses replies."""

    def __init__(self, backend: TutorBackend, prices: dict[str, float] | None = None):
        self.backend = backend
        self.prices = dict(prices or PRICES)
        self.calls = {kind: 0 for kind in self.prices}

    def _ask(self, kind: str, prompt: str) -> str | None:
        self.calls[kind] += 1
        return self.backend.request(_ascii(prompt, 3900), kind)

    def hint(self, task: Task) -> TutorReply:
        text = self._ask(
            "hint",
            "A student must solve the task below. Give one short hint (at most two "
            "sentences) that helps them get it right. Do not state the final answer."
            f"\n\nTASK:\n{task.prompt}",
        )
        return TutorReply("hint", text)

    def worked_example(self, sibling: Task) -> TutorReply:
        text = self._ask(
            "worked_example",
            "Solve the task below. First reason briefly if needed, then write a line "
            "'ANSWER:' followed by only the final answer in exactly the format the "
            f"task requests.\n\nTASK:\n{sibling.prompt}",
        )
        answer = parse_answer(text) if text else None
        return TutorReply("worked_example", text, sibling, answer)

    def explanation(self, task: Task, student_answer: str, feedback: str) -> TutorReply:
        text = self._ask(
            "explanation",
            "A student attempted the task below and failed. In one or two sentences "
            "explain the mistake, then write a line 'ANSWER:' followed by only the "
            "correct final answer in exactly the format the task requests.\n\n"
            f"TASK:\n{task.prompt}\n\nSTUDENT ANSWER:\n{_ascii(student_answer)}\n\n"
            f"GRADER FEEDBACK: {feedback}",
        )
        answer = parse_answer(text) if text else None
        return TutorReply("explanation", text, task, answer)

    def feedback(self, task: Task, student_answer: str, feedback: str) -> TutorReply:
        """Explanation priced like ``explanation`` that withholds the answer."""
        text = self._ask(
            "explanation",
            "A student attempted the task below and failed. In at most three "
            "sentences, explain what is wrong and how to fix it. Do not state the "
            "final answer, the full solution or complete code.\n\n"
            f"TASK:\n{task.prompt}\n\nSTUDENT ANSWER:\n{_ascii(student_answer)}\n\n"
            f"GRADER FEEDBACK: {feedback}",
        )
        return TutorReply("explanation", text, task)


class ScriptedTutorBackend:
    """Offline backend for tests and dry runs; returns canned text."""

    def __init__(self, replies: list[str] | None = None, seed: int = 0):
        self.replies = replies or ["Read the task carefully.\nANSWER: 0"]
        self.rng = random.Random(seed)
        self.requests: list[tuple[str, str]] = []

    def request(self, prompt: str, purpose: str) -> str | None:
        self.requests.append((purpose, prompt))
        return self.rng.choice(self.replies)
