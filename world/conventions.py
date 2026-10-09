"""Convention tasks: a small made-up command language over word lists.

The operation names are nonsense words, so a pretrained model cannot know them.
Tasks apply a short program to a list of words. A purchasable lesson of worked
examples teaches the convention; difficulty is the program length. This family
exists to test learning ahead: whether paying for a lesson now pays off over
many later tasks.
"""

from __future__ import annotations

import random

from world.tasks import Grade, Task, check_difficulty, new_task_id

WORDS = [
    "red",
    "blue",
    "green",
    "sky",
    "tree",
    "sea",
    "moon",
    "star",
    "sun",
    "rock",
    "leaf",
    "wind",
]

OPS = {
    "zib": lambda ws: ws[::-1],  # reverse the order
    "kor": lambda ws: [w.upper() for w in ws],  # uppercase every word
    "mup": lambda ws: ws + ws,  # repeat the list
    "tal": lambda ws: ws[1:],  # drop the first word
    "wex": lambda ws: sorted(ws),  # sort alphabetically
    "fen": lambda ws: ws[:2],  # keep the first two words
}


def run_program(program: list[str], words: list[str]) -> list[str]:
    for op in program:
        words = OPS[op](words)
    return words


def render(program: list[str], words: list[str]) -> str:
    return (
        f"Apply the program `{' | '.join(program)}` to the words: {' '.join(words)}\n"
        "Reply with only the resulting words separated by single spaces."
    )


class ConventionFamily:
    """One skill, ``conv.ops``: run a program written in the made-up language."""

    name = "conv"
    skills = ("conv.ops",)
    max_tokens = 64

    def sample(self, skill: str, difficulty: int, rng: random.Random) -> Task:
        check_difficulty(difficulty)
        if skill not in self.skills:
            raise ValueError(f"unknown convention skill {skill}")
        program = [rng.choice(list(OPS)) for _ in range(difficulty)]
        words = rng.sample(WORDS, rng.randint(3, 5))
        answer = " ".join(run_program(program, words))
        return Task(
            new_task_id(rng, skill, difficulty),
            self.name,
            skill,
            difficulty,
            render(program, words),
            {"value": answer, "program": program},
        )

    def grade(self, task: Task, answer: str) -> Grade:
        text = " ".join(answer.strip().strip("`").split())
        ok = text == task.hidden["value"]
        return Grade(ok, float(ok), "correct" if ok else "wrong result")


def lesson(n: int, rng: random.Random, max_len: int = 2) -> list[tuple[str, str]]:
    """Worked examples teaching the convention: every operation alone, then
    random programs up to ``max_len`` operations."""
    examples = []
    ops = list(OPS)
    for i in range(n):
        program = (
            [ops[i % len(ops)]]
            if i < len(ops)
            else [rng.choice(ops) for _ in range(rng.randint(1, max_len))]
        )
        words = rng.sample(WORDS, rng.randint(3, 5))
        examples.append((render(program, words), " ".join(run_program(program, words))))
    return examples
