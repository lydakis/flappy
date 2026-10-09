"""Chat tasks: arithmetic Q&A and format-constrained instructions, rule-graded."""

from __future__ import annotations

import json
import random
import re

from world.tasks import Grade, Task, check_difficulty, new_task_id

WORDS = [
    "apple",
    "banana",
    "cherry",
    "grape",
    "lemon",
    "mango",
    "orange",
    "peach",
    "pear",
    "plum",
    "river",
    "stone",
    "cloud",
    "forest",
    "garden",
    "window",
    "rocket",
    "pencil",
    "candle",
    "mirror",
    "silver",
    "basket",
]
NAMES = ("Ada", "Bruno", "Chen", "Dara", "Elif", "Femi", "Gus", "Hana")
INT_RE = re.compile(r"-?\d+")


def _strip(answer: str) -> str:
    text = answer.strip()
    fence = re.fullmatch(r"```[a-zA-Z]*\n?(.*?)\n?```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return text.strip().strip(".").strip()


class ChatFamily:
    """Two skills: ``chat.arith`` (exact integers) and ``chat.instruct`` (format)."""

    name = "chat"
    skills = ("chat.arith", "chat.instruct")

    def sample(self, skill: str, difficulty: int, rng: random.Random) -> Task:
        check_difficulty(difficulty)
        if skill == "chat.arith":
            prompt, hidden = _arith(difficulty, rng)
        elif skill == "chat.instruct":
            prompt, hidden = _instruct(difficulty, rng)
        else:
            raise ValueError(f"unknown chat skill {skill}")
        return Task(
            new_task_id(rng, skill, difficulty),
            self.name,
            skill,
            difficulty,
            prompt,
            hidden,
        )

    def grade(self, task: Task, answer: str) -> Grade:
        kind = task.hidden["kind"]
        text = _strip(answer)
        if kind == "int":
            found = INT_RE.findall(text.replace(",", ""))
            ok = bool(found) and int(found[-1]) == task.hidden["value"]
            return Grade(ok, float(ok), "correct" if ok else "wrong number")
        if kind == "exact":
            ok = text == task.hidden["value"]
            return Grade(ok, float(ok), "correct" if ok else "does not match")
        if kind == "json":
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                return Grade(False, 0.0, "not valid JSON")
            ok = value == task.hidden["value"]
            return Grade(ok, float(ok), "correct" if ok else "wrong JSON content")
        raise ValueError(f"unknown grading kind {kind}")


def _arith(difficulty: int, rng: random.Random) -> tuple[str, dict]:
    suffix = " Answer with just the number."
    if difficulty == 1:
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        return f"What is {a} + {b}?{suffix}", {"kind": "int", "value": a + b}
    if difficulty == 2:
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        op = rng.choice("+-")
        value = a + b if op == "+" else a - b
        return f"What is {a} {op} {b}?{suffix}", {"kind": "int", "value": value}
    if difficulty == 3:
        a, b = rng.randint(12, 99), rng.randint(3, 19)
        return f"What is {a} * {b}?{suffix}", {"kind": "int", "value": a * b}
    if difficulty == 4:
        a, b, c = rng.randint(11, 49), rng.randint(3, 12), rng.randint(10, 99)
        return (
            f"What is {a} * {b} - {c}?{suffix}",
            {"kind": "int", "value": a * b - c},
        )
    name = rng.choice(NAMES)
    boxes, per_box, eaten, friends = (
        rng.randint(3, 9),
        rng.randint(4, 12),
        rng.randint(1, 9),
        rng.randint(2, 5),
    )
    total = boxes * per_box - eaten
    total -= total % friends
    eaten = boxes * per_box - total
    return (
        (
            f"{name} buys {boxes} boxes with {per_box} cookies each, eats {eaten}, "
            f"and shares the rest equally among {friends} friends. How many cookies "
            f"does each friend get?{suffix}"
        ),
        {"kind": "int", "value": total // friends},
    )


def _instruct(difficulty: int, rng: random.Random) -> tuple[str, dict]:
    if difficulty == 1:
        word = rng.choice(WORDS)
        return (
            f"Reply with the word '{word}' in uppercase letters and nothing else.",
            {"kind": "exact", "value": word.upper()},
        )
    if difficulty == 2:
        word = rng.choice(WORDS)
        return (
            f"Spell the word '{word}' backwards. Reply with only the reversed word.",
            {"kind": "exact", "value": word[::-1]},
        )
    if difficulty == 3:
        words = rng.sample(WORDS, 4)
        return (
            (
                "Sort these words alphabetically and reply with them separated by "
                f"a comma and a space, nothing else: {', '.join(words)}"
            ),
            {"kind": "exact", "value": ", ".join(sorted(words))},
        )
    if difficulty == 4:
        word = rng.choice([w for w in WORDS if len(set(w)) < len(w)])
        letter = max(set(word), key=word.count)
        return (
            (
                f"How many times does the letter '{letter}' appear in '{word}'? "
                "Answer with just the number."
            ),
            {"kind": "int", "value": word.count(letter)},
        )
    name, age = rng.choice(NAMES), rng.randint(18, 90)
    tags = rng.sample(WORDS, 2)
    return (
        (
            "Reply with only a JSON object with exactly these keys: "
            f'"name" set to "{name}", "age" set to the number {age}, and "tags" '
            f"set to a list containing {tags[0]!r} then {tags[1]!r} as strings."
        ),
        {"kind": "json", "value": {"name": name, "age": age, "tags": tags}},
    )
