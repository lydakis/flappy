"""Buttons tasks: a text-rendered UI driven by ``click(id)``/``type(id, text)`` calls.

The element/action vocabulary mirrors BrowserGym so MiniWoB pages can be rendered
into the same shape later.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from world.tasks import Grade, Task, check_difficulty, new_task_id

ACTION_RE = re.compile(
    r"""(click)\(\s*(\d+)\s*\)|(type)\(\s*(\d+)\s*,\s*(["'])(.*?)\5\s*\)"""
)
BUTTONS = ("OK", "Cancel", "Next", "Back", "Save", "Delete", "Help", "Close")
FIELDS = ("Name", "Email", "City", "Phone", "Username")
VALUES = ("alice", "bob@mail.com", "Paris", "555-0101", "zed42", "Lima", "kim")
OPTIONS = ("Subscribe", "Remember me", "Accept terms", "Dark mode")
ITEMS = ("Tea", "Bread", "Milk", "Rice", "Jam", "Eggs", "Soap", "Salt")
INSTRUCTIONS = (
    'Actions: click(id) and type(id, "text"), one per line, in order, using the '
    "numbers in brackets as ids. Clicking a button ends the episode. Reply with "
    'only the actions, for example:\ntype(4, "hello")\nclick(9)'
)


@dataclass(frozen=True)
class Element:
    id: int
    kind: str  # button | textbox | checkbox
    label: str


def render(title: str, elements: list[Element], goal: str) -> str:
    lines = [f"Screen: {title}"]
    for el in elements:
        if el.kind == "button":
            lines.append(f'[{el.id}] button "{el.label}"')
        elif el.kind == "textbox":
            lines.append(f'[{el.id}] textbox "{el.label}" value=""')
        else:
            lines.append(f'[{el.id}] checkbox "{el.label}" checked=no')
    return "\n".join(lines + [f"Goal: {goal}", INSTRUCTIONS])


def parse_actions(answer: str) -> list[tuple]:
    actions = []
    for match in ACTION_RE.finditer(answer):
        if match.group(1):
            actions.append(("click", int(match.group(2))))
        else:
            actions.append(("type", int(match.group(4)), match.group(6)))
    return actions


def _assign_ids(rng: random.Random, specs: list[tuple[str, str]]) -> list[Element]:
    ids = rng.sample(range(1, 10 * len(specs)), len(specs))
    return [Element(i, kind, label) for i, (kind, label) in zip(ids, specs)]


class ButtonsFamily:
    """One skill, ``buttons.form``: operate a small form to reach a goal state."""

    name = "buttons"
    skills = ("buttons.form",)
    max_tokens = 128  # answer length cap for generation

    def sample(self, skill: str, difficulty: int, rng: random.Random) -> Task:
        check_difficulty(difficulty)
        if skill not in self.skills:
            raise ValueError(f"unknown buttons skill {skill}")
        title, specs, target, text, checks, goal = _scenario(difficulty, rng)
        elements = _assign_ids(rng, specs)
        by_label = {el.label: el for el in elements}
        hidden = {
            "elements": [(el.id, el.kind, el.label) for el in elements],
            "target": by_label[target].id,
            "text": {by_label[k].id: v for k, v in text.items()},
            "checked": sorted(by_label[k].id for k in checks),
        }
        prompt = render(title, elements, goal)
        return Task(
            new_task_id(rng, skill, difficulty),
            self.name,
            skill,
            difficulty,
            prompt,
            hidden,
        )

    def grade(self, task: Task, answer: str) -> Grade:
        kinds = {i: kind for i, kind, _ in task.hidden["elements"]}
        text = {i: "" for i, kind in kinds.items() if kind == "textbox"}
        checked: set[int] = set()
        pressed = None
        for action in parse_actions(answer):
            target = action[1]
            if target not in kinds:
                return Grade(False, 0.0, f"no element {target}")
            if action[0] == "type":
                if kinds[target] != "textbox":
                    return Grade(False, 0.0, f"cannot type into {target}")
                text[target] = action[2]
            elif kinds[target] == "checkbox":
                checked ^= {target}
            elif kinds[target] == "button":
                pressed = target
                break
        expected_text = {i: task.hidden["text"].get(i, "") for i in text}
        parts = [
            pressed == task.hidden["target"],
            text == expected_text,
            sorted(checked) == task.hidden["checked"],
        ]
        ok = all(parts)
        if pressed is None:
            feedback = "no button was clicked"
        elif not parts[0]:
            feedback = "clicked the wrong button"
        elif not ok:
            feedback = "form state does not match the goal"
        else:
            feedback = "correct"
        return Grade(ok, sum(parts) / 3, feedback)


def _scenario(difficulty: int, rng: random.Random):
    """Return (title, element specs, target label, text, checks, goal)."""
    if difficulty <= 2:
        labels = rng.sample(BUTTONS, 2 if difficulty == 1 else 5)
        target = rng.choice(labels)
        specs = [("button", label) for label in labels]
        return "Dialog", specs, target, {}, [], f"Click the {target!r} button."
    if difficulty == 3:
        field, value = rng.choice(FIELDS), rng.choice(VALUES)
        specs = [("textbox", field), ("button", "Submit"), ("button", "Cancel")]
        rng.shuffle(specs)
        goal = f"Type {value!r} into the {field} field, then click Submit."
        return "Form", specs, "Submit", {field: value}, [], goal
    if difficulty == 4:
        fields = rng.sample(FIELDS, 2)
        values = rng.sample(VALUES, 2)
        options = rng.sample(OPTIONS, 2)
        specs = [("textbox", f) for f in fields] + [("checkbox", o) for o in options]
        specs += [("button", "Submit"), ("button", "Cancel")]
        rng.shuffle(specs)
        goal = (
            f"Set {fields[0]} to {values[0]!r} and {fields[1]} to {values[1]!r}, "
            f"check {options[0]!r} (leave {options[1]!r} unchecked), then click Submit."
        )
        return "Sign up", specs, "Submit", dict(zip(fields, values)), [options[0]], goal
    items = rng.sample(ITEMS, 5)
    prices = [rng.randint(1, 9) for _ in items]
    limit = rng.randint(3, 8)
    labels = [f"{item} (${price})" for item, price in zip(items, prices)]
    specs = [("checkbox", label) for label in labels]
    specs += [("button", "Buy"), ("button", "Cancel")]
    checks = [label for label, price in zip(labels, prices) if price < limit]
    goal = f"Select every item that costs less than ${limit}, then click Buy."
    return "Shop", specs, "Buy", {}, checks, goal
