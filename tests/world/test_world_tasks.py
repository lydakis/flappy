"""Offline checks that every family samples all levels and grades both ways."""

import json
import random

import pytest

from world import ButtonsFamily, ChatFamily, CodeFamily, Wallet, World
from world.buttons import parse_actions
from world.code import PROBLEMS, run_tests


def reference_answer(task) -> str:
    """Construct a correct answer from grader-only data."""
    hidden = task.hidden
    if task.family == "chat":
        value = hidden["value"]
        return json.dumps(value) if hidden["kind"] == "json" else str(value)
    if task.family == "buttons":
        lines = [f'type({i}, "{v}")' for i, v in hidden["text"].items()]
        lines += [f"click({i})" for i in hidden["checked"]]
        return "\n".join(lines + [f"click({hidden['target']})"])
    raise ValueError(task.family)


@pytest.mark.parametrize("family", [ChatFamily(), ButtonsFamily()])
@pytest.mark.parametrize("difficulty", range(1, 6))
def test_rule_graders_accept_reference_and_reject_junk(family, difficulty):
    rng = random.Random(difficulty)
    for skill in family.skills:
        for _ in range(5):
            task = family.sample(skill, difficulty, rng)
            assert family.grade(task, reference_answer(task)).passed, task.prompt
            assert not family.grade(task, "I am not sure.").passed


def test_chat_grading_tolerates_wrapping_but_not_wrong_values():
    family = ChatFamily()
    task = family.sample("chat.arith", 3, random.Random(1))
    value = task.hidden["value"]
    assert family.grade(task, f"The answer is {value}.").passed
    assert not family.grade(task, str(value + 1)).passed


def test_buttons_order_and_state_matter():
    family = ButtonsFamily()
    task = family.sample("buttons.form", 4, random.Random(3))
    good = reference_answer(task)
    *setup, submit = good.splitlines()
    assert family.grade(task, good).passed
    # Clicking Submit first ends the episode before the form is filled.
    assert not family.grade(task, "\n".join([submit, *setup])).passed
    assert parse_actions('type(4, "a b")\nclick( 7 )') == [
        ("type", 4, "a b"),
        ("click", 7),
    ]


def test_code_tests_run_isolated_and_score_partially():
    family = CodeFamily()
    task = family.sample("code.func", 2, random.Random(0))
    name, cases = task.hidden["name"], task.hidden["cases"]
    assert not family.grade(task, "def nope(): pass").passed
    # A lookup table of the hidden cases passes; a wrong solution fails them all.
    table = {json.dumps(args): out for args, out in cases}
    lookup = f"import json\nT={table!r}\ndef {name}(*a): return T[json.dumps(list(a))]"
    assert family.grade(task, f"```python\n{lookup}\n```").passed
    wrong = family.grade(task, f"def {name}(*a): return object()")
    assert wrong.score == 0 and wrong.feedback.startswith("0/")


def test_code_sandbox_survives_hangs_and_forged_output():
    passed, error = run_tests("def f():\n    while True: pass", "f", [[[], None]])
    assert passed == 0 and error in {"timeout", "no result"}
    forged = "print('{\"passed\": 99}')\ndef f(): return 1"
    assert run_tests(forged, "f", [[[], 2]])[0] == 0


def test_every_code_problem_reference_is_consistent():
    family = CodeFamily(n_cases=4)
    rng = random.Random(5)
    for difficulty, problems in PROBLEMS.items():
        for _ in range(len(problems) * 3):
            task = family.sample("code.func", difficulty, rng)
            assert len(task.hidden["cases"]) >= 2
            assert "Reply with only the code" in task.prompt


def test_world_board_refills_expires_and_wallet_never_negative():
    world = World(seed=0, board_size=4, job_ttl=2)
    first = {job["job_id"] for job in world.board()}
    assert len(first) == 4 and {j["difficulty"] for j in world.board()} <= set(
        range(1, 6)
    )
    job = world.take(next(iter(first)))
    assert job.job_id not in {j["job_id"] for j in world.board()}
    world.tick()
    world.tick()
    assert len(world.board()) == 4 and not first & {j["job_id"] for j in world.board()}
    wallet = Wallet(2)
    assert wallet.spend(2) and not wallet.spend(0.5) and wallet.balance == 0


def test_code_grading_ignores_self_written_top_level_checks():
    family = CodeFamily()
    task = family.sample("code.func", 1, random.Random(2))
    name, cases = task.hidden["name"], task.hidden["cases"]
    table = {json.dumps(args): out for args, out in cases}
    solution = (
        f"import json\nT = {table!r}\n\ndef {name}(*a):\n    return T[json.dumps(list(a))]\n\n"
        "assert False, 'a wrong self-test'\nprint('demo')"
    )
    assert family.grade(task, solution).passed
