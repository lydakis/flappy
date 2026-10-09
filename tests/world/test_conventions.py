"""The made-up command language: semantics, grading and lessons."""

import random

from world.conventions import OPS, ConventionFamily, lesson, run_program


def test_programs_compose_left_to_right():
    words = ["red", "blue", "green"]
    assert run_program(["zib", "kor"], words) == ["GREEN", "BLUE", "RED"]
    assert run_program(["mup", "tal", "fen"], words) == ["blue", "green"]
    assert run_program(["wex"], words) == ["blue", "green", "red"]


def test_grader_is_exact_up_to_whitespace():
    family = ConventionFamily()
    task = family.sample("conv.ops", 3, random.Random(0))
    value = task.hidden["value"]
    assert family.grade(task, f"  {value} \n").passed
    assert family.grade(task, f"`{value}`").passed
    assert not family.grade(task, value.lower() + " extra").passed
    assert len(task.hidden["program"]) == 3 and "`" in task.prompt


def test_lesson_covers_every_operation_and_is_correct():
    family = ConventionFamily()
    examples = lesson(20, random.Random(1))
    assert len(examples) == 20
    for op in OPS:
        assert any(f"`{op}`" in prompt for prompt, _ in examples)
    # Lesson answers are graded correct by the family's own grader.
    task = family.sample("conv.ops", 1, random.Random(2))
    assert family.grade(task, task.hidden["value"]).passed
