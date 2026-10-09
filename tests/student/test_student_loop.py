"""Loop behaviour per arm with a mocked model, toy task family and scripted tutor."""

import random
import re

import pytest

from llm.budgeted_teacher import BudgetStop
from student.agent import CuriousStudent, LoopConfig, evaluate, evaluation_set
from student.tutor import Tutor, parse_answer
from world.board import Wallet, World
from world.tasks import Grade, Task, new_task_id


class EchoFamily:
    """Toy family: the answer is the word in quotes."""

    name = "toy"
    skills = ("toy.echo", "toy.other")

    def sample(self, skill, difficulty, rng):
        word = rng.choice(["red", "blue", "green"])
        task_id = new_task_id(rng, skill, difficulty)
        return Task(
            task_id, self.name, skill, difficulty, f"Say '{word}'.", {"w": word}
        )

    def grade(self, task, answer):
        ok = answer.strip() == task.hidden["w"]
        return Grade(ok, float(ok), "ok" if ok else "wrong")


class FakeModel:
    """Fails until it has trained on enough examples, then echoes correctly."""

    def __init__(self, learn_after=None):
        self.trained = 0
        self.learn_after = learn_after
        self.prompts = []

    def generate(self, prompts, *, sample=False):
        self.prompts += prompts
        smart = self.learn_after is not None and self.trained >= self.learn_after
        return [re.findall(r"'(\w+)'", p)[-1] if smart else "?" for p in prompts]

    def train_step(self, examples):
        self.trained += 1
        return 1.0 / self.trained


class SolvingBackend:
    """Scripted tutor that answers the last quoted word in the prompt."""

    def __init__(self, fail_after=None):
        self.requests = []
        self.fail_after = fail_after

    def request(self, prompt, purpose):
        if self.fail_after is not None and len(self.requests) >= self.fail_after:
            raise BudgetStop("Tutor total or per-run allowance reached")
        self.requests.append(purpose)
        task_text = prompt.split("TASK:\n", 1)[1].split("\n\n", 1)[0]
        word = re.findall(r"'(\w+)'", task_text)[-1]
        return f"Echo it.\nANSWER: {word}"


def make(arm, backend=None, credits=100.0, model=None):
    world = World([EchoFamily()], seed=1, board_size=3)
    student = CuriousStudent(
        world,
        model or FakeModel(),
        config=LoopConfig(arm=arm, train_every=1, train_steps=1, min_buffer=1),
        wallet=Wallet(credits),
        tutor=Tutor(backend or SolvingBackend()),
        seed=3,
    )
    return student


def test_no_tutor_arm_never_calls_or_pays_the_tutor():
    backend = SolvingBackend()
    student = make("no_tutor", backend)
    for _ in range(5):
        student.tick()
    assert backend.requests == [] and student.wallet.spent == 0
    assert student.losses == []  # no successes and no tutor -> nothing to learn from


def test_always_tutor_buys_help_and_only_verified_answers_become_training_data():
    backend = SolvingBackend()
    model = FakeModel(learn_after=2)
    student = make("always_tutor", backend, model=model)
    results = [r for _ in range(6) for r in student.tick()]
    assert {"hint", "worked_example", "explanation"} <= set(backend.requests)
    assert student.wallet.spent > 0
    assert any(src == "tutor" for *_, src in student.buffer)
    assert all(completion != "?" for _, completion, _ in student.buffer)
    # Once trained, paid work succeeds and earns credits.
    assert any(r["passed"] and r["mode"] == "work" and r["earned"] > 0 for r in results)


def test_progress_arm_spends_less_than_always_and_respects_the_wallet():
    always, progress = make("always_tutor"), make("progress")
    for _ in range(8):
        always.tick()
        progress.tick()
    assert progress.wallet.spent < always.wallet.spent
    poor = make("always_tutor", credits=0.5)
    poor.tick()
    assert poor.wallet.spent == 0 and poor.tutor_service.calls["hint"] == 0


def test_ledger_stop_disables_tutor_for_the_rest_of_the_run():
    student = make("always_tutor", SolvingBackend(fail_after=1))
    for _ in range(4):
        student.tick()
    assert "allowance" in student.tutor_stopped and student.tutor is None
    assert any(e["type"] == "tutor_stopped" for e in student.log.events)


def test_evaluation_is_fixed_and_learning_free():
    world = World([EchoFamily()], seed=1)
    tasks = evaluation_set(world, per_cell=2, seed=9)
    assert [t.prompt for t in tasks] == [t.prompt for t in evaluation_set(world, 2, 9)]
    model = FakeModel(learn_after=0)
    assert evaluate(world, model, tasks)["toy.echo"] == 1.0 and model.trained == 0


def test_parse_answer_takes_last_marker():
    assert parse_answer("ANSWER: 1\nthen ANSWER: 2") == "2"
    assert parse_answer("no marker") is None


@pytest.mark.parametrize("arm", ["bogus"])
def test_unknown_arm_is_rejected(arm):
    with pytest.raises(ValueError):
        LoopConfig(arm=arm)


def test_tutor_never_sees_hidden_grader_data():
    backend = SolvingBackend()
    student = make("always_tutor", backend)
    family = EchoFamily()
    task = family.sample("toy.echo", 1, random.Random(0))
    student.tutor.explanation(task, "?", "wrong")
    assert "hidden" not in str(backend.requests)
