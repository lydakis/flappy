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
        self.policy_steps = []
        self.hybrid_steps = []

    def generate(self, prompts, *, sample=False, max_new_tokens=None):
        self.prompts += prompts
        smart = self.learn_after is not None and self.trained >= self.learn_after
        return [
            re.findall(r"'(\w+)'", p)[-1] if smart or "Tutor feedback:" in p else "?"
            for p in prompts
        ]

    def train_step(self, examples):
        self.trained += 1
        return 1.0 / self.trained

    group_mode = "mixed"  # mixed | fail | pass

    def sample_group(self, prompt, n, max_new_tokens=None):
        word = re.findall(r"'(\w+)'", prompt)[-1]
        mode = "mixed" if "Tutor feedback:" in prompt else self.group_mode
        return {"mixed": ["?", word], "fail": ["?", "?"], "pass": [word, word]}[
            mode
        ] * (n // 2)

    def policy_step(self, prompt, completions, advantages, kl_coef):
        self.policy_steps.append((completions, advantages))
        return {"loss": 0.1, "kl": 0.0}

    def hybrid_step(self, group, replay, kl_coef):
        self.hybrid_steps.append((group, replay))
        return {"loss": 0.2, "grpo": 0.1 if group else 0.0, "sft": 0.1, "kl": 0.0}


class SolvingBackend:
    """Scripted tutor that answers the last quoted word in the prompt."""

    def __init__(self, fail_after=None, leak=False):
        self.requests = []
        self.fail_after = fail_after
        self.leak = leak

    def request(self, prompt, purpose):
        if self.fail_after is not None and len(self.requests) >= self.fail_after:
            raise BudgetStop("Tutor total or per-run allowance reached")
        self.requests.append(purpose)
        task_text = prompt.split("TASK:\n", 1)[1].split("\n\n", 1)[0]
        word = re.findall(r"'(\w+)'", task_text)[-1]
        if "Do not state the final answer" in prompt and not self.leak:
            return "Repeat the quoted word exactly, without punctuation."
        return f"Echo it.\nANSWER: {word}"


def make(
    arm, backend=None, credits=100.0, model=None, learner="sft", tutor_mode="retry"
):
    world = World([EchoFamily()], seed=1, board_size=3)
    student = CuriousStudent(
        world,
        model or FakeModel(),
        config=LoopConfig(
            arm=arm,
            learner=learner,
            tutor_mode=tutor_mode,
            train_every=1,
            train_steps=1,
            min_buffer=1,
        ),
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
    student = make("always_tutor", backend, model=model, tutor_mode="imitate")
    results = [r for _ in range(6) for r in student.tick()]
    assert {"hint", "worked_example", "explanation"} <= set(backend.requests)
    assert student.wallet.spent > 0
    assert student.tutor_examples() > 0
    assert all(example.completion != "?" for example in student.buffer)
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


def test_parse_answer_takes_last_marker_or_the_bare_reply():
    assert parse_answer("ANSWER: 1\nthen ANSWER: 2") == "2"
    # Regression: the tutor follows "reply with only the actions" and omits the marker.
    assert parse_answer("click(13)\n") == "click(13)"
    assert parse_answer("ANSWER:  ") is None


@pytest.mark.parametrize("kwargs", [{"arm": "bogus"}, {"learner": "ppo"}])
def test_unknown_arm_or_learner_is_rejected(kwargs):
    with pytest.raises(ValueError):
        LoopConfig(**kwargs)


def test_tutor_never_sees_hidden_grader_data():
    backend = SolvingBackend()
    student = make("always_tutor", backend)
    family = EchoFamily()
    task = family.sample("toy.echo", 1, random.Random(0))
    student.tutor.explanation(task, "?", "wrong")
    assert "hidden" not in str(backend.requests)


def test_grpo_learns_from_group_rewards_not_sft_or_explanations():
    backend = SolvingBackend()
    model = FakeModel()
    student = make("always_tutor", backend, model=model, learner="grpo")
    for _ in range(4):
        student.tick()
    assert model.trained == 0 and "explanation" not in backend.requests
    assert len(model.policy_steps) == 4
    completions, advantages = model.policy_steps[0]
    assert sum(advantages) == pytest.approx(0)
    for completion, advantage in zip(completions, advantages):
        assert (advantage > 0) == (completion != "?")
    # One tracker record per practice group, matching the SFT learner.
    practice = [e for e in student.log.events if e.get("mode") == "practice"]
    assert len(practice) == 4


def test_hybrid_routes_by_group_outcome():
    model = FakeModel()
    student = make("progress", model=model, learner="hybrid")
    model.group_mode = "mixed"
    student.tick()
    group, _ = model.hybrid_steps[-1]
    assert group is not None and sum(group[2]) == pytest.approx(0)

    # All samples pass: no GRPO term and no stuck-triggered tutor request.
    model.group_mode = "pass"
    student.tick()
    assert model.hybrid_steps[-1][0] is None
    assert not [e for e in student.log.events if e.get("trigger") == "stuck"]
    summary = student.learning_summary()["per_skill"]
    assert sum(s["mixed"] for s in summary.values()) == 1
    assert sum(s["all_pass"] for s in summary.values()) == 1


def test_hybrid_stuck_group_buys_verified_tutor_answer_for_replay():
    backend = SolvingBackend()
    model = FakeModel()
    model.group_mode = "fail"
    student = make(
        "always_tutor", backend, model=model, learner="hybrid", tutor_mode="imitate"
    )
    student.tick()
    group, replay = model.hybrid_steps[-1]
    stuck = [e for e in student.log.events if e.get("trigger") == "stuck"]
    assert group is None and len(stuck) == 1 and stuck[0]["verified"]
    # The fresh tutor answer leads the replay batch, weighted by the skill's lambda.
    prompt, completion, weight = replay[0]
    skill = stuck[0]["skill"]
    assert completion in prompt and weight == pytest.approx(student.sft_weight(skill))
    summary = student.learning_summary()["per_skill"][skill]
    assert summary["all_fail"] == 1 and summary["rescued"] == 1


def test_hybrid_without_tutor_or_data_takes_no_step():
    model = FakeModel()
    model.group_mode = "fail"
    student = make("no_tutor", model=model, learner="hybrid")
    student.tick()
    assert model.hybrid_steps == [] and model.trained == 0


def test_replay_weight_decays_as_a_skill_improves():
    student = make("no_tutor", learner="hybrid")
    before = student.sft_weight("toy.echo")
    for _ in range(10):
        student.tracker.record("toy.echo", 1, True)
    assert student.sft_weight("toy.echo") < before
    assert student.sft_weight("toy.other") == pytest.approx(before)


def test_progress_arm_can_afford_to_rescue_a_stuck_skill():
    # Regression: help on practice was valued at 2 credits, below the 3-credit
    # worked example, so the progress arm could never ask when stuck.
    backend = SolvingBackend()
    model = FakeModel()
    model.group_mode = "fail"
    student = make("progress", backend, model=model, learner="hybrid")
    for _ in range(10):
        student.tick()
    stuck = [e for e in student.log.events if e.get("trigger") == "stuck"]
    assert stuck and student.learning_summary()["per_skill"]["toy.echo"]["rescued"]


def test_retry_mode_learns_from_own_retries_never_the_tutor_answer():
    backend = SolvingBackend()
    model = FakeModel()
    student = make("always_tutor", backend, model=model)
    for _ in range(3):
        student.tick()
    sources = {e.source for e in student.buffer}
    assert "retry" in sources and "tutor" not in sources
    # Retries are stored against the bare task prompt, without the feedback.
    assert all("Tutor feedback" not in e.prompt for e in student.buffer)
    retries = [e for e in student.log.events if e["type"] == "retry"]
    assert retries and all(e["trigger"] == "failure" for e in retries)


def test_retry_mode_stuck_group_resamples_with_feedback_for_grpo_and_replay():
    model = FakeModel()
    model.group_mode = "fail"
    student = make("always_tutor", SolvingBackend(), model=model, learner="hybrid")
    student.tick()
    group, replay = model.hybrid_steps[-1]
    prompt, _, advantages = group
    assert "Tutor feedback:" in prompt and sum(advantages) == pytest.approx(0)
    assert replay[0][0] == prompt.split("\n\nYour previous answer:")[0]
    assert student.tutor_examples() == 0
    summary = student.learning_summary()["per_skill"]
    skill = next(s for s, v in summary.items() if v["stuck_asks"])
    stuck = [e for e in student.log.events if e.get("trigger") == "stuck"]
    assert summary[skill]["rescued"] == 1
    assert [e["passed"] for e in stuck if e["type"] == "retry"] == [2]


def test_feedback_that_leaks_a_passing_answer_is_discarded():
    model = FakeModel()
    model.group_mode = "fail"
    student = make(
        "always_tutor", SolvingBackend(leak=True), model=model, learner="hybrid"
    )
    student.tick()
    assert [e for e in student.log.events if e.get("leaked")]
    assert not [e for e in student.log.events if e["type"] == "retry"]
    assert all("Tutor feedback" not in p for p in model.prompts)


def test_leak_check_finds_answers_embedded_in_feedback_sentences():
    from world.chat import ChatFamily

    world = World([ChatFamily()], seed=0)
    student = CuriousStudent(
        world, FakeModel(), config=LoopConfig(), wallet=Wallet(10), seed=0
    )
    task = world.sample("chat.instruct", 2, random.Random(1))
    reversed_word = task.hidden["value"]
    assert student.leaks_answer(task, f"So the reversed word is '{reversed_word}'.")
    assert not student.leaks_answer(task, "Read the letters from the last one.")
