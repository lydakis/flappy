"""The curious student's loop: pick jobs, practice by learning progress, buy tutoring.

Arms:
- ``no_tutor``: never asks.
- ``always_tutor``: hint before paid work, worked example before practice and an
  explanation after every failure, whenever the wallet allows.
- ``progress``: asks only when the skill has plateaued (or is unmeasured) and a
  Thompson sample of that kind's learned usefulness says the expected gain in
  value exceeds its price.
"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field

from llm.budgeted_teacher import BudgetStop
from student.curiosity import ProgressTracker
from student.model import StudentModel
from student.runlog import RunLog
from student.tutor import Tutor
from world.board import Wallet, World
from world.tasks import MAX_DIFFICULTY, MIN_DIFFICULTY, Task

ARMS = ("no_tutor", "always_tutor", "progress")


@dataclass
class LoopConfig:
    arm: str = "progress"
    practice_value: float = 2.0  # credits a practice success is worth to the student
    plateau_eps: float = 0.1
    train_every: int = 2
    train_steps: int = 2
    batch_size: int = 4
    buffer_size: int = 512
    min_buffer: int = 4

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")


class HelpValue:
    """Beta posterior of P(success on the next attempt | help kind) per skill."""

    def __init__(self) -> None:
        self.counts: dict[tuple[str, str], list[float]] = defaultdict(
            lambda: [1.0, 1.0]
        )

    def update(self, skill: str, kind: str, success: bool) -> None:
        self.counts[(skill, kind)][0 if success else 1] += 1

    def sample(self, skill: str, kind: str, rng: random.Random) -> float:
        a, b = self.counts[(skill, kind)]
        return rng.betavariate(a, b)


@dataclass
class Attempt:
    mode: str  # work | practice
    task: Task
    pay: float
    prompt: str
    help: list[str] = field(default_factory=list)


class CuriousStudent:
    def __init__(
        self,
        world: World,
        model: StudentModel,
        *,
        config: LoopConfig,
        wallet: Wallet,
        tutor: Tutor | None = None,
        log: RunLog | None = None,
        spend: Callable[[], dict] | None = None,
        seed: int = 0,
    ):
        self.world = world
        self.model = model
        self.config = config
        self.wallet = wallet
        self.tutor_service = tutor if config.arm != "no_tutor" else None
        self.log = log or RunLog(None)
        self.spend = spend
        self.rng = random.Random(seed)
        self.tracker = ProgressTracker(world.skills)
        self.help = HelpValue()
        self.pending_explanation: set[str] = set()
        self.buffer: deque[tuple[str, str, str]] = deque(maxlen=config.buffer_size)
        self.ticks = 0
        self.losses: list[float] = []
        self.tutor_stopped = ""

    @property
    def tutor(self) -> Tutor | None:
        """The tutor while it is still available (a ledger stop removes it)."""
        return None if self.tutor_stopped else self.tutor_service

    # -- decisions -------------------------------------------------------------

    def choose_job(self):
        jobs = self.world.board()
        if not jobs:
            return None
        best = max(
            jobs,
            key=lambda j: self.tracker.job_value(j["skill"], j["difficulty"], j["pay"]),
        )
        return self.world.take(best["job_id"])

    def choose_practice(self) -> Task:
        skill = self.tracker.choose_practice_skill(self.rng)
        return self.world.practice_task(skill, self.tracker.frontier_difficulty(skill))

    def wants(self, kind: str, task: Task, value: float) -> bool:
        if self.tutor is None or not self.wallet.can_afford(self.tutor.prices[kind]):
            return False
        if self.config.arm == "always_tutor":
            return True
        if self.tracker.progress(task.skill) is not None and not self.tracker.plateaued(
            task.skill, self.config.plateau_eps
        ):
            return False  # still improving without help
        p_base, _ = self.tracker.estimate(task.skill, task.difficulty)
        gain = self.help.sample(task.skill, kind, self.rng) - p_base
        return gain * value > self.tutor.prices[kind]

    # -- tutoring --------------------------------------------------------------

    def _buy(self, kind: str) -> bool:
        """Charge the kind's price in credits; False if unavailable or unaffordable."""
        return self.tutor is not None and self.wallet.spend(self.tutor.prices[kind])

    def _call(self, fn, *args):
        try:
            return fn(*args)
        except BudgetStop as stop:
            self.tutor_stopped = str(stop)
            self.log.event(
                {"type": "tutor_stopped", "tick": self.ticks, "reason": str(stop)}
            )
            return None

    def prepare(self, mode: str, task: Task, pay: float) -> Attempt:
        attempt = Attempt(mode, task, pay, task.prompt)
        value = pay if mode == "work" else self.config.practice_value
        candidates = ["hint", "worked_example"]
        if self.config.arm == "always_tutor":
            candidates = ["hint"] if mode == "work" else ["worked_example"]
        for kind in candidates:
            if not self.wants(kind, task, value) or not self._buy(kind):
                continue
            if kind == "hint":
                reply = self._call(self.tutor.hint, task)
                if reply and reply.text:
                    attempt.prompt = f"{task.prompt}\n\nHint: {reply.text.strip()}"
                    attempt.help.append("hint")
            else:
                sibling = self.world.sample(task.skill, task.difficulty)
                reply = self._call(self.tutor.worked_example, sibling)
                verified = bool(
                    reply
                    and reply.answer
                    and self.world.grade(sibling, reply.answer).passed
                )
                self.log.event(
                    {
                        "type": "tutor",
                        "kind": kind,
                        "tick": self.ticks,
                        "skill": task.skill,
                        "verified": verified,
                    }
                )
                if verified:
                    self.buffer.append((sibling.prompt, reply.answer, "tutor"))
                    attempt.prompt = (
                        f"Example task:\n{sibling.prompt}\nExample answer:\n{reply.answer}"
                        f"\n\nNow solve this task.\n{task.prompt}"
                    )
                    attempt.help.append("worked_example")
            break  # at most one pre-attempt purchase
        return attempt

    def explain(self, attempt: Attempt, answer: str, feedback: str) -> None:
        if not self.wants("explanation", attempt.task, self.config.practice_value):
            return
        if not self._buy("explanation"):
            return
        reply = self._call(self.tutor.explanation, attempt.task, answer, feedback)
        verified = bool(
            reply
            and reply.answer
            and self.world.grade(attempt.task, reply.answer).passed
        )
        self.log.event(
            {
                "type": "tutor",
                "kind": "explanation",
                "tick": self.ticks,
                "skill": attempt.task.skill,
                "verified": verified,
            }
        )
        if verified:
            self.buffer.append((attempt.task.prompt, reply.answer, "tutor"))
        self.pending_explanation.add(attempt.task.skill)

    # -- loop ------------------------------------------------------------------

    def tick(self) -> list[dict]:
        attempts = []
        job = self.choose_job()
        if job is not None:
            attempts.append(self.prepare("work", job.task, job.pay))
        attempts.append(self.prepare("practice", self.choose_practice(), 0.0))
        answers = []
        for attempt in attempts:
            # Paid work is greedy; practice samples to explore new answers.
            answers += self.model.generate(
                [attempt.prompt], sample=attempt.mode == "practice"
            )
        results = [self.settle(a, ans) for a, ans in zip(attempts, answers)]
        self.ticks += 1
        if self.ticks % self.config.train_every == 0:
            self.train()
        self.world.tick()
        self.log_scalars()
        return results

    def settle(self, attempt: Attempt, answer: str) -> dict:
        task = attempt.task
        grade = self.world.grade(task, answer)
        if task.skill in self.pending_explanation:
            self.pending_explanation.discard(task.skill)
            self.help.update(task.skill, "explanation", grade.passed)
        for kind in attempt.help:
            self.help.update(task.skill, kind, grade.passed)
        self.tracker.record(task.skill, task.difficulty, grade.passed)
        earned = 0.0
        if grade.passed:
            # Train on the bare prompt so hints and examples are distilled away.
            self.buffer.append((task.prompt, answer, "self"))
            if attempt.mode == "work":
                earned = attempt.pay
                self.wallet.earn(earned)
        elif self.tutor is not None:
            self.explain(attempt, answer, grade.feedback)
        event = {
            "type": "attempt",
            "tick": self.ticks,
            "mode": attempt.mode,
            "skill": task.skill,
            "difficulty": task.difficulty,
            "passed": grade.passed,
            "score": grade.score,
            "earned": earned,
            "help": attempt.help,
            "credits": self.wallet.balance,
        }
        self.log.event(event)
        return event

    def train(self) -> None:
        if len(self.buffer) < self.config.min_buffer:
            return
        recent = list(self.buffer)[-32:]
        for _ in range(self.config.train_steps):
            half = self.config.batch_size // 2
            batch = self.rng.sample(recent, min(half, len(recent)))
            batch += self.rng.choices(
                list(self.buffer), k=self.config.batch_size - len(batch)
            )
            loss = self.model.train_step([(p, c) for p, c, _ in batch])
            self.losses.append(loss)
        self.log.event(
            {
                "type": "train",
                "tick": self.ticks,
                "loss": self.losses[-1],
                "buffer": len(self.buffer),
                "tutor_examples": sum(src == "tutor" for *_, src in self.buffer),
            }
        )

    def tutor_counts(self) -> dict[str, int]:
        return dict(self.tutor_service.calls) if self.tutor_service else {}

    def log_scalars(self) -> None:
        scalars = {
            f"success/{s}": self.tracker.success_rate(s) for s in self.world.skills
        }
        for skill in self.world.skills:
            lp = self.tracker.progress(skill)
            if lp is not None:
                scalars[f"progress/{skill}"] = lp
        scalars["wallet/credits"] = self.wallet.balance
        scalars["wallet/earned"] = self.wallet.earned
        scalars["wallet/spent"] = self.wallet.spent
        if self.losses:
            scalars["train/loss"] = self.losses[-1]
        if self.spend is not None:
            totals = self.spend()
            scalars["tutor/calls"] = totals["calls"]
            scalars["tutor/estimated_usd"] = totals["estimated_usd"]
            scalars["tutor/reserved_usd"] = totals["reserved_usd"]
        self.log.scalars(self.ticks, scalars)


def evaluation_set(world: World, per_cell: int, seed: int) -> list[Task]:
    """Fixed held-out tasks: ``per_cell`` per (skill, difficulty)."""
    rng = random.Random(seed)
    return [
        world.sample(skill, difficulty, rng)
        for skill in world.skills
        for difficulty in range(MIN_DIFFICULTY, MAX_DIFFICULTY + 1)
        for _ in range(per_cell)
    ]


def evaluate(
    world: World, model: StudentModel, tasks: list[Task], batch: int = 8
) -> dict:
    """Greedy pass rate per skill and per (skill, difficulty); no learning."""
    answers: list[str] = []
    for start in range(0, len(tasks), batch):
        answers += model.generate([t.prompt for t in tasks[start : start + batch]])
    cells: dict[str, list[bool]] = defaultdict(list)
    for task, answer in zip(tasks, answers):
        passed = world.grade(task, answer).passed
        cells[task.skill].append(passed)
        cells[f"{task.skill}/d{task.difficulty}"].append(passed)
    return {k: sum(v) / len(v) for k, v in sorted(cells.items())}
