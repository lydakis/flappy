"""The curious student's loop: pick jobs, practice by learning progress, buy tutoring.

Arms:
- ``no_tutor``: never asks.
- ``always_tutor``: hint before paid work, worked example before practice and an
  explanation after every failure, whenever the wallet allows.
- ``progress``: asks only when the skill has plateaued (or is unmeasured) and a
  Thompson sample of that kind's learned usefulness says the expected gain in
  value exceeds its price.

Learners (all update the same LoRA adapter):
- ``sft``: fine-tune on verified answers (own passes, graded tutor answers).
- ``grpo``: group of sampled practice answers, grade rewards, group advantages.
- ``hybrid``: route by the group's outcome. Mixed scores give a GRPO term; an
  all-fail group is "stuck" and may buy a tutor solution of that task, which is
  verified and added to replay; an all-pass group adds no GRPO term. Every
  practice step adds a cross-skill replay SFT term weighted per skill by
  ``lambda0 * (1 - recent success rate)``, so imitation fades as skills improve.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

from llm.budgeted_teacher import BudgetStop
from student.curiosity import ProgressTracker
from student.grpo import group_advantages
from student.model import StudentModel
from student.runlog import RunLog
from student.tutor import Tutor
from world.board import Wallet, World
from world.tasks import MAX_DIFFICULTY, MIN_DIFFICULTY, Grade, Task

ARMS = ("no_tutor", "always_tutor", "progress")
LEARNERS = ("sft", "grpo", "hybrid")


class Example(NamedTuple):
    """A verified prompt/answer pair in the replay buffer."""

    prompt: str
    completion: str
    source: str  # self | tutor
    skill: str


@dataclass
class LoopConfig:
    arm: str = "progress"
    learner: str = "sft"
    practice_value: float = 2.0  # credits a practice success is worth to the student
    plateau_eps: float = 0.1
    train_every: int = 2
    train_steps: int = 2
    batch_size: int = 4
    buffer_size: int = 512
    min_buffer: int = 4
    group_size: int = 4
    kl_coef: float = 0.02
    lambda0: float = 1.0  # hybrid: replay SFT weight at zero success

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        if self.learner not in LEARNERS:
            raise ValueError(f"learner must be one of {LEARNERS}")


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
        self.buffer: deque[Example] = deque(maxlen=config.buffer_size)
        self.group_outcomes: dict[str, Counter] = defaultdict(Counter)
        self.loss_parts: Counter = Counter()
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

    def wants(self, kind: str, task: Task, value: float, stuck: bool = False) -> bool:
        """Arm policy for buying help. ``stuck`` (an all-fail group) replaces the
        plateau test as evidence that the student cannot progress alone."""
        if self.tutor is None or not self.wallet.can_afford(self.tutor.prices[kind]):
            return False
        if self.config.arm == "always_tutor":
            return True
        if (
            not stuck
            and self.tracker.progress(task.skill) is not None
            and not self.tracker.plateaued(task.skill, self.config.plateau_eps)
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
                        "answer": (reply.answer or "")[:400] if reply else None,
                    }
                )
                if verified:
                    self.buffer.append(
                        Example(sibling.prompt, reply.answer, "tutor", task.skill)
                    )
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
                "answer": (reply.answer or "")[:400] if reply else None,
            }
        )
        if verified:
            self.buffer.append(
                Example(attempt.task.prompt, reply.answer, "tutor", attempt.task.skill)
            )
        self.pending_explanation.add(attempt.task.skill)

    # -- loop ------------------------------------------------------------------

    def tick(self) -> list[dict]:
        attempts = []
        job = self.choose_job()
        if job is not None:
            attempts.append(self.prepare("work", job.task, job.pay))
        attempts.append(self.prepare("practice", self.choose_practice(), 0.0))
        results = []
        for attempt in attempts:
            if attempt.mode == "practice" and self.config.learner != "sft":
                results.append(self.practice_group(attempt))
                continue
            # Paid work is greedy; practice samples to explore new answers.
            answer = self.model.generate(
                [attempt.prompt], sample=attempt.mode == "practice"
            )[0]
            results.append(self.settle(attempt, answer))
        self.ticks += 1
        if self.ticks % self.config.train_every == 0:
            self.train()
        self.world.tick()
        self.log_scalars()
        return results

    def practice_group(self, attempt: Attempt) -> dict:
        """Sample a group, grade it and take one GRPO or hybrid step.

        Bookkeeping (tracker, help value) uses the first sample only, so practice
        statistics match the single-sample SFT learner.
        """
        task = attempt.task
        answers = self.model.sample_group(attempt.prompt, self.config.group_size)
        grades = [self.world.grade(task, a) for a in answers]
        result = self.settle(attempt, answers[0], grades[0])
        rewards = [g.score for g in grades]
        advantages = group_advantages(rewards)
        if advantages is not None:
            outcome = "mixed"
        else:
            outcome = "all_pass" if all(g.passed for g in grades) else "all_fail"
        self.group_outcomes[task.skill][outcome] += 1
        group = (attempt.prompt, answers, advantages) if advantages else None
        if self.config.learner == "grpo":
            if group is None:
                return result
            stats = self.model.policy_step(*group, self.config.kl_coef)
            self._log_step(task.skill, outcome, rewards, stats)
            return result
        fresh = self.ask_when_stuck(task) if outcome == "all_fail" else None
        replay = self.replay_batch(fresh)
        if group is None and not replay:
            return result
        stats = self.model.hybrid_step(group, replay, self.config.kl_coef)
        self.loss_parts["grpo"] += abs(stats["grpo"])
        self.loss_parts["sft"] += abs(stats["sft"])
        self._log_step(
            task.skill,
            outcome,
            rewards,
            stats,
            replay=len(replay),
            tutor_injected=fresh is not None,
            lambdas={s: round(self.sft_weight(s), 3) for s in self.world.skills},
        )
        return result

    def _log_step(self, skill, outcome, rewards, stats, **extra) -> None:
        self.losses.append(stats["loss"])
        self.log.event(
            {
                "type": "train",
                "learner": self.config.learner,
                "tick": self.ticks,
                "skill": skill,
                "outcome": outcome,
                "reward_mean": sum(rewards) / len(rewards),
                **stats,
                **extra,
            }
        )

    def ask_when_stuck(self, task: Task) -> Example | None:
        """All samples failed: buy a tutor solution of this practice task."""
        kind = "worked_example"
        if not self.wants(kind, task, self.config.practice_value, stuck=True):
            return None
        if not self._buy(kind):
            return None
        reply = self._call(self.tutor.worked_example, task)
        verified = bool(
            reply and reply.answer and self.world.grade(task, reply.answer).passed
        )
        self.help.update(task.skill, kind, verified)
        self.log.event(
            {
                "type": "tutor",
                "kind": kind,
                "trigger": "stuck",
                "tick": self.ticks,
                "skill": task.skill,
                "verified": verified,
                "answer": (reply.answer or "")[:400] if reply else None,
            }
        )
        if not verified:
            return None
        example = Example(task.prompt, reply.answer, "tutor", task.skill)
        self.buffer.append(example)
        return example

    def sft_weight(self, skill: str) -> float:
        return self.config.lambda0 * (1.0 - self.tracker.success_rate(skill))

    def replay_batch(self, fresh: Example | None) -> list[tuple[str, str, float]]:
        """Weighted replay drawn evenly across skills, plus any fresh tutor answer."""
        if len(self.buffer) < self.config.min_buffer and fresh is None:
            return []
        by_skill: dict[str, list[Example]] = defaultdict(list)
        for example in self.buffer:
            by_skill[example.skill].append(example)
        picks = [fresh] if fresh is not None else []
        skills = sorted(by_skill)
        while skills and len(picks) < self.config.batch_size:
            picks.append(self.rng.choice(by_skill[self.rng.choice(skills)]))
        return [(e.prompt, e.completion, self.sft_weight(e.skill)) for e in picks]

    def settle(self, attempt: Attempt, answer: str, grade: Grade | None = None) -> dict:
        task = attempt.task
        grade = grade or self.world.grade(task, answer)
        if task.skill in self.pending_explanation:
            self.pending_explanation.discard(task.skill)
            self.help.update(task.skill, "explanation", grade.passed)
        for kind in attempt.help:
            self.help.update(task.skill, kind, grade.passed)
        self.tracker.record(task.skill, task.difficulty, grade.passed)
        earned = 0.0
        if grade.passed:
            # Train on the bare prompt so hints and examples are distilled away.
            self.buffer.append(Example(task.prompt, answer, "self", task.skill))
            if attempt.mode == "work":
                earned = attempt.pay
                self.wallet.earn(earned)
        elif self.tutor is not None and self.config.learner != "grpo":
            # Explanations only produce SFT data; GRPO learns from rewards alone.
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
        if self.config.learner != "sft" or len(self.buffer) < self.config.min_buffer:
            return
        recent = list(self.buffer)[-32:]
        for _ in range(self.config.train_steps):
            half = self.config.batch_size // 2
            batch = self.rng.sample(recent, min(half, len(recent)))
            batch += self.rng.choices(
                list(self.buffer), k=self.config.batch_size - len(batch)
            )
            loss = self.model.train_step([(e.prompt, e.completion) for e in batch])
            self.losses.append(loss)
        self.log.event(
            {
                "type": "train",
                "learner": "sft",
                "tick": self.ticks,
                "loss": self.losses[-1],
                "buffer": len(self.buffer),
                "tutor_examples": self.tutor_examples(),
            }
        )

    def tutor_examples(self) -> int:
        return sum(e.source == "tutor" for e in self.buffer)

    def learning_summary(self) -> dict:
        """Per-skill group outcomes, stuck asks and current SFT weight."""
        stuck = Counter(
            (e["skill"], e["verified"])
            for e in self.log.events
            if e["type"] == "tutor" and e.get("trigger") == "stuck"
        )
        per_skill = {}
        for skill in self.world.skills:
            outcomes = self.group_outcomes[skill]
            total = sum(outcomes.values())
            per_skill[skill] = {
                "groups": total,
                "zero_variance_rate": (
                    (outcomes["all_fail"] + outcomes["all_pass"]) / total
                    if total
                    else None
                ),
                **{k: outcomes[k] for k in ("mixed", "all_fail", "all_pass")},
                "stuck_asks": stuck[(skill, True)] + stuck[(skill, False)],
                "tutor_injected": stuck[(skill, True)],
                "sft_weight": round(self.sft_weight(skill), 3),
            }
        parts = self.loss_parts["grpo"] + self.loss_parts["sft"]
        return {
            "per_skill": per_skill,
            "sft_loss_share": self.loss_parts["sft"] / parts if parts else None,
        }

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
        for skill, outcomes in self.group_outcomes.items():
            total = sum(outcomes.values())
            scalars[f"groups/zero_variance/{skill}"] = (
                outcomes["all_fail"] + outcomes["all_pass"]
            ) / total
            scalars[f"groups/sft_weight/{skill}"] = self.sft_weight(skill)
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
