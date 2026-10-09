"""Job board, wallet and the single ``World`` interface over all task families."""

from __future__ import annotations

import random
from dataclasses import dataclass

from world.buttons import ButtonsFamily
from world.chat import ChatFamily
from world.code import CodeFamily
from world.tasks import MAX_DIFFICULTY, MIN_DIFFICULTY, Grade, Task, TaskFamily

PAY = {1: 1.0, 2: 2.0, 3: 4.0, 4: 7.0, 5: 12.0}


@dataclass(frozen=True)
class Job:
    job_id: str
    task: Task
    pay: float
    expires_at: int

    def public(self) -> dict:
        """Metadata the student may see before committing to the job."""
        return {
            "job_id": self.job_id,
            "skill": self.task.skill,
            "difficulty": self.task.difficulty,
            "pay": self.pay,
            "expires_at": self.expires_at,
        }


class Wallet:
    """Credits earned from paid jobs; spending never drives the balance negative."""

    def __init__(self, balance: float):
        self.balance = float(balance)
        self.earned = 0.0
        self.spent = 0.0

    def earn(self, amount: float) -> None:
        self.balance += amount
        self.earned += amount

    def can_afford(self, amount: float) -> bool:
        return self.balance >= amount

    def spend(self, amount: float) -> bool:
        if not self.can_afford(amount):
            return False
        self.balance -= amount
        self.spent += amount
        return True


class World:
    """Posts jobs across families, hands out practice tasks and grades answers."""

    def __init__(
        self,
        families: list[TaskFamily] | None = None,
        *,
        seed: int = 0,
        board_size: int = 6,
        job_ttl: int = 8,
    ):
        self.families = families or [ChatFamily(), CodeFamily(), ButtonsFamily()]
        self.by_skill = {s: f for f in self.families for s in f.skills}
        self.skills = tuple(self.by_skill)
        self.rng = random.Random(seed)
        self.board_size = board_size
        self.job_ttl = job_ttl
        self.now = 0
        self.jobs: dict[str, Job] = {}
        self._refill()

    def sample(
        self, skill: str, difficulty: int, rng: random.Random | None = None
    ) -> Task:
        return self.by_skill[skill].sample(skill, difficulty, rng or self.rng)

    def max_tokens(self, skill: str) -> int:
        return getattr(self.by_skill[skill], "max_tokens", 256)

    def grade(self, task: Task, answer: str) -> Grade:
        return self.by_skill[task.skill].grade(task, answer)

    def board(self) -> list[dict]:
        return [job.public() for job in self.jobs.values()]

    def take(self, job_id: str) -> Job:
        """Remove a job from the board; the caller attempts it once."""
        return self.jobs.pop(job_id)

    def practice_task(self, skill: str, difficulty: int) -> Task:
        return self.sample(skill, difficulty)

    def tick(self) -> None:
        self.now += 1
        self.jobs = {k: j for k, j in self.jobs.items() if j.expires_at > self.now}
        self._refill()

    def _refill(self) -> None:
        while len(self.jobs) < self.board_size:
            skill = self.rng.choice(self.skills)
            difficulty = self.rng.randint(MIN_DIFFICULTY, MAX_DIFFICULTY)
            task = self.sample(skill, difficulty)
            self.jobs[task.task_id] = Job(
                task.task_id, task, PAY[difficulty], self.now + self.job_ttl
            )
