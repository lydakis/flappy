"""Coach-guided random agent for ablations."""

from __future__ import annotations

from envs.browsergym_client import BrowserGymEnvWrapper
from llm.coach import Coach
from llm.memory import JsonlMemoryStore
from agents.hybrid import HybridAgent


class CoachRandomAgent(HybridAgent):
    """Uses coach guidance but samples uniformly within masks."""

    def __init__(
        self,
        env: BrowserGymEnvWrapper,
        coach: Coach,
        *,
        memory: JsonlMemoryStore | None = None,
        planner_interval: int = 10,
        max_steps: int = 200,
        reflexion_enabled: bool = True,
        reflexion_read_only: bool = False,
    ) -> None:
        super().__init__(
            env=env,
            coach=coach,
            learner=None,
            memory=memory,
            planner_interval=planner_interval,
            max_steps=max_steps,
            reflexion_enabled=reflexion_enabled,
            reflexion_read_only=reflexion_read_only,
        )
