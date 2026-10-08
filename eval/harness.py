"""Evaluation harness for MiniWoB++ tasks."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Protocol

import yaml

from envs.browsergym_client import BrowserGymEnvWrapper
from eval.metrics import (
    EpisodeStats,
    average_reward,
    normalized_return,
    steps_to_success,
    success_rate,
)

logger = logging.getLogger(__name__)


class AgentProtocol(Protocol):
    def run_episode(self, task_id: str) -> Dict[str, float]: ...


@dataclass
class EvalConfig:
    frozen: bool
    episodes: int
    online_updates: bool


@contextmanager
def evaluation_mode(agent: AgentProtocol, config: EvalConfig):
    """Restore learning and reflection settings even when evaluation fails."""
    training = not config.frozen and config.online_updates
    previous_training = getattr(agent, "training", True)
    previous_read_only = getattr(agent, "reflexion_read_only", None)
    try:
        if hasattr(agent, "set_training"):
            agent.set_training(training)
        if previous_read_only is not None and not training:
            agent.reflexion_read_only = True
        yield
    finally:
        if hasattr(agent, "set_training"):
            agent.set_training(previous_training)
        if previous_read_only is not None:
            agent.reflexion_read_only = previous_read_only


def load_task_list(path: str) -> Dict[str, Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return {task["id"]: task for task in data["tasks"]}


def evaluate_agent(
    agent: AgentProtocol,
    env_factory: Callable[[], BrowserGymEnvWrapper],
    task_config: Dict[str, Dict],
    eval_config: EvalConfig,
) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    results = []
    intervention_totals: float = 0.0
    intervention_count = 0
    for _ in range(eval_config.episodes):
        with evaluation_mode(agent, eval_config):
            outcome = agent.run_episode(task_config["id"])
        if "coach_interventions" in outcome:
            intervention_totals += float(outcome["coach_interventions"])
            intervention_count += 1
        results.append(
            EpisodeStats(
                success=bool(outcome.get("success", 0.0)),
                reward=float(outcome.get("reward", 0.0)),
                steps=int(outcome.get("steps", outcome.get("episode_steps", 0))),
            )
        )
        if not eval_config.frozen and eval_config.online_updates:
            logger.debug("Online update placeholder for continual learning.")
    stats["success_rate"] = success_rate(results)
    stats["avg_reward"] = average_reward(results)
    stats["steps_to_success"] = steps_to_success(results)
    stats["normalized_return"] = normalized_return(
        results, task_config.get("max_episode_steps", 1)
    )
    if intervention_count:
        stats["avg_coach_interventions"] = intervention_totals / intervention_count
    return stats
