#!/usr/bin/env python3
"""Unsupervised exploration runs for FLAPPY."""

from __future__ import annotations

import argparse
import logging

from agents.baseline_rl import PureRLAgent
from agents.coach_baseline import CoachRandomAgent
from agents.hybrid import HybridAgent
from envs.browsergym_client import BrowserGymEnvWrapper
from llm.coach import Coach
from llm.memory import load_memory
from llm.openai_client import OpenAIPlannerClient
from rl.rnd_ppo_agent import PPORNDLearner, RNDConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLAPPY unsupervised exploration")
    parser.add_argument(
        "--agent", choices=["hybrid", "coach_random", "baseline_rl"], default="hybrid"
    )
    parser.add_argument("--env", default="miniwob/click-checkboxes")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--intrinsic", choices=["rnd", "none"], default="rnd")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--memory", default="memory.jsonl")
    return parser.parse_args()


def make_env(env_id: str, headless: bool) -> BrowserGymEnvWrapper:
    return BrowserGymEnvWrapper(env_id=env_id, headless=headless)


def main() -> None:
    args = parse_args()
    env_id = args.env
    headless = args.headless

    if args.agent == "hybrid":
        llm_client = OpenAIPlannerClient()
        coach = Coach(llm_client)
        memory = load_memory(args.memory)
        learner = PPORNDLearner(
            rnd_config=RNDConfig(intrinsic_weight=1.0 if args.intrinsic == "rnd" else 0.0)
        )
        agent = HybridAgent(
            env=make_env(env_id, headless),
            coach=coach,
            learner=learner,
            memory=memory,
        )
        steps_collected = 0
        while steps_collected < args.steps:
            stats = agent.run_episode(env_id)
            steps_collected += int(stats.get("steps", 0))
        logger.info("Hybrid exploration collected %d steps", steps_collected)
    elif args.agent == "coach_random":
        llm_client = OpenAIPlannerClient()
        coach = Coach(llm_client)
        memory = load_memory(args.memory)
        agent = CoachRandomAgent(
            env=make_env(env_id, headless),
            coach=coach,
            memory=memory,
        )
        episodes = max(1, args.steps // 1000)
        for _ in range(episodes):
            agent.run_episode(env_id)
        logger.info("Coach-random exploration rolled %d episodes.", episodes)
    else:
        agent = PureRLAgent(env_fn=lambda: make_env(env_id, headless))
        agent.learn(total_timesteps=args.steps)
        logger.info("Pure RL exploration complete.")


if __name__ == "__main__":
    main()
