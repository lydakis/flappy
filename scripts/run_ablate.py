#!/usr/bin/env python3
"""Ablation runner for FLAPPY."""

from __future__ import annotations

import argparse
import logging

from agents.hybrid import HybridAgent
from envs.browsergym_client import BrowserGymEnvWrapper
from llm.coach import Coach
from llm.memory import load_memory
from llm.openai_client import OpenAIPlannerClient
from rl.rnd_ppo_agent import PPORNDLearner, RNDConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLAPPY ablation harness")
    parser.add_argument("--env", default="miniwob/click-checkboxes")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--disable-rnd", action="store_true")
    parser.add_argument("--disable-reflexion", action="store_true")
    parser.add_argument("--planner-interval", type=int, default=10)
    parser.add_argument("--memory", default="memory.jsonl")
    return parser.parse_args()


def make_env(env_id: str) -> BrowserGymEnvWrapper:
    return BrowserGymEnvWrapper(env_id=env_id)


def main() -> None:
    args = parse_args()
    llm_client = OpenAIPlannerClient()
    coach = Coach(llm_client)
    memory = load_memory(args.memory)
    rnd_config = RNDConfig(
        intrinsic_weight=0.0 if args.disable_rnd else 1.0,
    )
    learner = PPORNDLearner(env_fn=lambda: make_env(args.env), rnd_config=rnd_config)
    agent = HybridAgent(
        env=make_env(args.env),
        coach=coach,
        learner=learner,
        memory=memory,
        planner_interval=args.planner_interval,
        reflexion_enabled=not args.disable_reflexion,
    )
    learner.learn(total_timesteps=args.steps)
    logger.info("Ablation run completed.")


if __name__ == "__main__":
    main()
