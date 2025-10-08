#!/usr/bin/env python3
"""Unsupervised exploration runs for FLAPPY."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pathlib
import sys

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from agents.baseline_rl import PureRLAgent
from agents.coach_baseline import CoachRandomAgent
from agents.hybrid import HybridAgent
from envs.browsergym_client import BrowserGymEnvWrapper
from llm.coach import Coach
from llm.memory import load_memory
from llm.openai_client import OpenAIPlannerClient
from rl.rnd_ppo_agent import PPORNDLearner, RNDConfig

try:  # pragma: no cover - optional dependency
    from torch.utils.tensorboard import SummaryWriter  # type: ignore[import]
except ImportError:  # pragma: no cover
    SummaryWriter = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLAPPY unsupervised exploration")
    parser.add_argument(
        "--agent", choices=["hybrid", "coach_random", "baseline_rl"], default="hybrid"
    )
    parser.add_argument("--env", default="browsergym/miniwob.click-checkboxes")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--intrinsic", choices=["rnd", "none"], default="rnd")
    parser.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        default=True,
        help="Run Chromium in headless mode (default)",
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Disable headless mode to watch the browser window",
    )
    parser.add_argument("--memory", default="memory.jsonl")
    parser.add_argument("--resume-from", default=None, help="Path to a learner checkpoint")
    parser.add_argument("--save-path", default=None, help="Where to store learner checkpoints")
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save checkpoint every N learner steps (0 disables periodic saves)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Episodes between progress logs during explore runs",
    )
    parser.add_argument("--log-file", default=None, help="CSV file to append episode metrics")
    parser.add_argument(
        "--tensorboard",
        default=None,
        help="Directory for TensorBoard summaries (requires torch.utils.tensorboard)",
    )
    parser.add_argument(
        "--action-trace-file",
        default=None,
        help="Optional JSONL file capturing per-episode action traces",
    )
    return parser.parse_args()


def make_env(env_id: str, headless: bool) -> BrowserGymEnvWrapper:
    return BrowserGymEnvWrapper(env_id=env_id, headless=headless)


def main() -> None:
    args = parse_args()
    env_id = args.env

    if args.agent == "hybrid":
        llm_client = OpenAIPlannerClient()
        coach = Coach(llm_client)
        memory = load_memory(args.memory)
        learner = PPORNDLearner(
            rnd_config=RNDConfig(intrinsic_weight=1.0 if args.intrinsic == "rnd" else 0.0)
        )
        if args.resume_from:
            learner.load(args.resume_from)
            logger.info(
                "Resumed learner from %s (total_steps=%d)", args.resume_from, learner.total_steps
            )
        agent = HybridAgent(
            env=make_env(env_id, args.headless),
            coach=coach,
            learner=learner,
            memory=memory,
        )
        next_save_step = None
        if args.save_path and args.save_every > 0:
            ensure_parent_dir(args.save_path)
            next_save_step = learner.total_steps + args.save_every
        episodes = 0
        steps_collected = 0

        csv_writer = None
        csv_file_handle = None
        if args.log_file:
            ensure_parent_dir(args.log_file)
            file_exists = os.path.exists(args.log_file)
            csv_file_handle = open(args.log_file, "a", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file_handle)
            if not file_exists or os.stat(args.log_file).st_size == 0:
                csv_writer.writerow(
                    [
                        "episode",
                        "learner_steps",
                        "episode_steps",
                        "reward",
                        "intrinsic_reward",
                        "success",
                        "coach_interventions",
                    ]
                )

        tb_writer = None
        if args.tensorboard:
            if SummaryWriter is None:
                logger.warning(
                    "torch.utils.tensorboard is unavailable; ignoring --tensorboard=%s",
                    args.tensorboard,
                )
            else:
                ensure_parent_dir(os.path.join(args.tensorboard, "dummy"))
                tb_writer = SummaryWriter(log_dir=args.tensorboard)

        trace_file_handle = None
        if args.action_trace_file:
            ensure_parent_dir(args.action_trace_file)
            trace_file_handle = open(args.action_trace_file, "a", encoding="utf-8")
        while steps_collected < args.steps:
            stats = agent.run_episode(env_id)
            steps_collected += int(stats.get("steps", 0))
            episodes += 1
            if learner is not None and args.log_interval > 0 and episodes % args.log_interval == 0:
                logger.info(
                    "episode=%d | ep_steps=%d | reward=%.2f | intrinsic=%.2f | success=%.0f | coach_calls=%.0f | learner_steps=%d",
                    episodes,
                    stats.get("steps", 0),
                    stats.get("reward", 0.0),
                    stats.get("intrinsic_reward", 0.0),
                    stats.get("success", 0.0),
                    stats.get("coach_interventions", 0.0),
                    learner.total_steps,
                )
            if (
                args.save_path
                and args.save_every > 0
                and next_save_step is not None
                and learner.total_steps >= next_save_step
            ):
                learner.save(args.save_path)
                logger.info(
                    "Saved checkpoint to %s at learner step %d",
                    args.save_path,
                    learner.total_steps,
                )
                while learner.total_steps >= next_save_step:
                    next_save_step += args.save_every

            if csv_writer is not None:
                csv_writer.writerow(
                    [
                        episodes,
                        learner.total_steps,
                        stats.get("steps", 0),
                        stats.get("reward", 0.0),
                        stats.get("intrinsic_reward", 0.0),
                        stats.get("success", 0.0),
                        stats.get("coach_interventions", 0.0),
                    ]
                )
                csv_file_handle.flush()

            if tb_writer is not None:
                global_step = learner.total_steps
                tb_writer.add_scalar("episode/reward", stats.get("reward", 0.0), global_step)
                tb_writer.add_scalar(
                    "episode/intrinsic_reward", stats.get("intrinsic_reward", 0.0), global_step
                )
                tb_writer.add_scalar(
                    "episode/success", stats.get("success", 0.0), global_step
                )
                tb_writer.add_scalar(
                    "episode/coach_interventions",
                    stats.get("coach_interventions", 0.0),
                    global_step,
                )
                tb_writer.add_scalar(
                    "episode/steps", stats.get("steps", 0), global_step
                )

            if trace_file_handle is not None and "trace" in stats:
                trace_entry = {
                    "episode": episodes,
                    "learner_steps": learner.total_steps,
                    "success": stats.get("success", 0.0),
                    "reward": stats.get("reward", 0.0),
                    "intrinsic_reward": stats.get("intrinsic_reward", 0.0),
                    "actions": stats.get("trace", []),
                }
                trace_file_handle.write(json.dumps(trace_entry) + "\n")
                trace_file_handle.flush()
        logger.info("Hybrid exploration collected %d steps", steps_collected)
        if args.save_path:
            ensure_parent_dir(args.save_path)
            learner.save(args.save_path)
            logger.info(
                "Final checkpoint saved to %s at learner step %d",
                args.save_path,
                learner.total_steps,
            )

        if csv_file_handle is not None:
            csv_file_handle.close()
        if tb_writer is not None:
            tb_writer.close()
        if trace_file_handle is not None:
            trace_file_handle.close()
    elif args.agent == "coach_random":
        llm_client = OpenAIPlannerClient()
        coach = Coach(llm_client)
        memory = load_memory(args.memory)
        agent = CoachRandomAgent(
            env=make_env(env_id, args.headless),
            coach=coach,
            memory=memory,
        )
        episodes = max(1, args.steps // 1000)
        for _ in range(episodes):
            agent.run_episode(env_id)
        logger.info("Coach-random exploration rolled %d episodes.", episodes)
    else:
        agent = PureRLAgent(env_fn=lambda: make_env(env_id, args.headless))
        agent.learn(total_timesteps=args.steps)
        logger.info("Pure RL exploration complete.")


if __name__ == "__main__":
    main()
