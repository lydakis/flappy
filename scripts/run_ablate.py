#!/usr/bin/env python3
"""Ablation runner for FLAPPY."""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from agents.hybrid import HybridAgent
from envs.browsergym_client import BrowserGymEnvWrapper
from llm.coach import Coach
from llm.ideas import IdeaStore
from llm.memory import load_memory
from llm.openai_client import OpenAIPlannerClient
from rl.rnd_ppo_agent import PPORNDLearner, RNDConfig
from flappy.extract import DocumentExtractor
from flappy.memory import NoteStore
from flappy.rag import SimpleRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLAPPY ablation harness")
    parser.add_argument("--env", default="browsergym/miniwob.click-checkboxes")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--disable-rnd", action="store_true")
    parser.add_argument("--disable-reflexion", action="store_true")
    parser.add_argument("--planner-interval", type=int, default=10)
    parser.add_argument("--memory", default="memory.jsonl")
    parser.add_argument("--note-store", default="notes.jsonl")
    parser.add_argument("--idea-store", default="ideas.jsonl")
    parser.add_argument("--ddl-inject", action="store_true")
    parser.add_argument("--ddl-top-k", type=int, default=3)
    return parser.parse_args()


def make_env(env_id: str) -> BrowserGymEnvWrapper:
    return BrowserGymEnvWrapper(env_id=env_id)


def main() -> None:
    args = parse_args()
    llm_client = OpenAIPlannerClient()
    coach = Coach(llm_client)
    memory = load_memory(args.memory)
    note_store = NoteStore(pathlib.Path(args.note_store)) if args.note_store else None
    rag = SimpleRAG() if note_store else None
    extractor = DocumentExtractor() if note_store else None
    idea_store = IdeaStore(pathlib.Path(args.idea_store)) if args.idea_store else None
    rnd_config = RNDConfig(
        intrinsic_weight=0.0 if args.disable_rnd else 1.0,
    )
    learner = PPORNDLearner(rnd_config=rnd_config)
    agent = HybridAgent(
        env=make_env(args.env),
        coach=coach,
        learner=learner,
        memory=memory,
        planner_interval=args.planner_interval,
        reflexion_enabled=not args.disable_reflexion,
        note_store=note_store,
        rag=rag,
        extractor=extractor,
        idea_store=idea_store,
        ddl_inject=args.ddl_inject,
        ddl_top_k=args.ddl_top_k,
    )
    steps_collected = 0
    while steps_collected < args.steps:
        stats = agent.run_episode(args.env)
        steps_collected += int(stats.get("steps", 0))
    logger.info("Ablation run collected %d steps.", steps_collected)
    if hasattr(agent, "macro_registry"):
        logger.info(
            "Macro registry size=%d stats=%s",
            len(agent.macro_registry.list_for()),
            agent.macro_registry.stats(),
        )


if __name__ == "__main__":
    main()
