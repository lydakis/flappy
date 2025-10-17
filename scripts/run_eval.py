#!/usr/bin/env python3
"""Evaluate FLAPPY agents on MiniWoB++ tasks."""

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

from agents.baseline_rl import PureRLAgent
from agents.coach_baseline import CoachRandomAgent
from agents.hybrid import HybridAgent
from envs.browsergym_client import BrowserGymEnvWrapper
from eval.harness import EvalConfig, evaluate_agent, load_task_list
from llm.coach import Coach
from llm.ideas import IdeaStore
from llm.memory import load_memory
from llm.openai_client import OpenAIPlannerClient
from rl.rnd_ppo_agent import PPORNDLearner
from flappy.extract import DocumentExtractor
from flappy.memory import NoteStore
from flappy.rag import SimpleRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FLAPPY agents")
    parser.add_argument("--agent", choices=["baseline_rl", "coach_random", "hybrid"])
    parser.add_argument("--tasks", default="eval/tasks.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--frozen", action="store_true", default=False)
    parser.add_argument("--checkpoint", help="Path to PPO checkpoint", default=None)
    parser.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        default=True,
        help="Run Chromium headless during evaluation (default)",
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Disable headless mode to watch evaluation",
    )
    parser.add_argument("--memory", default="memory.jsonl")
    parser.add_argument("--env", default=None, help="Override environment id")
    parser.add_argument("--note-store", default="notes.jsonl")
    parser.add_argument("--idea-store", default="ideas.jsonl")
    parser.add_argument(
        "--ddl-inject",
        action="store_true",
        help="Inject day-dreaming ideas alongside reflections",
    )
    parser.add_argument("--ddl-top-k", type=int, default=3)
    parser.add_argument("--qa-question", default=None, help="Optional question for RAG answer")
    return parser.parse_args()


def make_env(env_id: str, headless: bool) -> BrowserGymEnvWrapper:
    return BrowserGymEnvWrapper(env_id=env_id, headless=headless)


def main() -> None:
    args = parse_args()
    tasks = load_task_list(args.tasks)
    eval_cfg = EvalConfig(
        frozen=args.frozen, episodes=args.episodes, online_updates=not args.frozen
    )
    llm_client = OpenAIPlannerClient()
    coach = Coach(llm_client)
    memory = load_memory(args.memory)
    note_store = NoteStore(pathlib.Path(args.note_store)) if args.note_store else None
    rag = SimpleRAG() if note_store else None
    extractor = DocumentExtractor() if note_store else None
    idea_store = IdeaStore(pathlib.Path(args.idea_store)) if args.idea_store else None

    for task_id, task in tasks.items():
        if args.env and task_id != args.env:
            continue
        logger.info("Evaluating %s on %s", args.agent, task_id)
        env_id = task_id
        env_factory = lambda: make_env(env_id, args.headless)

        if args.agent == "baseline_rl":
            rl_agent = PureRLAgent(env_fn=env_factory)
            if args.checkpoint:
                rl_agent.model.load(args.checkpoint)  # type: ignore[attr-defined]
            agent = rl_agent
        elif args.agent == "coach_random":
            agent = CoachRandomAgent(
                env=env_factory(),
                coach=coach,
                memory=memory,
                reflexion_read_only=args.frozen,
                note_store=note_store,
                rag=rag,
                extractor=extractor,
                idea_store=idea_store,
                ddl_inject=args.ddl_inject,
                ddl_top_k=args.ddl_top_k,
            )
        else:
            learner = PPORNDLearner()
            if args.checkpoint:
                learner.load(args.checkpoint)
            agent = HybridAgent(
                env=env_factory(),
                coach=coach,
                learner=learner,
                memory=memory,
                reflexion_read_only=args.frozen,
                note_store=note_store,
                rag=rag,
                extractor=extractor,
                idea_store=idea_store,
                ddl_inject=args.ddl_inject,
                ddl_top_k=args.ddl_top_k,
            )
        results = evaluate_agent(agent, env_factory, {"id": env_id}, eval_cfg)
        logger.info("Results for %s: %s", task_id, results)
        if hasattr(agent, "macro_registry"):
            stats = agent.macro_registry.stats()
            logger.info(
                "Macro registry size=%d latest=%s",
                len(stats),
                getattr(agent, "_current_macro_name", None),
            )
        if args.qa_question and hasattr(agent, "answer_question"):
            answer = agent.answer_question(args.qa_question)
            if answer is not None:
                logger.info(
                    "QA answer: %s | citations=%s | confidence=%.2f",
                    answer.text,
                    answer.citations,
                    getattr(answer, "confidence", 0.0),
                )


if __name__ == "__main__":
    main()
