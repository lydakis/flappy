#!/usr/bin/env python3
"""Run the day-dreaming loop to generate background ideas."""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import random
import sys
from typing import Dict

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from llm.ddl import DaydreamingConfig, DaydreamingLoop
from llm.ideas import IdeaStore
from llm.memory import load_memory
from llm.openai_client import OpenAIPlannerClient
from flappy.memory import NoteStore

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FLAPPY day-dreaming loop")
    parser.add_argument("--memory", default="memory.jsonl", help="Path to episodic memory JSONL")
    parser.add_argument("--notes", default="notes.jsonl", help="Path to note store JSONL")
    parser.add_argument("--ideas", default="ideas.jsonl", help="Path to idea store JSONL")
    parser.add_argument("--pairs", type=int, default=500, help="Number of concept pairs to sample")
    parser.add_argument("--k", type=int, default=2, help="Concepts per sample (≥2)")
    parser.add_argument(
        "--thresholds",
        default="7,6,6",
        help="Comma-separated thresholds for novelty,coherence,usefulness",
    )
    parser.add_argument("--max-accept", type=int, default=50, help="Max ideas to accept per run")
    parser.add_argument(
        "--time-budget-sec",
        type=float,
        default=None,
        help="Optional wall-clock limit in seconds",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="Optional input token budget before stopping",
    )
    parser.add_argument("--sleep-sec", type=float, default=0.0, help="Sleep between accepted ideas")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed for sampling")
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable hash-based deduplication against existing ideas",
    )
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="Optional path to write run statistics as JSON",
    )
    return parser.parse_args()


def parse_thresholds(raw: str) -> Dict[str, float]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("Expected three comma-separated thresholds: novelty,coherence,usefulness")
    novelty, coherence, usefulness = (float(value) for value in parts)
    return {"novelty": novelty, "coherence": coherence, "usefulness": usefulness}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting day-dreaming loop with args: %s", args)

    idea_store = IdeaStore(pathlib.Path(args.ideas))
    memory_store = load_memory(args.memory) if args.memory else None
    note_store = NoteStore(pathlib.Path(args.notes)) if args.notes else None

    config = DaydreamingConfig(
        pairs=args.pairs,
        k=args.k,
        thresholds=parse_thresholds(args.thresholds),
        max_accept=args.max_accept,
        time_budget_sec=args.time_budget_sec,
        token_budget=args.token_budget,
        sleep_sec=args.sleep_sec,
        dedup=not args.no_dedup,
    )

    rng = random.Random(args.seed)
    client = OpenAIPlannerClient()
    loop = DaydreamingLoop(
        client=client,
        idea_store=idea_store,
        memory_store=memory_store,
        note_store=note_store,
        config=config,
        rng=rng,
    )

    stats = loop.run_once()
    logger.info(
        "DDL run complete: pairs=%d generated=%d accepted=%d duplicates=%d rejects=%d",
        stats.pairs_evaluated,
        stats.ideas_generated,
        stats.ideas_accepted,
        stats.duplicates_skipped,
        stats.threshold_rejects,
    )
    logger.info(
        "LLM usage: requests=%d input_tokens=%d output_tokens=%d latency=%.2fs",
        stats.requests,
        stats.input_tokens,
        stats.output_tokens,
        stats.latency_sec,
    )

    if args.metrics_json:
        payload = {
            "pairs_evaluated": stats.pairs_evaluated,
            "ideas_generated": stats.ideas_generated,
            "ideas_accepted": stats.ideas_accepted,
            "duplicates_skipped": stats.duplicates_skipped,
            "threshold_rejects": stats.threshold_rejects,
            "parse_failures": stats.parse_failures,
            "requests": stats.requests,
            "input_tokens": stats.input_tokens,
            "output_tokens": stats.output_tokens,
            "latency_sec": stats.latency_sec,
            "client_stats": client.stats_snapshot(),
        }
        metrics_path = pathlib.Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote metrics to %s", metrics_path)


if __name__ == "__main__":
    main()

