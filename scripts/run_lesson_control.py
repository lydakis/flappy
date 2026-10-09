#!/usr/bin/env python3
"""Positive control: can the student learn the made-up convention from a lesson?

For each lesson size, start from a fresh adapter, run SFT on the lesson's worked
examples, and measure greedy accuracy on held-out programs (including program
lengths beyond the lesson) and on the original skills (forgetting check). No
tutor or API calls.

    .venv/bin/python scripts/run_lesson_control.py --lesson-sizes 0 20 60 200
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from student.agent import evaluate, evaluation_set
from student.model import DEFAULT_MODEL, HFStudent
from world.board import World
from world.conventions import ConventionFamily, lesson


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lesson-sizes", type=int, nargs="+", default=[0, 20, 60, 200])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-per-length", type=int, default=15)
    parser.add_argument("--retention-per-cell", type=int, default=2)
    parser.add_argument("--lesson-max-len", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "logs/curious-student/lesson-control"
    )
    args = parser.parse_args(argv)

    conv = World([ConventionFamily()], seed=args.seed)
    rng = random.Random(10_000 + args.seed)
    conv_eval = [
        conv.sample("conv.ops", length, rng)
        for length in (1, 2, 3, 4)
        for _ in range(args.eval_per_length)
    ]
    original = World(seed=args.seed)
    retention_eval = evaluation_set(
        original, args.retention_per_cell, 20_000 + args.seed
    )

    results = []
    for size in args.lesson_sizes:
        started = time.monotonic()
        model = HFStudent(args.model, device=args.device, lr=args.lr, seed=args.seed)
        examples = (
            lesson(size, random.Random(args.seed), max_len=args.lesson_max_len)
            if size
            else []
        )
        order = random.Random(args.seed)
        steps = 0
        for _ in range(args.epochs if examples else 0):
            order.shuffle(examples)
            for start in range(0, len(examples), args.batch_size):
                model.train_step(examples[start : start + args.batch_size])
                steps += 1
        row = {
            "lesson_size": size,
            "sft_steps": steps,
            "conv": evaluate(conv, model, conv_eval),
            "original_skills": evaluate(original, model, retention_eval),
            "seconds": round(time.monotonic() - started, 1),
        }
        results.append(row)
        print(json.dumps({k: row[k] for k in ("lesson_size", "sft_steps", "seconds")}))
        print("  conv", {k: round(v, 2) for k, v in row["conv"].items()})
        print(
            "  original",
            {k: round(v, 2) for k, v in row["original_skills"].items() if "/" not in k},
        )
        del model

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"seed{args.seed}-{time.strftime('%Y%m%dt%H%M%S')}.json"
    path.write_text(
        json.dumps(
            {"args": vars(args) | {"output": str(args.output)}, "results": results},
            indent=2,
        )
    )
    return {"results": results, "path": str(path)}


if __name__ == "__main__":
    main()
