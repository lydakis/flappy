#!/usr/bin/env python3
"""Run the curious student in the text world for a short, bounded session.

Example smoke run (tutor calls go through the fail-closed ledger):

    .venv/bin/python scripts/run_curious_student.py --arm progress --ticks 20 \
        --max-minutes 6 --max-spend 0.50 --init-ledger

Writes ``events.jsonl``, TensorBoard scalars and ``summary.json`` under
``logs/curious-student/<run-id>/``. The tutor ledger lives at
``logs/curious-student/tutor-ledger.json`` and caps cumulative reservations at $5.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm.tutor_ledger import DEFAULT_MODEL as TUTOR_MODEL
from llm.tutor_ledger import LedgerTutorClient, TutorLedger
from student.agent import (
    ARMS,
    CuriousStudent,
    LoopConfig,
    evaluate,
    evaluation_set,
)
from student.model import DEFAULT_MODEL, HFStudent
from student.runlog import RunLog
from student.tutor import ScriptedTutorBackend, Tutor
from world.board import Wallet, World


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=ARMS, default="progress")
    parser.add_argument("--ticks", type=int, default=20)
    parser.add_argument("--max-minutes", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-every", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=2)
    parser.add_argument("--start-credits", type=float, default=10.0)
    parser.add_argument("--eval-per-cell", type=int, default=2)
    parser.add_argument(
        "--tutor",
        choices=("openai", "scripted"),
        default="openai",
        help="'scripted' is an offline canned tutor for dry runs",
    )
    parser.add_argument("--tutor-model", default=TUTOR_MODEL)
    parser.add_argument("--key-file", type=Path, default=ROOT / ".env.local")
    parser.add_argument(
        "--ledger", type=Path, default=ROOT / "logs/curious-student/tutor-ledger.json"
    )
    parser.add_argument(
        "--init-ledger", action="store_true", help="create the ledger if it is missing"
    )
    parser.add_argument(
        "--max-spend",
        type=float,
        default=0.50,
        help="per-run USD reservation allowance (the ledger caps all runs at $5)",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "logs/curious-student")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--save-adapter", action="store_true")
    return parser.parse_args(argv)


def build_tutor(args, run_id: str):
    """Return (Tutor | None, spend callable | None, closer)."""
    if args.arm == "no_tutor":
        return None, None, lambda: None
    if args.tutor == "scripted":
        return Tutor(ScriptedTutorBackend()), None, lambda: None
    ledger = TutorLedger(args.ledger, args.tutor_model)
    if not args.ledger.exists():
        if not args.init_ledger:
            raise SystemExit("Tutor ledger missing; pass --init-ledger to create it")
        ledger.initialize()
    client = LedgerTutorClient.from_key_file(ledger, args.key_file)
    client.bind_run(run_id, args.max_spend)
    return Tutor(client), (lambda: ledger.totals(run_id)), client.close


def online_summary(events: list[dict]) -> dict:
    cells: dict[str, list[bool]] = defaultdict(list)
    for e in events:
        if e["type"] == "attempt":
            cells[f"{e['skill']}"].append(e["passed"])
            cells[f"{e['skill']}/{e['mode']}"].append(e["passed"])
    return {
        k: {"n": len(v), "success": sum(v) / len(v)} for k, v in sorted(cells.items())
    }


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    run_id = f"{args.arm}-s{args.seed}-{time.strftime('%Y%m%dt%H%M%S')}"
    deadline = time.monotonic() + args.max_minutes * 60
    log = RunLog(args.output / run_id, tensorboard=not args.no_tensorboard)
    tutor, spend, close_tutor = build_tutor(args, run_id)
    world = World(seed=args.seed)
    eval_tasks = evaluation_set(world, args.eval_per_cell, seed=10_000 + args.seed)

    started = time.monotonic()
    model = HFStudent(args.model, device=args.device, lr=args.lr, seed=args.seed)
    load_sec = time.monotonic() - started
    started = time.monotonic()
    before = evaluate(world, model, eval_tasks)
    eval_sec = time.monotonic() - started
    log.event({"type": "eval", "when": "before", "results": before})

    student = CuriousStudent(
        world,
        model,
        config=LoopConfig(
            arm=args.arm, train_every=args.train_every, train_steps=args.train_steps
        ),
        wallet=Wallet(args.start_credits),
        tutor=tutor,
        log=log,
        spend=spend,
        seed=args.seed,
    )
    started = time.monotonic()
    # Leave room for the closing evaluation inside the wall-clock budget.
    while student.ticks < args.ticks and time.monotonic() + eval_sec < deadline:
        student.tick()
    loop_sec = time.monotonic() - started
    after = evaluate(world, model, eval_tasks)
    log.event({"type": "eval", "when": "after", "results": after})
    close_tutor()

    summary = {
        "run_id": run_id,
        "arm": args.arm,
        "model": args.model,
        "device": model.device,
        "tutor": None if tutor is None else args.tutor,
        "tutor_model": args.tutor_model if args.tutor == "openai" else None,
        "ticks": student.ticks,
        "timing_sec": {
            "model_load": round(load_sec, 1),
            "eval": round(eval_sec, 1),
            "loop": round(loop_sec, 1),
            "per_tick": round(loop_sec / max(student.ticks, 1), 1),
        },
        "online": online_summary(log.events),
        "eval_before": before,
        "eval_after": after,
        "wallet": {
            "balance": student.wallet.balance,
            "earned": student.wallet.earned,
            "spent_on_tutor": student.wallet.spent,
        },
        "tutor_calls": student.tutor_counts(),
        "tutor_stopped": student.tutor_stopped,
        "tutor_spend_usd": spend() if spend else None,
        "train": {
            "steps": len(student.losses),
            "first_loss": student.losses[0] if student.losses else None,
            "last_loss": student.losses[-1] if student.losses else None,
            "buffer": len(student.buffer),
            "tutor_examples": sum(src == "tutor" for *_, src in student.buffer),
        },
    }
    (args.output / run_id / "summary.json").write_text(json.dumps(summary, indent=2))
    if args.save_adapter:
        model.save_adapter(ROOT / "checkpoints/curious-student" / run_id)
    log.close()
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
