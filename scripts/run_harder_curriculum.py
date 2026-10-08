#!/usr/bin/env python3
"""Bounded harder-menu calibration and comparison; entirely local and offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.run_adaptive_curriculum import (
    ARMS,
    BLOCK_SIZE,
    BLOCKS,
    COST_MILESTONES,
    ENV_MILESTONES,
    LEDGERS,
    ORDER,
    RETENTION_CASES,
    SEEDS,
    Selector,
    Student,
    evaluate,
    fingerprint,
    resources,
    save,
)

OUTPUT = ROOT / "logs/harder-curriculum"
CALIBRATION_SEEDS = [101, 103]
CALIBRATION_LABELS_AFTER_WARMUP = 4608
MAX_SECONDS = 600
NAMESPACES = {
    "training": 11001,
    "heldout": 12001,
    "calibration_train": 13001,
    "calibration_test": 14001,
}
CANDIDATES = [
    {
        "id": "quadratic",
        "index": 0,
        "rules": ["x*y > 0", "0.5-x*x-y*y > 0", "x*x-y*y > 0"],
    },
    {
        "id": "wavy",
        "index": 1,
        "rules": ["x*y > 0", "0.5-x*x-y*y > 0", "y-0.5*sin(2*pi*x) > 0"],
    },
    {
        "id": "checkerboard",
        "index": 2,
        "rules": ["x*y > 0", "0.5-x*x-y*y > 0", "sin(2*pi*x)*sin(2*pi*y) > 0"],
    },
]


def score(features: np.ndarray, candidate: str) -> float:
    family = int(np.argmax(features[2:]))
    x, y = map(float, features[:2])
    if family == 0:
        return x * y
    if family == 1:
        return 0.5 - x * x - y * y
    if candidate == "quadratic":
        return x * x - y * y
    if candidate == "wavy":
        return y - 0.5 * np.sin(2 * np.pi * x)
    if candidate == "checkerboard":
        return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
    raise ValueError("Unknown candidate")


def oracle(features: np.ndarray, candidate: str) -> int:
    return int(score(features, candidate) > 0)


def make_case(
    candidate: str, task: int, index: int, seed: int, split: str = "training"
) -> dict:
    candidate_index = next(c["index"] for c in CANDIDATES if c["id"] == candidate)
    rng = np.random.default_rng(
        np.random.SeedSequence([NAMESPACES[split], candidate_index, seed, task, index])
    )
    family = task // 2
    low, high = (0.20, 2.0) if task % 2 == 0 else (0.02, 0.15)
    for _ in range(10000):
        x, y = rng.uniform(-1, 1, 2)
        features = np.array(
            [x, y, *[float(k == family) for k in range(3)]], dtype=np.float32
        )
        margin = score(features, candidate)
        if low <= abs(margin) <= high and int(margin > 0) == index % 2:
            return {
                "id": f"{candidate}:{split}:{seed}:{task}:{index}",
                "candidate": candidate,
                "task": task,
                "features": features.tolist(),
                "label": int(margin > 0),
            }
    raise RuntimeError("Harder case generation failed")


def dataset(candidate: str, split: str, seed: int) -> list:
    return [
        make_case(candidate, task, i, seed, split)
        for task in range(6)
        for i in range(256)
    ]


def qualifies(warm: dict, learned: dict) -> dict:
    """Only independent calibration results enter selection, never main outcomes."""
    checks = {
        "warmup_macro_at_most_85_percent": warm["macro_accuracy"] <= 0.85,
        "at_least_two_hard_tasks_at_most_80_percent": sum(
            warm["per_task"][t]["teacher_free_accuracy"] <= 0.80 for t in [1, 3, 5]
        )
        >= 2,
        "supervised_macro_at_least_90_percent": learned["macro_accuracy"] >= 0.90,
        "supervised_every_task_at_least_80_percent": min(
            t["teacher_free_accuracy"] for t in learned["per_task"]
        )
        >= 0.80,
        "macro_gain_at_least_15_points": learned["macro_accuracy"]
        - warm["macro_accuracy"]
        >= 0.15,
    }
    return {"passed": all(checks.values()), "checks": checks}


def first_qualifying(results: list) -> str | None:
    for candidate in CANDIDATES:
        rows = [r for r in results if r["candidate"] == candidate["id"]]
        if (
            len(rows) == len(CALIBRATION_SEEDS)
            and {r["seed"] for r in rows} == set(CALIBRATION_SEEDS)
            and all(
                qualifies(r["warmup"], r["supervised_final"])["passed"] for r in rows
            )
        ):
            return candidate["id"]
    return None


def prepare() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name in LEDGERS:
        assert json.loads((ROOT / name).read_text())["closed"]
    plan = {
        "purpose": "address previous post-warmup ceiling using bounded independent calibration, not search for adaptive wins",
        "candidate_order": CANDIDATES,
        "inputs": "x,y in [-1,1], one-hot family; no answer, score, margin, difficulty or candidate ID supplied to student",
        "tasks": [
            {
                "id": t,
                "family": t // 2,
                "difficulty": "wide_margin" if t % 2 == 0 else "near_boundary",
                "absolute_score_range": [0.20, 2.0] if t % 2 == 0 else [0.02, 0.15],
            }
            for t in range(6)
        ],
        "class_balance": "rejection sampling with alternating label target per independent task stream; no noisy labels",
        "calibration": {
            "maximum_candidates": 3,
            "seeds": CALIBRATION_SEEDS,
            "warmup": "same six 32-case blocks and active query rule as main; 192 cases",
            "then_full_supervision_cases": CALIBRATION_LABELS_AFTER_WARMUP,
            "positive_control": "round-robin six tasks after warmup, one requested-label update per case; separate from independence claims",
            "maximum_total_practice_cases": 3
            * 2
            * (192 + CALIBRATION_LABELS_AFTER_WARMUP),
            "evaluation": "256 cases per task, private calibration_test namespace, seed99101; never the final probe",
            "gates_each_seed": "warmup macro<=0.85 and >=2 near-boundary tasks<=0.80; fully supervised macro>=0.90 and every task>=0.80; macro gain>=0.15",
            "choice": "first candidate in fixed order satisfying every gate in both seeds; stop testing candidates after first success",
            "if_none": "stop with calibration blocker; no main comparison, new candidates or retuning",
            "selection_exclusions": "no adaptive/random/fixed results or main held-out measurements used",
        },
        "seeds": SEEDS,
        "arms": ARMS,
        "student": "unchanged previous Student: 2x64 MLP, five inputs and one zero goal, 2 answer outputs, CE/Adam0.001, clip1, one update per requested label, current example+31 queried-label replay samples",
        "teacher_query_rule": "unchanged max answer probability<0.90; always available; not a learned ask-action policy; no updates on unqueried examples",
        "scheduler": "unchanged hand-designed progress EMA: Q[t]=.7Q[t]+.3*mean positive queried-example log-probability gain; P(t)=.9*(Q[t]+1e-6)/sum(Q+1e-6)+.1/6; unqueried score0; no learned choice-policy network",
        "common_warmup": ORDER,
        "blocks": BLOCKS,
        "cases_per_block": BLOCK_SIZE,
        "practice_cases_per_arm_seed": BLOCKS * BLOCK_SIZE,
        "random": "uniform tasks after common warmup",
        "fixed": "31 blocks per task after common warmup, order0,2,4,1,3,5; 1024 cases/task",
        "paired_streams": "same model/replay/selector initialization within seed; same kth case within task across schedules",
        "heldout": "256 balanced cases per task; main heldout namespace seed20261009; same fixed probe at every checkpoint; never used for selection, gate calibration or stopping",
        "environment_checkpoints": ENV_MILESTONES,
        "teacher_cost_checkpoints": COST_MILESTONES,
        "matched_cost": "exact labels=exact optimizer steps; compare common checkpoints and state environment consumption separately",
        "retention_probe": {
            "cases": RETENTION_CASES,
            "task": 5,
            "query_rule_unchanged": True,
            "acquired_nonfocus_threshold": 0.85,
            "replay_enabled": True,
        },
        "teacher_ceiling_per_run": BLOCKS * BLOCK_SIZE + RETENTION_CASES,
        "teacher_budget": "maximum equals total cases and never denies help; no forced declining help",
        "primary_success": "same criterion as easy study: adaptive >=2 percentage points better than both baselines in >=2/3 seeds at final matched practice AND highest label checkpoint reached by all nine; descriptive, not significance testing",
        "help_success": "balanced fixed-probe help drops>=0.20 from post-warmup to final, final accuracy>=0.85, confident error fraction<=0.05; also show every task separately",
        "retention_metrics": "pre/post accuracy on acquired nonfocus tasks, retained>=0.85, number of new updates, and peak-to-final drops",
        "resources": "CPU only, 1 Torch thread, nice15; RSS<=1800MiB; disk>=8GiB; 600 seconds total calibration and 600 seconds total main phase",
        "stops": "fixed budgets; no qualifying candidate; nonfinite update; evaluation mutation; source/data/ledger mismatch; resource/time threshold",
        "ledger_hashes": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in LEDGERS
        },
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
    }
    save(OUTPUT / "plan.json", plan)
    names = [
        "scripts/run_harder_curriculum.py",
        "scripts/run_adaptive_curriculum.py",
        "rl/policy.py",
        "logs/harder-curriculum/plan.json",
    ]
    for candidate in CANDIDATES:
        for prefix, split, seed in [
            ("calibration-probe", "calibration_test", 99101),
            ("final-probe", "heldout", 20261009),
        ]:
            path = OUTPUT / f"{prefix}-{candidate['id']}.json"
            save(path, dataset(candidate["id"], split, seed))
            names.append(str(path.relative_to(ROOT)))
    save(
        OUTPUT / "source-hashes.json",
        {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in names
        },
    )


def verify_inputs() -> None:
    for name, expected in json.loads(
        (OUTPUT / "source-hashes.json").read_text()
    ).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    plan = json.loads((OUTPUT / "plan.json").read_text())
    for name, expected in plan["ledger_hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected


def calibrate_one(candidate: str, seed: int, started: float, directory: Path) -> dict:
    student = Student(seed)
    probe = json.loads((OUTPUT / f"calibration-probe-{candidate}.json").read_text())
    fresh = evaluate(student, probe)
    counters = [0] * 6
    records = []
    for task in ORDER:
        for _ in range(BLOCK_SIZE):
            example = make_case(
                candidate, task, counters[task], seed, "calibration_train"
            )
            counters[task] += 1
            outcome = student.practise(
                np.array(example["features"], dtype=np.float32),
                lambda x: oracle(x, candidate),
            )
            records.append({"phase": "active_warmup", "example": example, **outcome})
    warmup = evaluate(student, probe)
    warm_queries = student.queries
    for index in range(CALIBRATION_LABELS_AFTER_WARMUP):
        if index % 128 == 0:
            resources()
            if time.monotonic() - started > MAX_SECONDS:
                raise TimeoutError("Hard-menu calibration time limit")
        task = index % 6
        example = make_case(candidate, task, counters[task], seed, "calibration_train")
        counters[task] += 1
        update = student.learn_label(
            np.array(example["features"], dtype=np.float32), example["label"]
        )
        records.append({"phase": "full_supervision", "example": example, **update})
    trained = evaluate(student, probe)
    result = {
        "candidate": candidate,
        "seed": seed,
        "fresh": fresh,
        "warmup": warmup,
        "warmup_labels": warm_queries,
        "supervised_final": trained,
        "total_oracle_labels": student.queries,
        "qualification": qualifies(warmup, trained),
        "resources": resources(),
        "wall_seconds": time.monotonic() - started,
    }
    save(directory / f"{candidate}-{seed}.json", result)
    with (directory / f"{candidate}-{seed}-training.jsonl").open("x") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    print(
        json.dumps(
            {
                "candidate": candidate,
                "seed": seed,
                "warmup_accuracy": warmup["macro_accuracy"],
                "supervised_accuracy": trained["macro_accuracy"],
                "passed": result["qualification"]["passed"],
            }
        ),
        flush=True,
    )
    return result


def calibrate() -> None:
    verify_inputs()
    directory = OUTPUT / "calibration"
    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    results = []
    selected = None
    for candidate in CANDIDATES:
        for seed in CALIBRATION_SEEDS:
            results.append(calibrate_one(candidate["id"], seed, started, directory))
        selected = first_qualifying(results)
        if selected:
            break
    verify_inputs()
    summary = {
        "selected": selected,
        "calibration_runs": results,
        "selection_uses_only_calibration": True,
        "no_main_comparison_run_yet": True,
        "new_paid_calls": 0,
        "source_and_ledgers_unchanged": True,
        "wall_seconds": time.monotonic() - started,
        "resources": resources(),
    }
    if not selected:
        summary["blocker"] = (
            "No declared candidate meets independent headroom and learnability gates in both calibration seeds"
        )
    save(OUTPUT / "calibration-summary.json", summary)
    print(
        json.dumps(
            {
                "selected": selected,
                "runs": len(results),
                "wall_seconds": summary["wall_seconds"],
            }
        ),
        flush=True,
    )


def run_one(
    candidate: str, arm: str, seed: int, probe: list, directory: Path, started: float
) -> dict:
    directory.mkdir(exist_ok=False)
    student, selector = Student(seed), Selector(arm, seed)
    initial_hash = fingerprint(student.model.state_dict())
    counters, requested, checkpoints = [0] * 6, 0, []

    def measure(kind: str, cases_seen: int):
        before = fingerprint(selector.state())
        outcome = evaluate(student, probe)
        assert fingerprint(selector.state()) == before
        row = {
            "kind": kind,
            "cases_seen": cases_seen,
            "teacher_labels": student.queries,
            "optimizer_steps": student.updates,
            "result": outcome,
        }
        save(directory / f"evaluation-{kind}-{cases_seen}-{student.queries}.json", row)
        checkpoints.append(row)

    measure("environment", 0)
    with (directory / "cases.jsonl").open("x") as case_log, (
        directory / "blocks.jsonl"
    ).open("x") as block_log:
        for block in range(BLOCKS + RETENTION_CASES // BLOCK_SIZE):
            if time.monotonic() - started > MAX_SECONDS:
                raise TimeoutError("Hard-menu comparison time limit")
            retention = block >= BLOCKS
            task, probabilities = (
                (5, np.eye(6)[5].tolist()) if retention else selector.choose(block)
            )
            before = selector.state()
            progress = []
            queries_before = student.queries
            for offset in range(BLOCK_SIZE):
                example = make_case(candidate, task, counters[task], seed)
                counters[task] += 1

                def teacher(x):
                    nonlocal requested
                    requested += 1
                    return oracle(x, candidate)

                outcome = student.practise(
                    np.array(example["features"], dtype=np.float32), teacher
                )
                assert requested == student.queries == student.updates
                assert requested <= BLOCKS * BLOCK_SIZE + RETENTION_CASES
                progress.append(outcome["positive_loss_reduction"])
                cases_seen = block * BLOCK_SIZE + offset + 1
                case_log.write(
                    json.dumps(
                        {
                            "block": block,
                            "phase": "retention" if retention else "curriculum",
                            "cases_seen": cases_seen,
                            "example": example,
                            **outcome,
                            "shadow_correct": outcome["prediction_before_help"]
                            == example["label"],
                            "executed_correct": outcome["executed_answer"]
                            == example["label"],
                            "cumulative_teacher_labels": student.queries,
                        }
                    )
                    + "\n"
                )
                if (
                    outcome["asked"]
                    and student.queries in COST_MILESTONES
                    and not retention
                ):
                    measure("teacher_cost", cases_seen)
            if not retention:
                selector.observe(task, float(np.mean(progress)))
            row = {
                "block": block,
                "phase": "retention" if retention else "curriculum",
                "task": task,
                "selection_probabilities": probabilities,
                "scheduler_before": before,
                "scheduler_after": selector.state(),
                "block_teacher_labels": student.queries - queries_before,
                "cumulative_teacher_labels": student.queries,
                "mean_progress": float(np.mean(progress)),
                "resources": resources(),
                "wall_seconds": time.monotonic() - started,
            }
            block_log.write(json.dumps(row) + "\n")
            case_log.flush()
            block_log.flush()
            cases_seen = (block + 1) * BLOCK_SIZE
            if cases_seen in ENV_MILESTONES:
                measure("environment", cases_seen)
            if (block + 1) % 32 == 0:
                print(
                    json.dumps(
                        {
                            "candidate": candidate,
                            "arm": arm,
                            "seed": seed,
                            "cases": cases_seen,
                            "teacher_labels": student.queries,
                            "wall_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
    measure("after_retention", BLOCKS * BLOCK_SIZE + RETENTION_CASES)
    directory_cp = ROOT / "checkpoints/harder-curriculum"
    directory_cp.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "candidate": candidate,
            "model": student.model.state_dict(),
            "optimizer": student.optimizer.state_dict(),
            "teacher_labels": student.queries,
            "replay_x": torch.from_numpy(np.stack(student.replay_x)),
            "replay_y": student.replay_y,
            "selector": selector.state(),
        },
        directory_cp / f"{arm}-{seed}.pt",
    )
    result = {
        "candidate": candidate,
        "arm": arm,
        "seed": seed,
        "initial_weight_hash": initial_hash,
        "final_weight_hash": fingerprint(student.model.state_dict()),
        "teacher_labels_including_retention": student.queries,
        "optimizer_steps": student.updates,
        "task_case_counts_including_retention": counters,
        "checkpoints": checkpoints,
        "resources": resources(),
        "wall_seconds": time.monotonic() - started,
    }
    save(directory / "summary.json", result)
    return result


def run() -> None:
    verify_inputs()
    calibration = json.loads((OUTPUT / "calibration-summary.json").read_text())
    selected = first_qualifying(calibration["calibration_runs"])
    if not selected or selected != calibration["selected"]:
        raise RuntimeError("No qualifying declared candidate; comparison blocked")
    directory = OUTPUT / "runs"
    directory.mkdir(exist_ok=False)
    probe = json.loads((OUTPUT / f"final-probe-{selected}.json").read_text())
    # Compatibility for the reused read-only analysis schema; no easy results altered.
    learnability = {
        "passed": True,
        "selected_candidate": selected,
        "independent_calibration": calibration,
    }
    save(OUTPUT / "learnability.json", learnability)
    started = time.monotonic()
    status = {
        "complete": False,
        "selected_candidate": selected,
        "new_paid_calls": 0,
        "teacher": "simulated oracle",
    }
    try:
        summaries = []
        for seed in SEEDS:
            for arm in ARMS:
                summaries.append(
                    run_one(
                        selected, arm, seed, probe, directory / f"{arm}-{seed}", started
                    )
                )
                save(OUTPUT / "summaries.json", summaries)
        status["complete"] = True
    except Exception as exc:
        status["stop_reason"] = str(exc)
        raise
    finally:
        verify_inputs()
        status.update(
            {
                "source_and_ledgers_unchanged": True,
                "wall_seconds": time.monotonic() - started,
                "resources": resources(),
            }
        )
        save(OUTPUT / "status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "calibrate", "run"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.phase == "prepare":
        prepare()
    elif args.phase == "calibrate":
        calibrate()
    else:
        run()
