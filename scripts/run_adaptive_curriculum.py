#!/usr/bin/env python3
"""Offline active-learning curriculum diagnostic; no browser, API or credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from rl.policy import MaskedCategoricalPolicy, PolicyConfig

OUTPUT = ROOT / "logs/adaptive-curriculum"
SEEDS = [7, 19, 43]
ARMS = ["adaptive", "random", "fixed"]
ORDER = [0, 2, 4, 1, 3, 5]
BLOCKS = 192
BLOCK_SIZE = 32
RETENTION_CASES = 256
ENV_MILESTONES = [0, 192, 512, 1024, 2048, 4096, 6144]
COST_MILESTONES = [64, 128, 256, 512, 1024, 2048, 4096]
HELP_THRESHOLD = 0.90
REPLAY_BATCH = 32
VALIDATION_CASES = 3072
VALIDATION_SEED = 101
MAX_SECONDS = 600
LEDGERS: list[str] = []  # Offline runs have no provider-account dependency.
NAMESPACES = {
    "training": 1001,
    "heldout": 2001,
    "validation_train": 3001,
    "validation_test": 4001,
}
TASKS = [
    {
        "id": 2 * family + difficulty,
        "family": family,
        "rule": rule,
        "difficulty": "wide_margin" if difficulty == 0 else "near_boundary",
        "absolute_margin": [0.6, 1.4] if difficulty == 0 else [0.05, 0.35],
    }
    for family, rule in enumerate(["x > 0", "y > 0", "(x + y) / sqrt(2) > 0"])
    for difficulty in range(2)
]


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def resources() -> dict:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = usage.ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    disk = shutil.disk_usage(ROOT).free / 1024**3
    if rss > 1800 or disk < 8:
        raise RuntimeError("Curriculum resource boundary reached")
    return {
        "peak_rss_mib": rss,
        "free_disk_gib": disk,
        "cpu_seconds": usage.ru_utime + usage.ru_stime,
    }


def fingerprint(value) -> str:
    h = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            h.update(item.detach().cpu().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            h.update(item.tobytes())
        elif isinstance(item, dict):
            for key, val in item.items():
                h.update(str(key).encode())
                visit(val)
        elif isinstance(item, (list, tuple)):
            for val in item:
                visit(val)
        else:
            h.update(repr(item).encode())

    visit(value)
    return h.hexdigest()


def oracle(features: np.ndarray) -> int:
    family = int(np.argmax(features[2:]))
    score = [features[0], features[1], (features[0] + features[1]) / np.sqrt(2)][family]
    return int(score > 0)


def case(task: int, index: int, seed: int, split: str = "training") -> dict:
    spec = TASKS[task]
    rng = np.random.default_rng(
        np.random.SeedSequence([NAMESPACES[split], seed, task, index])
    )
    for _ in range(10000):
        x, y = rng.uniform(-1, 1, 2)
        features = np.array(
            [x, y, *[float(k == spec["family"]) for k in range(3)]], dtype=np.float32
        )
        margin = [
            float(features[0]),
            float(features[1]),
            float((features[0] + features[1]) / np.sqrt(2)),
        ][spec["family"]]
        if (
            spec["absolute_margin"][0] <= abs(margin) <= spec["absolute_margin"][1]
            and int(margin > 0) == index % 2
        ):
            return {
                "id": f"{split}:{seed}:{task}:{index}",
                "task": task,
                "features": features.tolist(),
                "label": int(margin > 0),
            }
    raise RuntimeError("Case generation failed")


def dataset(split: str, seed: int, count: int = 256) -> list:
    return [case(task, i, seed, split) for task in range(6) for i in range(count)]


class Student:
    """Learns an answer classifier; the query threshold itself is not trained."""

    def __init__(self, seed: int):
        torch.manual_seed(seed)
        self.model = MaskedCategoricalPolicy(
            PolicyConfig(
                state_dim=5,
                subgoal_dim=1,
                hidden_dim=64,
                action_dim=2,
                use_mask_head=False,
            )
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.rng = np.random.default_rng(seed + 70000)
        self.replay_x = []
        self.replay_y = []
        self.queries = 0
        self.updates = 0

    def state(self):
        return [
            self.model.state_dict(),
            self.optimizer.state_dict(),
            self.replay_x,
            self.replay_y,
            self.queries,
            self.updates,
            self.rng.bit_generator.state,
        ]

    @torch.no_grad()
    def probabilities(self, features: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(features, dtype=np.float32).reshape(-1, 5))
        return self.model(x, torch.zeros((len(x), 1))).softmax(-1).numpy()

    def learn_label(self, features: np.ndarray, label: int) -> dict:
        """One update per requested label: current example + 31 replay draws."""
        before = float(self.probabilities(features)[0, label])
        self.replay_x.append(np.asarray(features, dtype=np.float32).copy())
        self.replay_y.append(int(label))
        self.queries += 1
        indices = self.rng.integers(
            0, len(self.replay_x), REPLAY_BATCH - 1
        ).tolist() + [len(self.replay_x) - 1]
        x = torch.from_numpy(np.stack([self.replay_x[i] for i in indices]))
        y = torch.tensor([self.replay_y[i] for i in indices])
        loss = F.cross_entropy(self.model(x, torch.zeros((REPLAY_BATCH, 1))), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite student loss")
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 1.0, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.updates += 1
        after = float(self.probabilities(features)[0, label])
        return {
            "training_loss": float(loss.detach()),
            "positive_loss_reduction": max(
                0.0, float(np.log(max(after, 1e-12)) - np.log(max(before, 1e-12)))
            ),
        }

    def practise(self, features: np.ndarray, teacher) -> dict:
        probabilities = self.probabilities(features)[0]
        prediction = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
        ask = confidence < HELP_THRESHOLD
        result = {
            "prediction_before_help": prediction,
            "confidence": confidence,
            "asked": bool(ask),
            "positive_loss_reduction": 0.0,
        }
        if ask:
            label = teacher(features)
            result.update(self.learn_label(features, label))
            result["executed_answer"] = label
        else:
            result["executed_answer"] = prediction
        return result


@dataclass
class Selector:
    """Hand-designed stochastic learning-progress scheduler with learned EMAs."""

    arm: str
    seed: int

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed + 80000)
        self.q = np.zeros(6)
        self.counts = np.zeros(6, dtype=int)

    def state(self):
        return {
            "q": self.q.tolist(),
            "counts": self.counts.tolist(),
            "rng": self.rng.bit_generator.state,
        }

    def choose(self, block: int) -> tuple[int, list]:
        if block < 6:
            task = ORDER[block]
            return task, np.eye(6)[task].tolist()
        if self.arm == "fixed":
            task = ORDER[(block - 6) // 31]
            return task, np.eye(6)[task].tolist()
        probabilities = np.full(6, 1 / 6)
        if self.arm == "adaptive":
            weights = self.q + 1e-6
            probabilities = 0.9 * weights / weights.sum() + 0.1 / 6
        return int(self.rng.choice(6, p=probabilities)), probabilities.tolist()

    def observe(self, task: int, progress: float) -> None:
        self.counts[task] += 1
        self.q[task] = 0.7 * self.q[task] + 0.3 * progress


def evaluate(student: Student, cases: list) -> dict:
    before = fingerprint(student.state())
    rng_before = torch.get_rng_state().clone()
    predictions = student.probabilities(np.array([c["features"] for c in cases]))
    labels = np.array([c["label"] for c in cases])
    tasks = np.array([c["task"] for c in cases])
    correct = predictions.argmax(-1) == labels
    needs_help = predictions.max(-1) < HELP_THRESHOLD
    result = []
    for task in range(6):
        selected = tasks == task
        independent = selected & ~needs_help
        result.append(
            {
                "task": task,
                "cases": int(selected.sum()),
                "teacher_free_accuracy": float(correct[selected].mean()),
                "would_request_help": float(needs_help[selected].mean()),
                "autonomous_correct_fraction": float(
                    (correct & ~needs_help)[selected].mean()
                ),
                "confident_error_fraction": float(
                    (~correct & ~needs_help)[selected].mean()
                ),
                "selective_accuracy_without_help": (
                    float(correct[independent].mean()) if independent.any() else None
                ),
            }
        )
    assert fingerprint(student.state()) == before
    assert torch.equal(rng_before, torch.get_rng_state())
    return {
        "per_task": result,
        "macro_accuracy": float(correct.mean()),
        "balanced_help_rate": float(needs_help.mean()),
        "balanced_autonomous_correct": float((correct & ~needs_help).mean()),
        "balanced_confident_error": float((~correct & ~needs_help).mean()),
        "state_hash_before": before,
        "state_hash_after": fingerprint(student.state()),
    }


def prepare() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name in LEDGERS:
        assert json.loads((ROOT / name).read_text())["closed"]
    save(OUTPUT / "heldout.json", dataset("heldout", 20261008))
    save(
        OUTPUT / "validation-heldout.json", dataset("validation_test", VALIDATION_SEED)
    )
    plan = {
        "seeds": SEEDS,
        "arms": ARMS,
        "tasks": TASKS,
        "student": "existing 2x64 MLP, five visible numeric/task-ID features, one zero goal feature, 2 answer actions; CE/Adam 0.001; no PPO in this mechanism test",
        "teacher": "free deterministic rule oracle; queried label only when max student probability < 0.90; availability never decreases",
        "teacher_cost_unit": "one requested training label; cached replay free; validation labels never used by scheduler/student",
        "query_gate": "fixed confidence threshold; answer probabilities learn, gate is not a learned ask-action policy",
        "update": "one Adam step per queried label, batch=current label plus 31 draws with replacement from queried-label replay; no training on unqueried labels",
        "blocks": BLOCKS,
        "cases_per_block": BLOCK_SIZE,
        "practice_cases_per_arm_seed": BLOCKS * BLOCK_SIZE,
        "common_warmup": ORDER,
        "warmup_cases": 192,
        "adaptive_selection": "after common warmup: p[t]=0.9*(Q[t]+1e-6)/sum(Q+1e-6)+0.1/6; Q[t]=0.7*Q[t]+0.3*mean(positive log-p(true-label) gain over 32 cases); unqueried cases contribute zero; Q initially zero",
        "selection_learning_claim": "hand-designed scheduler learning progress EMAs, not an end-to-end learned task-selection policy",
        "fixed_selection": "after common warmup, 31 consecutive blocks per task in order 0,2,4,1,3,5 (wide margins before near boundary); each task gets 1024 cases total",
        "random_selection": "uniform per block after same warmup",
        "training_streams": "same per-seed/per-task kth cases across arms, independent of selection; balanced labels by alternating stream index",
        "heldout": "256 balanced cases per task, fixed split/seed20261008; never used for scheduler decisions, replay, confidence calibration or early stopping",
        "environment_checkpoints": ENV_MILESTONES,
        "teacher_cost_checkpoints": COST_MILESTONES,
        "matched_cost_caveat": "equal labels means equal optimizer steps; report environment cases consumed too. Only jointly reached cost milestones compared; never force extra queries.",
        "retention_probe": {
            "cases": RETENTION_CASES,
            "task": 5,
            "selection": "same focused task after curriculum for every arm",
            "teacher_still_available": True,
            "metrics": "before/after accuracy on other tasks, acquired>=0.85; also peak-to-final descriptive drops",
        },
        "teacher_ceiling_per_arm_seed": BLOCKS * BLOCK_SIZE + RETENTION_CASES,
        "no_forced_help_decline": "ceiling equals all available cases, so it never denies a request; no time-dependent threshold; confidence errors and fixed balanced help measured",
        "primary_metrics": [
            "balanced teacher-free competence at matched environment cost",
            "competence at jointly reached label costs",
            "actual query counts by task and early/late practice",
            "balanced fixed-probe help rate and confident errors",
            "retained acquired non-focus skills",
        ],
        "descriptive_decision_rules": {
            "adaptive_advantage": "at least +0.02 macro accuracy versus both baselines in >=2/3 seeds, at final matched practice and highest label milestone reached by all nine runs; descriptive, not significance testing",
            "less_help_with_competence": "balanced held-out request fraction drops >=0.20 from post-warmup192 to final6144, final macro accuracy>=0.85 and confident error fraction<=0.05; also report all six task/difficulty buckets",
            "retention": "tasks other than focus5 with pre-probe accuracy>=0.85 are acquired; report their before/after differences and how many stay>=0.85; replay contributes to retention",
        },
        "learnability_gate": {
            "initialization_seed": VALIDATION_SEED,
            "all_oracle_training_cases": VALIDATION_CASES,
            "balanced_round_robin": True,
            "minimum_macro_accuracy": 0.90,
            "minimum_task_accuracy": 0.80,
            "if_failed": "stop; no main experiment or hyperparameter fishing",
        },
        "stops": "fixed budget; nonfinite update; evaluation mutation; ledger/source mismatch; 600 seconds per phase; RSS1800MiB or disk<8GiB",
        "resources": "CPU only, one thread, nice15; existing environment; no API clients/credentials/browser imports",
        "ledger_hashes": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in LEDGERS
        },
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
    }
    save(OUTPUT / "plan.json", plan)
    names = [
        "scripts/run_adaptive_curriculum.py",
        "rl/policy.py",
        "logs/adaptive-curriculum/plan.json",
        "logs/adaptive-curriculum/heldout.json",
        "logs/adaptive-curriculum/validation-heldout.json",
    ]
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
    for name, expected in json.loads((OUTPUT / "plan.json").read_text())[
        "ledger_hashes"
    ].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected


def verify() -> None:
    verify_inputs()
    destination = OUTPUT / "learnability.json"
    if destination.exists():
        raise FileExistsError("Preserve the declared learnability check")
    heldout = json.loads((OUTPUT / "validation-heldout.json").read_text())
    student = Student(VALIDATION_SEED)
    fresh = evaluate(student, heldout)
    started = time.monotonic()
    for i in range(VALIDATION_CASES):
        example = case(i % 6, i // 6, VALIDATION_SEED, "validation_train")
        student.learn_label(
            np.array(example["features"], dtype=np.float32), example["label"]
        )
        if i % 256 == 0:
            resources()
            if time.monotonic() - started > MAX_SECONDS:
                raise TimeoutError("Learnability budget")
    trained = evaluate(student, heldout)
    passed = (
        trained["macro_accuracy"] >= 0.90
        and min(t["teacher_free_accuracy"] for t in trained["per_task"]) >= 0.80
    )
    result = {
        "passed": passed,
        "fresh": fresh,
        "trained": trained,
        "oracle_cases": VALIDATION_CASES,
        "not_part_of_main_cost_or_independence_claim": True,
        "random_and_constant_expected_accuracy": 0.5,
        "oracle_accuracy": 1.0,
        "resources": resources(),
        "wall_seconds": time.monotonic() - started,
    }
    save(destination, result)
    print(json.dumps(result), flush=True)
    if not passed:
        raise RuntimeError("Task learnability gate failed; stop without retuning")


def run_one(
    arm: str, seed: int, heldout: list, directory: Path, started: float
) -> dict:
    directory.mkdir(exist_ok=False)
    student = Student(seed)
    selector = Selector(arm, seed)
    initial_hash = fingerprint(student.model.state_dict())
    counters = [0] * 6
    requested = 0
    checkpoints = []

    def measure(kind: str, cases_seen: int):
        before = fingerprint(selector.state())
        result = evaluate(student, heldout)
        assert fingerprint(selector.state()) == before
        row = {
            "kind": kind,
            "cases_seen": cases_seen,
            "teacher_labels": student.queries,
            "optimizer_steps": student.updates,
            "result": result,
        }
        save(directory / f"evaluation-{kind}-{cases_seen}-{student.queries}.json", row)
        checkpoints.append(row)
        return result

    measure("environment", 0)
    with (directory / "cases.jsonl").open("x") as case_log, (
        directory / "blocks.jsonl"
    ).open("x") as block_log:
        for block in range(BLOCKS + RETENTION_CASES // BLOCK_SIZE):
            if time.monotonic() - started > MAX_SECONDS:
                raise TimeoutError("Curriculum study time limit")
            retention = block >= BLOCKS
            task, probabilities = (
                (5, np.eye(6)[5].tolist()) if retention else selector.choose(block)
            )
            before = selector.state()
            progress = []
            queries_before = student.queries
            for offset in range(BLOCK_SIZE):
                example = case(task, counters[task], seed)
                counters[task] += 1
                features = np.asarray(example["features"], dtype=np.float32)

                def teacher(x):
                    nonlocal requested
                    requested += 1
                    return oracle(x)

                outcome = student.practise(features, teacher)
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
    checkpoint = ROOT / "checkpoints/adaptive-curriculum" / f"{arm}-{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": student.model.state_dict(),
            "optimizer": student.optimizer.state_dict(),
            "teacher_labels": student.queries,
            "optimizer_steps": student.updates,
            "replay_x": np.stack(student.replay_x),
            "replay_y": student.replay_y,
            "selector": selector.state(),
        },
        checkpoint,
    )
    result = {
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
    if not json.loads((OUTPUT / "learnability.json").read_text())["passed"]:
        raise RuntimeError("Learnability not established")
    directory = OUTPUT / "runs"
    directory.mkdir(exist_ok=False)
    heldout = json.loads((OUTPUT / "heldout.json").read_text())
    started = time.monotonic()
    status = {"complete": False, "new_paid_calls": 0, "teacher": "simulated oracle"}
    try:
        summaries = []
        for seed in SEEDS:
            for arm in ARMS:
                summaries.append(
                    run_one(arm, seed, heldout, directory / f"{arm}-{seed}", started)
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
    parser.add_argument("phase", choices=["prepare", "verify", "run"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(0)
    if args.phase == "prepare":
        prepare()
    elif args.phase == "verify":
        verify()
    else:
        run()
