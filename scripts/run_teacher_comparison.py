#!/usr/bin/env python3
"""Bounded teacher distillation comparison; prepare/collect/train are separate.

Paid collection is retired; training requires separately supplied cached labels.
All available execution phases are offline. This synthetic structured-observation diagnostic exercises
the repository's PPO learner, not its BrowserGym/DOM encoder or live coach loop.
"""

from __future__ import annotations

import argparse
import copy
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
from torch.nn import functional as F

from llm.budgeted_teacher import (
    PRICING,
    BudgetStop,
    SharedBudget,
)
from rl.rnd_ppo_agent import LearnerConfig, PPORNDLearner, RNDConfig
from scripts.run_local_learning import digest_state, resources, seed_all

OUTPUT = ROOT / "logs/teacher-comparison"
SEEDS = [7, 19, 43]
STEPS = 4096
DEMONSTRATIONS = 128
BC_STEPS = 128
GOAL = np.zeros(4, dtype=np.float32)
RULE = (
    "Select one of four items (action index 0, 1, 2, or 3). An item is eligible "
    "only if in_stock=1 AND quality>=minimum_quality. Choose the eligible item "
    "with the smallest price. Prices are unique; at least one item is eligible."
)


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def make_case(seed: int, variant: str = "train") -> dict:
    rng = np.random.default_rng(seed)
    prices = [2, 4, 6, 8, 10] if variant == "new_prices" else [1, 3, 5, 7, 9]
    minimum = 6 if variant == "new_threshold" else 5
    while True:
        chosen_prices = rng.choice(prices, 4, replace=False)
        items = [
            {
                "in_stock": int(rng.integers(0, 2)),
                "quality": int(rng.integers(1, 10)),
                "price": int(price),
            }
            for price in chosen_prices
        ]
        if any(x["in_stock"] and x["quality"] >= minimum for x in items):
            return {"minimum_quality": minimum, "items": items}


def correct_action(case: dict) -> int:
    eligible = [
        i
        for i, item in enumerate(case["items"])
        if item["in_stock"] and item["quality"] >= case["minimum_quality"]
    ]
    return min(eligible, key=lambda i: case["items"][i]["price"])


def features(case: dict) -> np.ndarray:
    """Only visible fields, normalized; no oracle/eligibility/rank feature."""
    values = [case["minimum_quality"] / 10]
    for item in case["items"]:
        values.extend([item["in_stock"], item["quality"] / 10, item["price"] / 10])
    return np.asarray(values, dtype=np.float32)


def canonical(case: dict) -> str:
    return json.dumps(case, sort_keys=True, separators=(",", ":"))


def prepare(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    training = []
    occupied = set()
    seed = 200000
    while len(training) < STEPS:
        case = make_case(seed)
        seed += 1
        if canonical(case) not in occupied:
            training.append(case)
            occupied.add(canonical(case))
    evaluation = {}
    for index, variant in enumerate(["iid", "new_prices", "new_threshold"]):
        cases = []
        seed = 900000 + index * 100000
        while len(cases) < 512:
            case = make_case(seed, variant)
            seed += 1
            if canonical(case) not in occupied:
                cases.append(case)
                occupied.add(canonical(case))
        evaluation[variant] = cases
    save(output / "training-cases.json", training)
    save(output / "evaluation-cases.json", evaluation)
    save(
        output / "plan.json",
        {
            "rule": RULE,
            "seeds": SEEDS,
            "training_transitions_per_arm_per_seed": STEPS,
            "teacher_demonstrations": DEMONSTRATIONS,
            "teacher_batches": 4,
            "demonstrations_are_first_training_cases": True,
            "behavior_cloning_steps": BC_STEPS,
            "behavior_cloning_batch_size": 32,
            "arms": ["no_teacher", "teacher", "shuffled_teacher"],
            "held_out_episodes_per_variant": 512,
            "teacher_removed_for_all_driver_actions": True,
            "feature_encoder": "13 visible numeric fields, normalized; no oracle fields",
            "mask": "all four actions always permitted",
            "reflection": "disabled; same budget client tested for reflection path",
            "teacher_role": "offline direct demonstrations; prose strategy is cached only",
            "training_seed_range_start": 200000,
            "evaluation_seed_range_starts": [900000, 1000000, 1100000],
            "pricing": PRICING,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu; 1 thread; nice 15",
            "time_limit_seconds": 900,
            "resource_limits": "peak RSS 1800 MiB; disk free at least 8 GiB",
        },
    )
    SharedBudget(output / "budget.json").initialize()


def parse_labels(text: str, expected: int = 32) -> dict:
    """Strict data parser; never execute teacher output or repair it via another call."""
    try:
        result = json.loads(text)
        assert set(result) == {"actions", "strategy"}
        assert isinstance(result["actions"], list)
        assert len(result["actions"]) == expected
        assert all(type(x) is int and 0 <= x < 4 for x in result["actions"])
        assert isinstance(result["strategy"], str) and len(result["strategy"]) <= 1000
        return result
    except (ValueError, KeyError, TypeError, AssertionError):
        raise BudgetStop(
            "Teacher labels malformed; no repair request allowed"
        ) from None


def collect(output: Path) -> None:
    raise BudgetStop(
        "Historical paid collection is retired; use separately supplied cached labels"
    )


def build(seed: int) -> PPORNDLearner:
    seed_all(seed)
    return PPORNDLearner(
        learner_config=LearnerConfig(
            feature_dim=13,
            subgoal_dim=4,
            hidden_dim=64,
            max_actions=4,
            rollout_size=256,
            minibatch_size=64,
            policy_epochs=4,
        ),
        rnd_config=RNDConfig(embedding_dim=13, intrinsic_weight=0),
    )


def evaluate(learner: PPORNDLearner, evaluation: dict) -> dict:
    before = digest_state(learner)
    was_training = learner.training
    learner.set_training(False)
    outcome = {}
    try:
        with torch.no_grad():
            for name, cases in evaluation.items():
                inputs = torch.from_numpy(np.stack([features(case) for case in cases]))
                goals = torch.zeros((len(cases), 4))
                logits = learner.policy(inputs, goals)
                probabilities = logits.softmax(-1).numpy()
                labels = np.array([correct_action(case) for case in cases])
                predictions = probabilities.argmax(-1)
                outcome[name] = {
                    "episodes": len(cases),
                    "greedy_accuracy": float(np.mean(predictions == labels)),
                    "expected_sampled_accuracy": float(
                        probabilities[np.arange(len(cases)), labels].mean()
                    ),
                    "predictions": predictions.tolist(),
                    "correct_action_probabilities": probabilities[
                        np.arange(len(cases)), labels
                    ].tolist(),
                }
    finally:
        learner.set_training(was_training)
    after = digest_state(learner)
    if before != after:
        raise RuntimeError("Evaluation mutated learning state")
    return {"learning_hash_before": before, "learning_hash_after": after, **outcome}


def pretrain(learner: PPORNDLearner, inputs: torch.Tensor, labels: list[int]) -> dict:
    optimizer = torch.optim.Adam(learner.policy.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(12345)
    targets = torch.tensor(labels, dtype=torch.long)
    losses = []
    for _ in range(BC_STEPS):
        indices = torch.randperm(len(inputs), generator=generator)[:32]
        logits = learner.policy(inputs[indices], torch.zeros((len(indices), 4)))
        loss = F.cross_entropy(logits, targets[indices])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite imitation loss")
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(learner.policy.parameters(), 1)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite imitation gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
    return {"optimizer_steps": BC_STEPS, "losses": losses}


def train(output: Path) -> None:
    if not (output / "teacher-labels.json").exists():
        raise BudgetStop("No complete teacher cache; comparison cannot proceed")
    results_dir = output / "results"
    results_dir.mkdir(exist_ok=False)
    training = json.loads((output / "training-cases.json").read_text())
    evaluation = json.loads((output / "evaluation-cases.json").read_text())
    teacher = json.loads((output / "teacher-labels.json").read_text())["actions"]
    if len(training) != STEPS or len(teacher) != DEMONSTRATIONS:
        raise RuntimeError("Unexpected dataset length")
    inputs = torch.from_numpy(
        np.stack([features(c) for c in training[:DEMONSTRATIONS]])
    )
    shuffled = np.random.default_rng(20261008).permutation(teacher).tolist()
    save(results_dir / "shuffled-labels.json", shuffled)
    started = time.monotonic()
    summaries = []
    with (results_dir / "training.jsonl").open("x") as log:
        for seed in SEEDS:
            for arm in ["no_teacher", "teacher", "shuffled_teacher"]:
                learner = build(seed)
                initial = copy.deepcopy(learner.policy.state_dict())
                fresh = evaluate(learner, evaluation)
                save(results_dir / f"fresh-{arm}-{seed}.json", fresh)
                imitation = {"optimizer_steps": 0}
                if arm != "no_teacher":
                    imitation = pretrain(
                        learner, inputs, teacher if arm == "teacher" else shuffled
                    )
                save(results_dir / f"imitation-{arm}-{seed}.json", imitation)
                save(
                    results_dir / f"warm-{arm}-{seed}.json",
                    evaluate(learner, evaluation),
                )
                # Match action sampling RNG across arms after optional BC.
                seed_all(seed + 10000)
                updates = 0
                block_rewards = []
                milestones = {}
                for step, case in enumerate(training, 1):
                    if time.monotonic() - started > 900:
                        raise TimeoutError("Training time budget exhausted")
                    state = features(case)
                    sample = learner.sample_action_with_context(state, GOAL, None, 4)
                    reward = float(sample.action == correct_action(case))
                    intrinsic = learner.compute_intrinsic(state)
                    learner.observe_transition(
                        state=state,
                        subgoal=GOAL,
                        sample=sample,
                        reward=reward,
                        intrinsic=intrinsic,
                        done=True,
                        next_state=state,
                        next_subgoal=GOAL,
                    )
                    block_rewards.append(reward)
                    if learner.policy_updates != updates:
                        updates = learner.policy_updates
                        row = {
                            "seed": seed,
                            "arm": arm,
                            "transitions": step,
                            "ppo_updates": updates,
                            "optimizer_steps": learner.optimizer_steps,
                            "rnd_updates": learner.rnd_updates,
                            "success_last_256": float(np.mean(block_rewards)),
                            "update": learner.last_update,
                            "resources": resources(),
                            "wall_seconds": time.monotonic() - started,
                        }
                        log.write(json.dumps(row) + "\n")
                        log.flush()
                        block_rewards.clear()
                    if step in {1024, STEPS}:
                        result = evaluate(learner, evaluation)
                        save(results_dir / f"trained-{arm}-{seed}-{step}.json", result)
                        milestones[step] = {
                            key: {
                                metric: value[metric]
                                for metric in [
                                    "greedy_accuracy",
                                    "expected_sampled_accuracy",
                                ]
                            }
                            for key, value in result.items()
                            if isinstance(value, dict)
                        }
                checkpoint = (
                    ROOT / "checkpoints/teacher-comparison" / f"{arm}-{seed}.pt"
                )
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                learner.save(checkpoint)
                delta = (
                    sum(
                        float((value - initial[key]).square().sum())
                        for key, value in learner.policy.state_dict().items()
                    )
                    ** 0.5
                )
                summary = {
                    "arm": arm,
                    "seed": seed,
                    "ppo_updates": learner.policy_updates,
                    "ppo_optimizer_steps": learner.optimizer_steps,
                    "imitation_steps": imitation["optimizer_steps"],
                    "rnd_updates": learner.rnd_updates,
                    "weight_l2_change": delta,
                    "pending_rollout": len(learner.rollout),
                    "milestones": milestones,
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                }
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
    save(
        results_dir / "summary.json",
        {
            "runs": summaries,
            "resources": resources(),
            "wall_seconds": time.monotonic() - started,
            "uniform_random_expected_accuracy": 0.25,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "collect", "train"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    resources()
    {"prepare": prepare, "collect": collect, "train": train}[args.phase](OUTPUT)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        # In the credential-bearing phase never emit exception bodies or tracebacks.
        safe_reason = str(exc) if isinstance(exc, BudgetStop) else type(exc).__name__
        print(json.dumps({"stopped": safe_reason}), flush=True)
        raise SystemExit(1) from None
