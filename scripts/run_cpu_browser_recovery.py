#!/usr/bin/env python3
"""Separate fixed-budget no-teacher recovery on the repaired local browser task."""

from __future__ import annotations

import argparse
import copy
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
from playwright.sync_api import sync_playwright

from envs.browsergym_client import make_planner_action
from rl.features import DomTextHasher
from scripts.run_interactive_teacher import (
    HORIZON,
    SEEDS,
    LocalBrowserCue,
    NoTeacher,
    build,
    episode,
    evaluate,
    save,
    weight_delta,
)
from scripts.run_local_learning import resources

OUTPUT = ROOT / "logs/cpu-browser-recovery"
EPISODES = 64
EVALUATE_AT = [16, 32, 64]
MAX_SECONDS = 600
LEDGERS: list[str] = []  # Offline runs have no provider-account dependency.
SOURCE_FILES = [
    "scripts/run_cpu_browser_recovery.py",
    "scripts/run_interactive_teacher.py",
    "scripts/interactive_cue.html",
    "scripts/run_local_learning.py",
    "agents/hybrid.py",
    "envs/browsergym_client.py",
    "rl/rnd_ppo_agent.py",
    "rl/policy.py",
    "rl/features.py",
    "rl/context.py",
]


def file_hash(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def prepare() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name in LEDGERS:
        assert json.loads((ROOT / name).read_text())["closed"]
    save(
        OUTPUT / "plan.json",
        {
            "seeds": SEEDS,
            "training_episodes_per_seed": EPISODES,
            "transitions_per_episode": HORIZON,
            "transitions_per_seed": EPISODES * HORIZON,
            "total_training_transitions": EPISODES * HORIZON * len(SEEDS),
            "training_cue_start": "alternate left/right each episode; balanced overall; same practice page label",
            "evaluation_at_transitions": [0] + [n * HORIZON for n in EVALUATE_AT],
            "evaluation": "unchanged four held-out labels/left-right starts, 96 stochastic decisions; RNG seeds 900000..900003",
            "controls": "fresh same-seed controller and uniform random; same complete environment observations and constant subgoal",
            "architecture": "unchanged 2-layer 128 MLP; 2048 hashed DOM + 256 hashed constant instruction",
            "ppo": {
                "rollout_size": 8,
                "minibatch_size": 4,
                "epochs": 4,
                "learning_rate": 0.0003,
            },
            "expected_per_seed": {
                "ppo_updates": 192,
                "ppo_optimizer_steps": 1536,
                "rnd_updates": 1536,
            },
            "rnd_reward_weight": 0,
            "teacher": "constant offline NoTeacher stub; no key reads, API client, advice labels, masks or demonstrations",
            "baseline": "new fresh training, not continuation or replacement of corrupted pilot checkpoints",
            "stops": "fixed 1536 transitions/seed; any missed click/nonfinite loss/learner failure; indistinguishable cue features; 600 seconds total; resource limits",
            "resources": {
                "torch_threads": 1,
                "nice": 15,
                "parent_mib": 1800,
                "process_tree_mib": 2200,
                "minimum_disk_gib": 8,
            },
            "ledger_hashes": {name: file_hash(name) for name in LEDGERS},
            "source_hashes": {name: file_hash(name) for name in SOURCE_FILES},
        },
    )


class AuditedCue(LocalBrowserCue):
    """Same task; fail on any lost click and record the pre-action observation."""

    def step(self, action):
        encoded = self.encode_observation(self.current_observation)
        before = hashlib.sha256(
            json.dumps(encoded, sort_keys=True).encode()
        ).hexdigest()
        result = super().step(action)
        row = self.rows[-1]
        row["observation_sha256"] = before
        row["click_delivered"] = action.name != "click" or row[
            "actual_choice"
        ] == action.selector.lstrip("#")
        if not row["click_delivered"]:
            raise RuntimeError("Click delivery failed: stop recovery")
        return result


def preflight(env) -> dict:
    # Compare actual browser/encoder outputs while changing only the visible cue.
    vectors, observations = {}, {}
    hasher = DomTextHasher(2048)
    for first in ["left", "right"]:
        env.first, env.variant = first, "practice"
        raw, _ = env.reset()
        observations[first] = env.encode_observation(raw)
        vectors[first] = hasher.encode(observations[first])
    contrast = float(np.linalg.norm(vectors["left"] - vectors["right"]))
    if not np.isfinite(contrast) or contrast == 0:
        raise RuntimeError("Visible cue is lost by original representation")
    oracle = []
    # This is an execution check only; its choices never become training labels.
    for first in ["left", "right"]:
        env.first = first
        env.reset()
        for _ in range(HORIZON):
            env.step(make_planner_action("click", selector="#" + env.cue()))
        oracle.extend(copy.deepcopy(env.rows))
    assert all(r["correct"] and r["click_delivered"] for r in oracle)
    return {
        "cue_feature_l2_distance": contrast,
        "observations": observations,
        "execution_oracle_correct": len(oracle),
        "execution_oracle_decisions": len(oracle),
        "oracle_used_for_training": False,
    }


def evaluation_signature(result) -> list:
    return [
        row["observation_sha256"]
        for ep in result["episodes"]
        for row in ep["browser_trials"]
    ]


def inspect_actions(result) -> None:
    assert result["original_result"]["learner_failures"] == 0
    assert all(r["click_delivered"] for r in result["browser_trials"])
    assert all(len(r["allowed_actions"]) == 7 for r in result["actions"])


def run() -> None:
    plan = json.loads((OUTPUT / "plan.json").read_text())
    assert all(file_hash(n) == h for n, h in plan["source_hashes"].items())
    assert all(file_hash(n) == h for n, h in plan["ledger_hashes"].items())
    destination = OUTPUT / "results"
    destination.mkdir(exist_ok=False)
    checkpoint_dir = ROOT / "checkpoints/cpu-browser-recovery"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    summary = []
    status = {"completed": False, "additional_api_calls": 0, "credential_reads": 0}
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 600, "height": 400})
                env = AuditedCue(page)
                env.deadline = started + MAX_SECONDS
                save(OUTPUT / "preflight.json", preflight(env))
                for seed in SEEDS:
                    agent = build(env, seed, NoTeacher(), "recovery")
                    before_policy = copy.deepcopy(agent.learner.policy.state_dict())
                    before_value = copy.deepcopy(agent.learner.value_net.state_dict())
                    fresh = evaluate(agent)
                    signature = evaluation_signature(fresh)
                    for ep in fresh["episodes"]:
                        inspect_actions(ep)
                    save(destination / f"fresh-{seed}.json", fresh)
                    random = evaluate(build(env, seed, NoTeacher(), "random", True))
                    assert evaluation_signature(random) == signature
                    save(destination / f"random-{seed}.json", random)
                    # Building the random control reset global RNG; restore the
                    # declared training seed by rebuilding the identical learner.
                    agent = build(env, seed, NoTeacher(), "recovery")
                    assert weight_delta(before_policy, agent.learner.policy) == 0
                    learning_curve = []
                    with (destination / f"training-{seed}.jsonl").open("x") as stream:
                        for number in range(1, EPISODES + 1):
                            if time.monotonic() - started > MAX_SECONDS:
                                raise TimeoutError("Recovery wall-clock budget")
                            env.first = "left" if number % 2 else "right"
                            env.variant = "practice"
                            result = episode(agent)
                            inspect_actions(result)
                            row = {
                                "seed": seed,
                                "episode": number,
                                "transitions": number * HORIZON,
                                "result": result,
                                "policy_updates": agent.learner.policy_updates,
                                "ppo_optimizer_steps": agent.learner.optimizer_steps,
                                "rnd_updates": agent.learner.rnd_updates,
                                "last_ppo_update": agent.learner.last_update,
                                "resource_samples": copy.deepcopy(env.resource_samples),
                                "wall_seconds": time.monotonic() - started,
                            }
                            env.resource_samples.clear()
                            stream.write(json.dumps(row) + "\n")
                            stream.flush()
                            if number % 8 == 0:
                                print(
                                    json.dumps(
                                        {
                                            k: row[k]
                                            for k in [
                                                "seed",
                                                "episode",
                                                "transitions",
                                                "policy_updates",
                                                "wall_seconds",
                                            ]
                                        }
                                    ),
                                    flush=True,
                                )
                            if number in EVALUATE_AT:
                                frozen = evaluate(agent)
                                assert evaluation_signature(frozen) == signature
                                for ep in frozen["episodes"]:
                                    inspect_actions(ep)
                                save(
                                    destination
                                    / f"trained-{seed}-{number * HORIZON}.json",
                                    frozen,
                                )
                                learning_curve.append(
                                    {
                                        "transitions": number * HORIZON,
                                        "held_out_accuracy": frozen["choice_accuracy"],
                                    }
                                )
                                print(
                                    json.dumps(
                                        {"seed": seed, "evaluation": learning_curve[-1]}
                                    ),
                                    flush=True,
                                )
                    learner = agent.learner
                    assert learner.policy_updates == 192
                    assert learner.optimizer_steps == learner.rnd_updates == 1536
                    assert len(learner.rollout) == 0
                    final = {
                        "seed": seed,
                        "fresh_accuracy": fresh["choice_accuracy"],
                        "random_accuracy": random["choice_accuracy"],
                        "learning_curve": learning_curve,
                        "policy_weight_l2_change": weight_delta(
                            before_policy, learner.policy
                        ),
                        "critic_weight_l2_change": weight_delta(
                            before_value, learner.value_net
                        ),
                        "policy_updates": learner.policy_updates,
                        "ppo_optimizer_steps": learner.optimizer_steps,
                        "rnd_updates": learner.rnd_updates,
                        "pending_rollout": len(learner.rollout),
                        "resources": resources(),
                        "wall_seconds": time.monotonic() - started,
                    }
                    learner.save(str(checkpoint_dir / f"seed-{seed}.pt"))
                    summary.append(final)
                    save(destination / "summary.json", summary)
                save(OUTPUT / "last-resource-samples.json", env.resource_samples)
                status["completed"] = True
            finally:
                browser.close()
    except Exception as exc:
        status["stop_reason"] = str(exc)
        raise
    finally:
        status["wall_seconds"] = time.monotonic() - started
        status["ledger_hashes_unchanged"] = {
            n: file_hash(n) == h for n, h in plan["ledger_hashes"].items()
        }
        status["source_hashes_unchanged"] = {
            n: file_hash(n) == h for n, h in plan["source_hashes"].items()
        }
        save(OUTPUT / "status.json", status)
        assert all(status["ledger_hashes_unchanged"].values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "run"])
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    else:
        run()
