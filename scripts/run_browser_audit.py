#!/usr/bin/env python3
"""Small local MiniWoB scaffold audit with a constant offline coach."""

import argparse
import copy
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.hybrid import HybridAgent
from envs.browsergym_client import BrowserGymEnvWrapper
from eval.harness import EvalConfig, evaluation_mode
from rl.rnd_ppo_agent import LearnerConfig, PPORNDLearner, RNDConfig
from scripts.run_local_learning import (
    ROOT,
    OfflineCoach,
    digest_state,
    resources,
    seed_all,
)


class SeededBrowser(BrowserGymEnvWrapper):
    def encode_observation(self, obs):
        # CDP generates a different frame UUID in otherwise identical seeded DOMs.
        # Canonicalize only opaque metadata, equally for every controller.
        clean = dict(obs)
        clean["dom_object"] = canonical_dom(obs["dom_object"])
        return super().encode_observation(clean)

    def reset(self, **kwargs):
        result = super().reset(seed=self.next_seed, **kwargs)
        self.last_seed = self.next_seed
        self.next_seed += 1
        self.last_info = {}
        self.action_errors = []
        obs = result[0]
        self.initial_hash = hashlib.sha256(
            self.encode_observation(obs)["dom_text"].encode()
        ).hexdigest()
        return result

    def step(self, action):
        result = super().step(action)
        self.last_info = result[-1]
        if result[0].get("last_action_error"):
            self.action_errors.append(result[0]["last_action_error"])
        return result


def canonical_dom(dom: dict) -> dict:
    clean = copy.deepcopy(dom)
    for document in clean.get("documents", []):
        index = document.get("frameId", -1)
        if index >= 0:
            clean["strings"][index] = "[frame]"
        nodes = document.get("nodes", {})
        if "backendNodeId" in nodes:
            nodes["backendNodeId"] = list(range(len(nodes["backendNodeId"])))
    return clean


def own_process_tree_mib() -> float:
    rows = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss="], text=True)
    table = [list(map(int, row.split())) for row in rows.splitlines()]
    included = {os.getpid()}
    for _ in range(8):
        included.update(pid for pid, ppid, _ in table if ppid in included)
    return sum(rss for pid, _, rss in table if pid in included) / 1024


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--train-episodes", type=int, default=64)
    p.add_argument("--max-seconds", type=int, default=1200)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not os.environ.get("MINIWOB_URL", "").startswith("file://"):
        raise ValueError("This experiment requires local file:// MiniWoB assets")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    seed_all(args.seed)
    cfg = LearnerConfig(
        feature_dim=256,
        subgoal_dim=32,
        hidden_dim=64,
        max_actions=32,
        rollout_size=64,
        minibatch_size=32,
        policy_epochs=4,
    )
    learner = PPORNDLearner(
        learner_config=cfg, rnd_config=RNDConfig(intrinsic_weight=0.0)
    )
    env = SeededBrowser("browsergym/miniwob.click-checkboxes", max_episode_steps=8)
    agent = HybridAgent(
        env,
        OfflineCoach(),
        learner=learner,
        max_steps=8,
        reflexion_enabled=False,
        planner_interval=100,
        stuck_entropy_threshold=100,
    )
    # No remote request is routed by this file. The browser uses a new temporary profile.
    config = {
        **vars(args),
        "output": str(args.output),
        "learner": asdict(cfg),
        "rnd": asdict(learner.rnd_config),
        "coach": "constant offline",
        "max_episode_steps": 8,
        "url": os.environ["MINIWOB_URL"],
        "observation": "DOM JSON with opaque CDP frame/backend-node IDs canonicalized equally for all arms",
        "pid": os.getpid(),
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    logging.basicConfig(filename=args.output / "learner.log", level=logging.INFO)
    started = time.monotonic()
    summaries = []
    reference_observations = {}
    with (args.output / "episodes.jsonl").open("w") as log:

        def phase(
            name,
            count,
            start_seed,
            training=False,
            random_controller=False,
            guardrails=True,
        ):
            seed_all(args.seed + (1000 if not training else 0))
            agent.learner = None if random_controller else learner
            agent.guardrails_enabled = guardrails
            env.next_seed = start_seed
            before = digest_state(learner)
            rows = []
            with evaluation_mode(agent, EvalConfig(not training, count, training)):
                for index in range(count):
                    if time.monotonic() - started > args.max_seconds:
                        raise TimeoutError("Browser audit wall-clock budget exhausted")
                    result = agent.run_episode("browsergym/miniwob.click-checkboxes")
                    if not training:
                        expected = reference_observations.setdefault(
                            env.last_seed, env.initial_hash
                        )
                        assert (
                            expected == env.initial_hash
                        ), "Paired seeded initial observations differ"
                    raw = env.last_info.get("task_info", {}).get("RAW_REWARD_GLOBAL", 0)
                    row = {
                        "phase": name,
                        "env_seed": env.last_seed,
                        "observation_hash": env.initial_hash,
                        **result,
                        "raw_reward": raw,
                        "strict_success": raw == 1,
                        "action_errors": env.action_errors,
                        "ppo_updates": learner.policy_updates,
                        "learner_steps": learner.total_steps,
                        "wall_seconds": time.monotonic() - started,
                        "resources": resources(),
                    }
                    if index % 10 == 0:
                        row["process_tree_mib"] = own_process_tree_mib()
                        if row["process_tree_mib"] > 2200:
                            raise RuntimeError(
                                "Browser process tree exceeded 2.2 GiB budget"
                            )
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in [
                                    "phase",
                                    "env_seed",
                                    "success",
                                    "strict_success",
                                    "steps",
                                    "ppo_updates",
                                    "wall_seconds",
                                ]
                            }
                        ),
                        flush=True,
                    )
            after = digest_state(learner)
            if not training:
                assert before == after, "Frozen browser evaluation changed learner"
            summary = {
                "phase": name,
                "episodes": count,
                "success_rate": sum(r["success"] for r in rows) / count,
                "strict_success_rate": sum(r["strict_success"] for r in rows) / count,
                "mean_steps": sum(r["steps"] for r in rows) / count,
                "action_errors": sum(len(r["action_errors"]) for r in rows),
                "ppo_updates": learner.policy_updates,
                "learner_steps": learner.total_steps,
                "learning_hash_before": before,
                "learning_hash_after": after,
            }
            summaries.append(summary)
            (args.output / "summary.json").write_text(json.dumps(summaries, indent=2))

        try:
            phase("fresh_guardrails", args.eval_episodes, 10000)
            phase(
                "random_guardrails", args.eval_episodes, 10000, random_controller=True
            )
            phase(
                "random_no_guardrails",
                args.eval_episodes,
                10000,
                random_controller=True,
                guardrails=False,
            )
            phase("train_guardrails", args.train_episodes, 7000000, training=True)
            phase("trained_guardrails", args.eval_episodes, 10000)
            checkpoint = ROOT / "checkpoints/browser-audit"
            checkpoint.mkdir(parents=True, exist_ok=True)
            learner.save(str(checkpoint / f"seed-{args.seed}.pt"))
        finally:
            env.close()
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
