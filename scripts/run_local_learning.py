#!/usr/bin/env python3
"""Bounded CPU experiment on a synthetic DOM cue task, with no network or API coach.

This is a learner diagnostic, not a MiniWoB benchmark. All controllers see the
same DOM strings and legal actions. The memory variants retain the same four
observations; no previous actions, rewards, coach memory, or hidden cue enter them.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import random
import resource
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn

from agents.hybrid import HybridAgent
from envs.browsergym_client import make_planner_action
from eval.harness import EvalConfig, evaluation_mode
from llm.coach import CoachDirective
from rl.features import DomTextHasher
from rl.rnd_ppo_agent import LearnerConfig, PPORNDLearner, RNDConfig


class OfflineCoach:
    """Constant instruction, empty masks, no plan/reflection or task-state access."""

    def advise(self, **kwargs):
        return CoachDirective(subgoal="Choose the side shown by the cue.")


class CueEnv:
    """Balanced two-choice task with an optional three-step cue-to-choice delay."""

    def __init__(self, delay: int = 0):
        self.delay = delay
        self.next_seed = 0

    def reset(self, **kwargs):
        self.episode_seed = self.next_seed
        self.next_seed += 1
        self.side = self.episode_seed % 2
        self.t = 0
        return self.observation(), {}

    def observation(self):
        if self.t == 0:
            text = "Cue: " + ("left" if self.side == 0 else "right")
        elif self.t < self.delay:
            text = "Wait for the choice screen."
        else:
            text = "Choose the side remembered from the cue."
        if self.t == self.delay:
            text += " Buttons: left right."
        return {"dom_text": text, "step_id": (self.episode_seed, self.t)}

    def encode_observation(self, obs):
        return obs

    def actions(self):
        if self.t < self.delay:
            return [make_planner_action("wait", wait_ms=0)]
        return [
            make_planner_action("click", selector="#left"),
            make_planner_action("click", selector="#right"),
        ]

    def step(self, action):
        done = self.t == self.delay
        success = done and action.selector == ("#left" if self.side == 0 else "#right")
        reward = float(success)
        self.t += 1
        return (
            self.observation(),
            reward,
            done,
            False,
            {"success": success, "episode_reward": reward},
        )


class WindowAgent(HybridAgent):
    def __init__(self, *args, window: int = 1, token_dim: int = 64, **kwargs):
        self.window = window
        self.token_hasher = DomTextHasher(token_dim)
        self.history = []
        self.last_observation_id = None
        super().__init__(*args, **kwargs)

    def _reset_episode_state(self):
        super()._reset_episode_state()
        self.history = []
        self.last_observation_id = None

    def _state_vector(self, observation):
        if observation["step_id"] != self.last_observation_id:
            self.history.append(self.token_hasher.encode(observation))
            self.history = self.history[-self.window :]
            self.last_observation_id = observation["step_id"]
        padded = [np.zeros(self.token_hasher.dim, np.float32)] * (
            self.window - len(self.history)
        ) + self.history
        return np.concatenate(padded)

    def _action_catalog(self, raw_obs):
        actions = self.env.actions()
        return actions, self._inventory_strings(actions)


class TinyTransformer(nn.Module):
    """Two causal attention layers over the stored observation window; dropout=0."""

    def __init__(self, token_dim: int, window: int, outputs: int, subgoal_dim: int):
        super().__init__()
        self.token_dim, self.window = token_dim, window
        self.projection = nn.Linear(token_dim, 32)
        self.position = nn.Parameter(torch.randn(window, 32) * 0.02)
        layer = nn.TransformerEncoderLayer(
            32, 2, dim_feedforward=64, dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        # TransformerEncoder clones the prototype; initialize layers independently.
        for block in self.encoder.layers:
            for parameter in block.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.head = nn.Linear(32 + subgoal_dim, outputs)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(window, window, dtype=torch.bool), diagonal=1),
        )

    def forward(self, state, subgoal, mask=None):
        tokens = (
            self.projection(state.reshape(-1, self.window, self.token_dim))
            + self.position
        )
        hidden = self.encoder(tokens, mask=self.causal_mask)[:, -1]
        logits = self.head(torch.cat([hidden, subgoal], dim=-1))
        if mask is not None:
            logits = (logits + mask.clamp(min=1e-6).log()).masked_fill(
                mask <= 0, -torch.inf
            )
        return logits

    @torch.no_grad()
    def sample(self, state, subgoal, mask=None, deterministic=False):
        values = [
            torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
            for x in (state, subgoal)
        ]
        mask_tensor = torch.from_numpy(mask).unsqueeze(0) if mask is not None else None
        logits = self(*values, mask_tensor)
        return (
            int(logits.argmax(-1).item())
            if deterministic
            else int(torch.distributions.Categorical(logits=logits).sample().item())
        )

    def predict_mask(self, *args):
        return None


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def digest_state(learner: PPORNDLearner) -> str:
    """Hash tensors and scalar/optimizer/rollout state, excluding sampling RNG."""
    h = hashlib.sha256()

    def visit(value):
        if isinstance(value, torch.Tensor):
            h.update(value.detach().cpu().numpy().tobytes())
        elif isinstance(value, np.ndarray):
            h.update(value.tobytes())
        elif isinstance(value, dict):
            for key, item in value.items():
                h.update(str(key).encode())
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        else:
            h.update(repr(value).encode())

    visit(
        [
            learner.policy.state_dict(),
            learner.value_net.state_dict(),
            learner.rnd_module.state_dict(),
            learner.optimizer.state_dict(),
            learner.rnd_optimizer.state_dict(),
            [vars(t) for t in learner.rollout],
            learner.total_steps,
            learner.policy_updates,
            learner.optimizer_steps,
            learner.rnd_updates,
            learner._running_mean,
            learner._running_var,
            learner._count,
        ]
    )
    return h.hexdigest()


def resources() -> dict[str, float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = usage.ru_maxrss / (1024**2)  # macOS reports bytes.
    free = shutil.disk_usage(ROOT).free / (1024**3)
    if rss > 1800 or free < 8:
        raise RuntimeError(
            f"Resource stop: peak RSS={rss:.1f} MiB, free disk={free:.1f} GiB"
        )
    return {
        "peak_rss_mib": rss,
        "free_disk_gib": free,
        "cpu_seconds": usage.ru_utime + usage.ru_stime,
    }


def evaluate(agent: WindowAgent, episodes: int, seed: int = 100000) -> dict:
    learner = agent.learner
    before = digest_state(learner) if learner else None
    saved_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    saved_seed = agent.env.next_seed
    agent.env.next_seed = seed
    seed_all(seed)
    results = []
    try:
        with evaluation_mode(agent, EvalConfig(True, episodes, False)):
            for i in range(episodes):
                outcome = agent.run_episode("local-cue")
                results.append(
                    {
                        "seed": seed + i,
                        "success": outcome["success"],
                        "steps": outcome["steps"],
                        "reward": outcome["reward"],
                        "trace": outcome["trace"],
                    }
                )
    finally:
        random.setstate(saved_rng[0])
        np.random.set_state(saved_rng[1])
        torch.set_rng_state(saved_rng[2])
        agent.env.next_seed = saved_seed
    after = digest_state(learner) if learner else None
    assert before == after, "Evaluation mutated learning state"
    return {
        "success_rate": float(np.mean([r["success"] for r in results])),
        "episodes": results,
        "learning_hash_before": before,
        "learning_hash_after": after,
    }


def build(
    args: argparse.Namespace, seed: int, random_controller: bool = False
) -> WindowAgent:
    seed_all(seed)
    window = 4 if args.arch in {"history", "transformer"} else 1
    default_size = args.preset == "default"
    token_dim = 2048 if default_size else 64
    config = LearnerConfig(
        feature_dim=token_dim * window,
        subgoal_dim=256 if default_size else 8,
        hidden_dim=128 if default_size else 64,
        max_actions=32 if default_size else 2,
        rollout_size=args.rollout,
        minibatch_size=256 if default_size else 64,
        policy_epochs=4,
    )
    rnd_config = RNDConfig(
        embedding_dim=config.feature_dim, intrinsic_weight=args.rnd_weight
    )
    learner = (
        None
        if random_controller
        else PPORNDLearner(learner_config=config, rnd_config=rnd_config)
    )
    if learner and args.arch == "transformer":
        learner.policy = TinyTransformer(64, window, 2, config.subgoal_dim)
        learner.value_net = TinyTransformer(64, window, 1, config.subgoal_dim)
        learner.optimizer = torch.optim.Adam(
            list(learner.policy.parameters()) + list(learner.value_net.parameters()),
            lr=config.learning_rate,
        )
    env = CueEnv(args.delay)
    env.next_seed = seed * 1000000
    agent = WindowAgent(
        env,
        OfflineCoach(),
        learner=learner,
        max_steps=args.delay + 1,
        window=window,
        token_dim=token_dim,
        guardrails_enabled=False,
        reflexion_enabled=False,
        planner_interval=100,
        stuck_entropy_threshold=100,
    )
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch", choices=["current", "history", "transformer"], default="current"
    )
    parser.add_argument("--delay", type=int, choices=[0, 3], default=0)
    parser.add_argument("--preset", choices=["small", "default"], default="small")
    parser.add_argument("--steps", type=int, default=8192)
    parser.add_argument("--rollout", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 43])
    parser.add_argument("--eval-episodes", type=int, default=400)
    parser.add_argument("--rnd-weight", type=float, default=0.0)
    parser.add_argument("--max-seconds", type=int, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.preset == "default" and args.arch != "current":
        parser.error(
            "The default-sized diagnostic is supported for the current-screen MLP only"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    logging.basicConfig(filename=args.output / "learner.log", level=logging.INFO)
    config = vars(args).copy()
    config["output"] = str(args.output)
    config.update(
        {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": sys.version,
            "coach": "constant offline; empty masks; no API",
            "device": "cpu",
            "threads": 1,
        }
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    started = time.monotonic()
    summaries = []
    with (args.output / "training.jsonl").open("w") as logfile:
        for seed in args.seeds:
            agent = build(args, seed)
            learner = agent.learner
            initial = copy.deepcopy(learner.policy.state_dict())
            fresh = evaluate(agent, args.eval_episodes)
            random_result = evaluate(
                build(args, seed, random_controller=True), args.eval_episodes
            )
            (args.output / f"fresh-{seed}.json").write_text(json.dumps(fresh))
            (args.output / f"random-{seed}.json").write_text(json.dumps(random_result))
            episode = 0
            milestones = []
            last_updates = 0
            while learner.total_steps < args.steps:
                if time.monotonic() - started > args.max_seconds:
                    raise TimeoutError("Experiment wall-clock budget exhausted")
                result = agent.run_episode("local-cue")
                episode += 1
                row = {
                    "seed": seed,
                    "episode": episode,
                    "steps": learner.total_steps,
                    "success": result["success"],
                    "intrinsic": result["intrinsic_reward"],
                    "ppo_updates": learner.policy_updates,
                    "rnd_updates": learner.rnd_updates,
                    "learner_failures": agent.learner_failures,
                }
                if learner.policy_updates != last_updates:
                    row.update(
                        {
                            "update": learner.last_update,
                            "resources": resources(),
                            "wall_seconds": time.monotonic() - started,
                        }
                    )
                    last_updates = learner.policy_updates
                    print(json.dumps(row), flush=True)
                logfile.write(json.dumps(row) + "\n")
                if learner.total_steps in {args.steps // 2, args.steps}:
                    frozen = evaluate(agent, args.eval_episodes)
                    (
                        args.output / f"trained-{seed}-{learner.total_steps}.json"
                    ).write_text(json.dumps(frozen))
                    milestones.append(
                        {
                            "steps": learner.total_steps,
                            "success_rate": frozen["success_rate"],
                        }
                    )
                if episode % 128 == 0:
                    logfile.flush()
            trained = evaluate(agent, args.eval_episodes)
            delta = (
                sum(
                    float((value - initial[key]).square().sum())
                    for key, value in learner.policy.state_dict().items()
                    if value.dtype.is_floating_point
                )
                ** 0.5
            )
            summary = {
                "seed": seed,
                "fresh": fresh["success_rate"],
                "random": random_result["success_rate"],
                "trained": trained["success_rate"],
                "steps": learner.total_steps,
                "ppo_updates": learner.policy_updates,
                "optimizer_steps": learner.optimizer_steps,
                "rnd_updates": learner.rnd_updates,
                "policy_parameter_l2_change": delta,
                "learner_failures": agent.learner_failures,
                "milestones": milestones,
                "learner_config": asdict(learner.config),
                "rnd_config": asdict(learner.rnd_config),
                "actor_parameters": sum(p.numel() for p in learner.policy.parameters()),
                "critic_parameters": sum(
                    p.numel() for p in learner.value_net.parameters()
                ),
                "resources": resources(),
                "wall_seconds": time.monotonic() - started,
            }
            checkpoint = ROOT / "checkpoints" / args.output.name
            checkpoint.mkdir(parents=True, exist_ok=True)
            learner.save(str(checkpoint / f"seed-{seed}.pt"))
            summaries.append(summary)
            (args.output / "summary.json").write_text(json.dumps(summaries, indent=2))
            print("RESULT " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
