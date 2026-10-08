#!/usr/bin/env python3
"""Live original HybridAgent/Coach interaction on an isolated local browser task.

No cloning, replay training, custom policy loss or teacher-generated features.
The smoke entrypoint uses an offline coach and never reads a credential or budget.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from agents.hybrid import HybridAgent
from envs.browsergym_client import BrowserGymEnvWrapper
from llm.budgeted_teacher import BudgetedTeacher, BudgetStop
from llm.coach import Coach, CoachDirective
from llm.linked_budget import LinkedBudget
from rl.rnd_ppo_agent import LearnerConfig, PPORNDLearner, RNDConfig
from scripts.run_browser_audit import own_process_tree_mib
from scripts.run_local_learning import digest_state, resources, seed_all

OUTPUT = ROOT / "logs/interactive-teacher"
PREVIOUS = ROOT / "logs/teacher-comparison/budget.json"
SEEDS = [7, 19, 43]
HORIZON = 24
RULE = (
    "Click the button named by the current cue. Every action advances one trial. "
    "The cue can change; choose using the visible current cue."
)
TASK_ID = "local-browser/changing-cue"


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


class LocalBrowserCue:
    """24 choice trials in a real page, using the original action translator.

    The cue changes every eight attempts, independent of controller actions.
    Each trial rebuilds its two buttons, so focus does not leak between trials.
    Navigation/wait actions are legal but earn zero; all actions consume a trial.
    """

    navigation_timeout = 5.0

    def __init__(self, page):
        self.page = page
        self.first = "left"
        self.variant = "practice"
        self.t = 0
        self.rows = []
        self.current_observation = {}
        self.deadline = time.monotonic() + 600
        self.resource_samples = []
        self.page.goto((ROOT / "scripts/interactive_cue.html").as_uri())

    def cue(self) -> str:
        first = self.first
        return (
            first
            if (self.t // 8) % 2 == 0
            else ("right" if first == "left" else "left")
        )

    def _render(self) -> dict:
        cue = self.cue()
        self.page.evaluate(
            "([cue, trial, variant]) => window.renderTrial(cue, trial, variant)",
            [cue, self.t + 1, self.variant],
        )
        self.current_observation = {
            "goal": RULE,
            "dom_text": self.page.locator("body").inner_text(),
            "url": "file://local-cue",
            "extra_element_properties": {
                side: {"tag": "button", "selector": "#" + side}
                for side in ["left", "right"]
            },
        }
        return self.current_observation

    def reset(self, **kwargs):
        self.t = 0
        self.rows = []
        return self._render(), {}

    def encode_observation(self, obs: dict) -> dict:
        result = BrowserGymEnvWrapper.encode_observation(self, obs)
        result.pop("timestamp", None)
        return result

    def step(self, action):
        if time.monotonic() > self.deadline:
            raise TimeoutError("Interactive experiment time limit")
        if self.t % 8 == 0:
            sample = {**resources(), "browser_tree_mib": own_process_tree_mib()}
            if sample["browser_tree_mib"] > 2200:
                raise RuntimeError("Interactive browser process tree over memory cap")
            self.resource_samples.append(sample)
        cue = self.cue()
        source = BrowserGymEnvWrapper._planner_action_to_browser_action(self, action)
        # Only repository-generated code for its fixed PlannerAction catalog executes.
        exec(source, {"page": self.page})  # noqa: S102
        choice = self.page.evaluate("window.lastChoice")
        correct = choice == cue
        self.rows.append(
            {
                "trial": self.t,
                "cue": cue,
                "action": asdict(action),
                "actual_choice": choice,
                "correct": correct,
            }
        )
        self.t += 1
        done = self.t >= HORIZON
        observation = self.current_observation if done else self._render()
        return (
            observation,
            float(correct),
            done,
            False,
            {"success": correct, "stuck": False},
        )


class NoTeacher:
    """Same constant user instruction, empty masks; no environment-state knowledge."""

    def advise(self, **kwargs):
        return CoachDirective(subgoal=RULE)

    def reflect(self, *args, **kwargs):
        raise BudgetStop("Reflection is disabled in this experiment")


class RecordingClient:
    """Narrow adapter; original Coach constructs all prompts and parses responses."""

    def __init__(self, teacher: BudgetedTeacher, directory: Path):
        self.teacher = teacher
        self.directory = directory
        self.metadata = {}
        self.calls = 0

    def invoke_text(self, messages: list[dict]) -> str:
        # Advice has a developer prompt. Reflection uses a user prompt. Deny it.
        if len(messages) != 2 or messages[1].get("role") != "developer":
            raise BudgetStop("Only live advice is enabled; reflection/ideas disabled")
        text = self.teacher.invoke_text(messages, purpose="advice")
        save(
            self.directory / f"call-{self.calls:02}.json",
            {"metadata": self.metadata, "messages": messages, "response_text": text},
        )
        self.calls += 1
        return text


class InstrumentedHybrid(HybridAgent):
    """Observe original methods without replacing their decisions or math."""

    def __init__(self, *args, seed: int, arm: str, **kwargs):
        self.run_seed = seed
        self.arm = arm
        self.help_events = []
        self.action_events = []
        self.next_reasons = ["episode_start"]
        super().__init__(*args, **kwargs)
        # Disable only the original unconditional debug dump to a shared old log.
        self._debug_obs_dumped = True

    def _reset_episode_state(self):
        super()._reset_episode_state()
        self.help_events = []
        self.action_events = []
        self.next_reasons = ["episode_start"]

    def _should_request_guidance(self, step, info):
        requested = super()._should_request_guidance(step, info)
        reasons = []
        if step > 0 and step % self.planner_interval == 0:
            reasons.append("periodic")
        if (
            self.entropy_window
            and np.mean(self.entropy_window) > self.stuck_entropy_threshold
        ):
            reasons.append("entropy")
        if info.get("stuck", False):
            reasons.append("environment_stuck")
        if requested != bool(reasons):
            raise RuntimeError("Instrumentation diverged from original help predicate")
        self.next_reasons = reasons
        return requested

    def _request_guidance(self, **kwargs):
        event = {
            "seed": self.run_seed,
            "arm": self.arm,
            "after_actions": self.env.t,
            "reasons": list(self.next_reasons),
            "prior_entropy_window": list(self.entropy_window),
            "dom_summary": kwargs["dom_summary"],
            "inventory": list(kwargs["inventory"]),
        }
        if isinstance(self.coach, Coach) and isinstance(
            self.coach.client, RecordingClient
        ):
            self.coach.client.metadata = copy.deepcopy(event)
        super()._request_guidance(**kwargs)
        event["directive"] = asdict(self.current_directive)
        self.help_events.append(event)

    def _select_action(self, state_vec, subgoal_vec, mask_decision, action_count):
        action, sample = super()._select_action(
            state_vec, subgoal_vec, mask_decision, action_count
        )
        self.action_events.append(
            {
                "trial": self.env.t,
                "action_index": action,
                "allowed_actions": np.flatnonzero(mask_decision.final > 0).tolist(),
                "mask_source": mask_decision.source,
                "entropy": sample.entropy if sample else None,
                "log_prob": sample.log_prob if sample else None,
                "subgoal": self.current_subgoal,
                "ppo_updates_before_action": (
                    self.learner.policy_updates if self.learner else None
                ),
            }
        )
        return action, sample


def build(env, seed: int, coach, arm: str, random_controller: bool = False):
    seed_all(seed)
    learner = (
        None
        if random_controller
        else PPORNDLearner(
            learner_config=LearnerConfig(
                feature_dim=2048,
                subgoal_dim=256,
                hidden_dim=128,
                max_actions=32,
                rollout_size=8,
                minibatch_size=4,
                policy_epochs=4,
            ),
            rnd_config=RNDConfig(embedding_dim=2048, intrinsic_weight=0),
        )
    )
    return InstrumentedHybrid(
        env,
        coach,
        learner=learner,
        seed=seed,
        arm=arm,
        planner_interval=10,
        max_steps=HORIZON,
        stuck_entropy_threshold=2.0,
        stuck_window=5,
        guardrails_enabled=False,
        reflexion_enabled=False,
        memory=None,
        note_store=None,
        idea_store=None,
        ddl_inject=False,
    )


def episode(agent) -> dict:
    result = agent.run_episode(TASK_ID)
    return {
        "correct_choices": sum(r["correct"] for r in agent.env.rows),
        "choice_accuracy": sum(r["correct"] for r in agent.env.rows) / HORIZON,
        "steps": result["steps"],
        "reward": result["reward"],
        "help_events": copy.deepcopy(agent.help_events),
        "actions": copy.deepcopy(agent.action_events),
        "browser_trials": copy.deepcopy(agent.env.rows),
        "original_result": result,
    }


def evaluate(agent) -> dict:
    saved_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    learner = agent.learner
    before = digest_state(learner) if learner else None
    old = (agent.coach, agent.training, agent.env.first, agent.env.variant)
    results = []
    try:
        agent.coach = NoTeacher()
        agent.set_training(False)
        for index, first in enumerate(["left", "right", "left", "right"]):
            seed_all(900000 + index)
            agent.env.first, agent.env.variant = first, f"held-out-{index}"
            results.append(episode(agent))
    finally:
        agent.coach, training, agent.env.first, agent.env.variant = old
        agent.set_training(training)
        random.setstate(saved_rng[0])
        np.random.set_state(saved_rng[1])
        torch.set_rng_state(saved_rng[2])
    after = digest_state(learner) if learner else None
    if before != after:
        raise RuntimeError("Frozen evaluation mutated learner")
    return {
        "choice_accuracy": sum(r["correct_choices"] for r in results) / (4 * HORIZON),
        "learning_hash_before": before,
        "learning_hash_after": after,
        "episodes": results,
    }


def prepare() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "calls").mkdir()
    config = {
        "model": "gpt-5.4-mini-2026-03-17",
        "seeds": SEEDS,
        "task": RULE,
        "task_type": "custom real local Chromium page; not MiniWoB benchmark",
        "training_episode_per_seed_per_arm": 1,
        "transitions_per_run": HORIZON,
        "training_cues": ["left"] * 8 + ["right"] * 8 + ["left"] * 8,
        "evaluation": "four new page labels; alternating left/right initial cues; 96 decisions",
        "original_action_catalog": "two buttons plus wait, scroll down/up, Tab, Enter (7 actions)",
        "original_triggers": "start, step>0 and step%10==0, mean entropy>2, info.stuck",
        "anticipated_calls": 9,
        "maximum_new_calls": 10,
        "rollout_size": 8,
        "minibatch_size": 4,
        "ppo_epochs": 4,
        "features": "original 2048-dimensional DOM hasher; 256-dimensional subgoal hasher",
        "architecture": "original two-layer 128-unit MLP policy and critic",
        "learning_rate": 0.0003,
        "rnd_weight": 0,
        "disabled": [
            "checkbox guardrails",
            "untrained policy-mask head",
            "reflection",
            "notes",
            "DDL ideas",
        ],
        "unchanged": [
            "HybridAgent.run_episode",
            "original Coach prompts and parser",
            "subgoal encoding",
            "coach mask matching and combination",
            "PPO loss",
            "action translator",
        ],
        "limits": "1 CPU thread, nice 15, 600 seconds, parent RSS 1800 MiB, process tree 2200 MiB, free disk 8 GiB",
    }
    save(OUTPUT / "plan.json", config)
    files = [
        "agents/hybrid.py",
        "llm/coach.py",
        "llm/prompts.py",
        "llm/budgeted_teacher.py",
        "llm/linked_budget.py",
        "scripts/run_interactive_teacher.py",
        "scripts/interactive_cue.html",
        "rl/policy.py",
        "rl/rnd_ppo_agent.py",
        "rl/features.py",
        "rl/context.py",
    ]
    save(
        OUTPUT / "preflight-source-hashes.json",
        {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files},
    )


def weight_delta(before, module) -> float:
    return (
        sum(
            float((value - before[k]).square().sum())
            for k, value in module.state_dict().items()
        )
        ** 0.5
    )


def run(smoke: bool = False) -> None:
    if not smoke:
        raise BudgetStop(
            "Historical paid browser run is retired; use the offline smoke or recovery runner"
        )
    from playwright.sync_api import sync_playwright

    budget = None if smoke else LinkedBudget(OUTPUT / "budget.json", PREVIOUS)
    if budget and (budget.snapshot()["attempts"] or budget.snapshot()["closed"]):
        raise BudgetStop("Interactive run already attempted; no automatic resumption")
    destination = OUTPUT / ("smoke" if smoke else "results")
    destination.mkdir(exist_ok=False)
    started = time.monotonic()
    teacher = None
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 600, "height": 400})
        page = context.new_page()
        env = LocalBrowserCue(page)
        try:
            recording = None  # Live execution is retired before browser creation.
            summaries = []
            for seed in ([7] if smoke else SEEDS):
                fresh = build(env, seed, NoTeacher(), "fresh")
                fresh_eval = evaluate(fresh)
                save(destination / f"fresh-{seed}.json", fresh_eval)
                random_eval = evaluate(build(env, seed, NoTeacher(), "random", True))
                save(destination / f"random-{seed}.json", random_eval)
                for arm in (
                    ["no_teacher"] if smoke else ["no_teacher", "live_teacher"]
                ):
                    if time.monotonic() - started > 600:
                        raise TimeoutError("Interactive experiment time limit")
                    coach = Coach(recording) if arm == "live_teacher" else NoTeacher()
                    agent = build(env, seed, coach, arm)
                    env.first, env.variant = "left", "practice"
                    learner = agent.learner
                    policy_before = copy.deepcopy(learner.policy.state_dict())
                    value_before = copy.deepcopy(learner.value_net.state_dict())
                    training = episode(agent)
                    save(destination / f"training-{arm}-{seed}.json", training)
                    frozen = evaluate(agent)
                    save(destination / f"trained-{arm}-{seed}.json", frozen)
                    tree_rss = own_process_tree_mib()
                    if tree_rss > 2200:
                        raise RuntimeError(
                            "Interactive browser process tree over memory cap"
                        )
                    row = {
                        "seed": seed,
                        "arm": arm,
                        "training_choice_accuracy": training["choice_accuracy"],
                        "held_out_teacher_free_accuracy": frozen["choice_accuracy"],
                        "fresh_teacher_free_accuracy": fresh_eval["choice_accuracy"],
                        "random_teacher_free_accuracy": random_eval["choice_accuracy"],
                        "help_requests": len(training["help_events"]),
                        "singleton_mask_steps": sum(
                            len(e["allowed_actions"]) == 1 for e in training["actions"]
                        ),
                        "policy_updates": learner.policy_updates,
                        "ppo_optimizer_steps": learner.optimizer_steps,
                        "rnd_updates": learner.rnd_updates,
                        "policy_weight_l2_change": weight_delta(
                            policy_before, learner.policy
                        ),
                        "critic_weight_l2_change": weight_delta(
                            value_before, learner.value_net
                        ),
                        "last_ppo_update": learner.last_update,
                        "pending_rollout": len(learner.rollout),
                        "resources": resources(),
                        "browser_tree_mib": tree_rss,
                        "wall_seconds": time.monotonic() - started,
                        "resource_samples": copy.deepcopy(env.resource_samples),
                    }
                    env.resource_samples.clear()
                    checkpoint = (
                        ROOT / "checkpoints/interactive-teacher" / f"{arm}-{seed}.pt"
                    )
                    if not smoke:
                        checkpoint.parent.mkdir(parents=True, exist_ok=True)
                        learner.save(str(checkpoint))
                    summaries.append(row)
                    save(destination / "summary.json", summaries)
                    print(json.dumps(row), flush=True)
        finally:
            if teacher:
                teacher.close()
            if budget:
                budget.close()
            context.close()
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "smoke", "run"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    logging.basicConfig(level=logging.ERROR)
    if args.phase == "prepare":
        prepare()
    else:
        run(args.phase == "smoke")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        # Never expose SDK or credential-bearing error bodies/tracebacks.
        reason = str(exc) if isinstance(exc, BudgetStop) else type(exc).__name__
        print(json.dumps({"stopped": reason}), flush=True)
        raise SystemExit(1) from None
