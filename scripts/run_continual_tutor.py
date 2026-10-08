#!/usr/bin/env python3
"""Bounded learned ASK/SKIP policy, synthetic economy and real tutor labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from llm.continual_budget import (
    PRICING,
    ContinualTeacher,
)
from scripts.run_adaptive_curriculum import fingerprint, resources, save

OUTPUT = ROOT / "logs/continual-tutor"
ARMS = ["learned", "heuristic", "no_tutor"]
SEEDS = [7, 19, 43]
PACKETS = 256
PACKET_SIZE = 8
QUERY_CAP = 12
QUERY_CREDITS = 2.0
CHECKPOINTS = [0, 64, 192, 256]
MAX_SECONDS = 600


def answer(p: float, v: float, horizon: int) -> int:
    value = p
    for _ in range(horizon):
        value += v
    return int(value >= -1) + int(value >= 0) + int(value >= 1)


def case(seed: int, index: int, horizon: int, split: str) -> dict:
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                87091,
                seed,
                index,
                horizon,
                0 if split == "calibration" else 1 if split == "main" else 2,
            ]
        )
    )
    target = index % 4
    for _ in range(10000):
        p, v = map(int, rng.integers(-2000, 2001, size=2))
        final = p / 1000 + v / 1000 * horizon
        key = f"{p}:{v}:{horizon}"
        partition = hashlib.sha256(key.encode()).digest()[0] % 3
        required = {"calibration": 0, "main": 1, "probe": 2}[split]
        if (
            partition == required
            and answer(p / 1000, v / 1000, horizon) == target
            and min(abs(final - boundary) for boundary in [-1, 0, 1]) >= 0.15
        ):
            return {
                "id": f"{split}:{seed}:{horizon}:{index}",
                "p": p / 1000,
                "v": v / 1000,
                "horizon": horizon,
                "label": target,
            }
    raise RuntimeError("Case generation bound reached")


def horizon_at(packet: int) -> int:
    return 1 if packet < 64 or packet >= 192 else 2


def stream(seed: int, split: str) -> list[list[dict]]:
    result, used = [], set()
    for packet in range(PACKETS):
        rows = []
        for offset in range(PACKET_SIZE):
            index = packet * PACKET_SIZE + offset
            # Deterministic collision rejection, without changing class balance.
            for collision in range(1000):
                c = case(seed, index + collision * 100000, horizon_at(packet), split)
                key = (c["p"], c["v"], c["horizon"])
                if key not in used:
                    break
            else:
                raise RuntimeError("Duplicate stream collision limit")
            used.add(key)
            rows.append(c)
        # Balanced class generation must not become a positional answer key.
        order_rng = np.random.default_rng(
            np.random.SeedSequence(
                [991781, seed, packet, 0 if split == "calibration" else 1]
            )
        )
        order_rng.shuffle(rows)
        result.append(rows)
    return result


def positional_baseline(packets: list[list[dict]]) -> float:
    """Best constant class per packet slot, an integrity check only."""
    counts = np.zeros((8, 4), dtype=int)
    for rows in packets:
        for position, row in enumerate(rows):
            counts[position, row["label"]] += 1
    return float(counts.max(axis=1).sum() / (len(packets) * 8))


def features(rows: list[dict]) -> torch.Tensor:
    return torch.tensor(
        [[r["p"] / 2, r["v"] / 2, r["horizon"] - 1.5] for r in rows],
        dtype=torch.float32,
    )


class AnswerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )
        self.policy = nn.Linear(32, 4)
        self.value = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x)
        return self.policy(hidden), self.value(hidden).squeeze(-1)


class AskModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(61, 32), nn.Tanh())
        self.policy = nn.Linear(32, 2)
        self.value = nn.Linear(32, 1)
        nn.init.zeros_(self.policy.weight)
        with torch.no_grad():
            self.policy.bias.copy_(torch.tensor([0.0, math.log(0.05 / 0.95)]))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x)
        return self.policy(hidden), self.value(hidden).squeeze(-1)


def ask_observation(
    x: torch.Tensor,
    probabilities: torch.Tensor,
    credits: float,
    calls: int,
    recent_accuracy: float,
    recent_income: float,
    packet: int,
) -> torch.Tensor:
    return torch.cat(
        [
            x.reshape(-1),
            probabilities.reshape(-1),
            torch.tensor(
                [
                    min(credits, 64) / 64,
                    (QUERY_CAP - calls) / QUERY_CAP,
                    recent_accuracy,
                    recent_income / 8,
                    packet / PACKETS,
                ]
            ),
        ]
    )


def parse_answers(text: str) -> list[int]:
    try:
        result = json.loads(text)
        labels = result["answers"]
        if (
            set(result) - {"answers", "rule"}
            or not isinstance(labels, list)
            or len(labels) != 8
            or any(type(a) is not int or not 0 <= a <= 3 for a in labels)
        ):
            raise ValueError("Invalid tutor labels")
        if "rule" in result and (
            not isinstance(result["rule"], str) or len(result["rule"]) > 240
        ):
            raise ValueError("Invalid tutor rule")
        return labels
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid tutor JSON") from exc


def prompt_for(rows: list[dict]) -> str:
    records = [
        {"position": r["p"], "velocity": r["v"], "steps": r["horizon"]} for r in rows
    ]
    return (
        "You are a tutor for a small online learner. For each case, add velocity to position once per step. Classify final position into 0 if less than -1; 1 if at least -1 but less than 0; 2 if at least 0 but less than 1; 3 if at least 1. Return only JSON with exactly eight integer answers in input order and an optional short rule (at most 240 characters). Do not execute code or use tools. Cases: "
        + json.dumps(records, separators=(",", ":"))
    )


def update_answers(
    model: AnswerModel,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    actions: torch.Tensor,
    old_log: torch.Tensor,
    old_values: torch.Tensor,
    reward: torch.Tensor,
    tutor_labels: list[int] | None,
) -> dict:
    advantages = reward - old_values
    losses = []
    for _ in range(4):
        logits, values = model(x)
        log_probs = logits.log_softmax(-1)
        ratios = (log_probs[torch.arange(8), actions] - old_log).exp()
        actor = -torch.minimum(
            ratios * advantages, ratios.clamp(0.8, 1.2) * advantages
        ).mean()
        critic = F.mse_loss(values, reward)
        entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
        ce = (
            F.cross_entropy(logits, torch.tensor(tutor_labels))
            if tutor_labels is not None
            else torch.zeros(())
        )
        loss = actor + 0.5 * critic - 0.01 * entropy + ce
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite answer update")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    return {
        "optimizer_steps": 4,
        "mean_loss": float(np.mean(losses)),
        "tutor_labels_used": 0 if tutor_labels is None else 8,
    }


def update_ask(
    model: AskModel,
    optimizer: torch.optim.Optimizer,
    rollout: list[dict],
    bootstrap: float,
) -> dict:
    obs = torch.stack([r["obs"] for r in rollout])
    acts = torch.tensor([r["action"] for r in rollout])
    old_logs = torch.tensor([r["log_prob"] for r in rollout])
    old_values = torch.tensor([r["value"] for r in rollout])
    advantage, advantages = 0.0, []
    next_value = bootstrap
    for row in reversed(rollout):
        delta = row["reward"] + 0.98 * next_value - row["value"]
        advantage = delta + 0.98 * 0.95 * advantage
        advantages.append(advantage)
        next_value = row["value"]
    advantages = torch.tensor(list(reversed(advantages)))
    returns = advantages + old_values
    normalized = (advantages - advantages.mean()) / advantages.std(
        unbiased=False
    ).clamp_min(0.01)
    for _ in range(4):
        logits, values = model(obs)
        log_probs = logits.log_softmax(-1)
        ratios = (log_probs[torch.arange(len(rollout)), acts] - old_logs).exp()
        actor = -torch.minimum(
            ratios * normalized, ratios.clamp(0.8, 1.2) * normalized
        ).mean()
        entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
        loss = actor + 0.5 * F.mse_loss(values, returns) - 0.01 * entropy
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite ask update")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    return {"ask_optimizer_steps": 4, "ask_loss": float(loss.detach())}


@torch.no_grad()
def measure(
    model: AnswerModel,
    optimizer: torch.optim.Optimizer,
    ask: AskModel,
    ask_optimizer: torch.optim.Optimizer,
    probes: list[dict],
    packet: int,
    calls: int,
) -> dict:
    state = [
        model.state_dict(),
        optimizer.state_dict(),
        ask.state_dict(),
        ask_optimizer.state_dict(),
    ]
    before = fingerprint(state)
    rng = torch.get_rng_state().clone()
    logits, _ = model(features(probes))
    probabilities = logits.softmax(-1).numpy()
    labels = np.array([c["label"] for c in probes])
    result = []
    for horizon in (1, 2):
        indices = np.array([i for i, c in enumerate(probes) if c["horizon"] == horizon])
        p = probabilities[indices]
        correct = p.argmax(-1) == labels[indices]
        result.append(
            {
                "horizon": horizon,
                "cases": len(indices),
                "greedy_accuracy": float(correct.mean()),
                "correct_label_probability": float(
                    p[np.arange(len(indices)), labels[indices]].mean()
                ),
                "confident_error_fraction": float(
                    (~correct & (p.max(-1) >= 0.9)).mean()
                ),
            }
        )
    after = fingerprint(state)
    assert before == after and torch.equal(rng, torch.get_rng_state())
    return {
        "packet": packet,
        "cases_seen": packet * 8,
        "calls": calls,
        "teacher_labels": calls * 8,
        "per_horizon": result,
        "macro_greedy_accuracy": float(np.mean([r["greedy_accuracy"] for r in result])),
        "state_hash_before": before,
        "state_hash_after": after,
    }


def fit(
    arm: str,
    seed: int,
    packets: list[list[dict]],
    probes: list[dict],
    directory: Path,
    teacher: ContinualTeacher | None,
    *,
    simulated: bool,
    started: float,
) -> dict:
    directory.mkdir(exist_ok=False)
    torch.manual_seed(seed)
    model = AnswerModel()
    answer_initial = fingerprint(model.state_dict())
    torch.manual_seed(seed + 90000)
    ask = AskModel()
    ask_initial = fingerprint(ask.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    ask_optimizer = torch.optim.Adam(ask.parameters(), lr=0.001)
    answer_rng = torch.Generator().manual_seed(seed + 40000)
    ask_rng = torch.Generator().manual_seed(seed + 50000)
    credits, income, calls, recent_accuracy, recent_income = 4.0, 0.0, 0, 0.25, 0.0
    records, evaluations, rollout = [], [], []
    ask_updates, labels_used = 0, 0
    evaluations.append(measure(model, optimizer, ask, ask_optimizer, probes, 0, calls))
    with (directory / "packets.jsonl").open("x") as log:
        for packet, rows in enumerate(packets):
            if time.monotonic() - started > MAX_SECONDS:
                raise TimeoutError("Continual study phase time limit")
            x = features(rows)
            with torch.no_grad():
                logits, old_values = model(x)
                probabilities = logits.softmax(-1)
                actions = torch.multinomial(
                    probabilities, 1, generator=answer_rng
                ).squeeze(-1)
                old_log = logits.log_softmax(-1)[torch.arange(8), actions]
                obs = ask_observation(
                    x,
                    probabilities,
                    credits,
                    calls,
                    recent_accuracy,
                    recent_income,
                    packet,
                )
                ask_logits, ask_value = ask(obs)
                ask_probs = ask_logits.softmax(-1)
                entropy = float(
                    -(probabilities * probabilities.clamp_min(1e-12).log())
                    .sum(-1)
                    .mean()
                )
                raw_ask = (
                    bool(torch.multinomial(ask_probs, 1, generator=ask_rng).item())
                    if arm == "learned"
                    else entropy > 1.0 if arm == "heuristic" else False
                )
            eligible = credits >= QUERY_CREDITS and calls < QUERY_CAP
            granted = raw_ask and eligible
            credits_before = credits
            tutor_labels, attempt, tutor_error = None, None, None
            # All scored answers and the query decision are committed before feedback.
            if granted:
                if simulated:
                    tutor_labels = [r["label"] for r in rows]
                else:
                    if teacher is None:
                        raise RuntimeError("Real tutor missing")
                    text, attempt = teacher.request(prompt_for(rows), f"{arm}-{seed}")
                    tutor_labels = parse_answers(text)
                    save(
                        directory / f"tutor-response-{calls}.json",
                        {
                            "attempt": attempt,
                            "packet": packet,
                            "prompt": prompt_for(rows),
                            "response": text,
                            "labels": tutor_labels,
                        },
                    )
                calls += 1
                credits -= QUERY_CREDITS
                tutor_error = (
                    sum(
                        a != r["label"] for a, r in zip(tutor_labels, rows, strict=True)
                    )
                    / 8
                )
            labels = torch.tensor([r["label"] for r in rows])
            correct = actions == labels
            job_pay = 0.25 if rows[0]["horizon"] == 1 else 1.0
            earned = float(correct.sum()) * job_pay
            income += earned
            credits += earned
            training = update_answers(
                model,
                optimizer,
                x,
                actions,
                old_log,
                old_values,
                correct.float() * job_pay,
                tutor_labels,
            )
            labels_used += training["tutor_labels_used"]
            net_reward = (earned - QUERY_CREDITS * granted) / 8
            row = {
                "packet": packet,
                "horizon": rows[0]["horizon"],
                "examples": rows,
                "committed_actions": actions.tolist(),
                "pre_feedback_greedy": probabilities.argmax(-1).tolist(),
                "correct": correct.tolist(),
                "income": earned,
                "total_income": income,
                "credits_before": credits_before,
                "credits_after": credits,
                "raw_ask": raw_ask,
                "raw_learned_ask_probability": float(ask_probs[1]),
                "eligible": eligible,
                "granted": granted,
                "calls": calls,
                "synthetic_tutor_spend": QUERY_CREDITS * calls,
                "net_reward": net_reward,
                "teacher_error_fraction_analysis_only": tutor_error,
                "attempt": attempt,
                "entropy": entropy,
                "loan_request": (
                    None
                    if not raw_ask or eligible
                    else {
                        "kind": (
                            "synthetic_credit"
                            if credits_before < 2
                            else "real_api_allowance"
                        ),
                        "reason": (
                            "sampled learned ask"
                            if arm == "learned"
                            else "heuristic uncertainty"
                        ),
                        "ask_value_estimate": float(ask_value),
                        "approved": False,
                        "real_ceiling_changed": False,
                    }
                ),
                **training,
            }
            if arm == "learned":
                rollout.append(
                    {
                        "obs": obs.detach(),
                        "action": int(raw_ask),
                        "log_prob": float(ask_logits.log_softmax(-1)[int(raw_ask)]),
                        "value": float(ask_value),
                        "reward": net_reward,
                    }
                )
            recent_accuracy = 0.9 * recent_accuracy + 0.1 * float(
                correct.float().mean()
            )
            recent_income = 0.9 * recent_income + 0.1 * earned
            if arm == "learned" and ((packet + 1) % 16 == 0):
                if packet + 1 == PACKETS:
                    bootstrap = 0.0
                else:
                    with torch.no_grad():
                        next_x = features(packets[packet + 1])
                        next_probs = model(next_x)[0].softmax(-1)
                        next_obs = ask_observation(
                            next_x,
                            next_probs,
                            credits,
                            calls,
                            recent_accuracy,
                            recent_income,
                            packet + 1,
                        )
                        bootstrap = float(ask(next_obs)[1])
                row.update(update_ask(ask, ask_optimizer, rollout, bootstrap))
                ask_updates += 4
                rollout.clear()
            assert credits >= 0 and calls <= QUERY_CAP and labels_used == calls * 8
            row["resources"] = resources()
            records.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if granted and calls in (4, 8, 12):
                evaluation = measure(
                    model, optimizer, ask, ask_optimizer, probes, packet + 1, calls
                )
                evaluation["kind"] = "teacher_cost"
                evaluations.append(evaluation)
            if packet + 1 in CHECKPOINTS:
                evaluation = measure(
                    model, optimizer, ask, ask_optimizer, probes, packet + 1, calls
                )
                evaluation["kind"] = "environment"
                evaluations.append(evaluation)
                print(
                    json.dumps(
                        {
                            "arm": arm,
                            "seed": seed,
                            "packet": packet + 1,
                            "calls": calls,
                            "net_income": income - calls * 2,
                            "accuracy": evaluation["macro_greedy_accuracy"],
                        }
                    ),
                    flush=True,
                )
    checkpoint_dir = ROOT / "checkpoints/continual-tutor"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "answer_model": model.state_dict(),
            "ask_model": ask.state_dict(),
            "answer_optimizer": optimizer.state_dict(),
            "ask_optimizer": ask_optimizer.state_dict(),
            "credits": credits,
            "calls": calls,
        },
        checkpoint_dir / f"{'calibration' if simulated else 'real'}-{arm}-{seed}.pt",
    )
    result = {
        "arm": arm,
        "seed": seed,
        "simulated_teacher": simulated,
        "answer_initial_hash": answer_initial,
        "answer_final_hash": fingerprint(model.state_dict()),
        "ask_initial_hash": ask_initial,
        "ask_final_hash": fingerprint(ask.state_dict()),
        "answer_updates": PACKETS * 4,
        "ask_updates": ask_updates,
        "calls": calls,
        "teacher_labels_used": labels_used,
        "income": income,
        "net_income": income - calls * 2,
        "credits": credits,
        "eligible_packets": sum(r["eligible"] for r in records),
        "eligible_raw_asks": sum(r["eligible"] and r["raw_ask"] for r in records),
        "blocked_requests": sum(r["raw_ask"] and not r["eligible"] for r in records),
        "evaluations": evaluations,
        "resources": resources(),
        "phase_elapsed_seconds": time.monotonic() - started,
    }
    save(directory / "summary.json", result)
    return result


def prepare() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    plan = {
        "purpose": "learned decision to acquire real LLM labels for future online income, compared with heuristic and no-tutor controls; not learned task selection",
        "arms": ARMS,
        "seeds": SEEDS,
        "packets": PACKETS,
        "cases_per_packet": 8,
        "task": "predict one of four zones of position after 1 or 2 velocity steps; observations p/2,v/2,horizon-1.5; balanced zones; no hidden answer field in model input; fresh deterministic streams paired across arms",
        "phases": "64 horizon1 packets,128 horizon2 packets,64 fresh horizon1 packets; fixed offered jobs, no task choice; task incomes0.25/1 synthetic credits per correct committed answer",
        "economy": {
            "initial_credits": 4,
            "tutor_cost": 2,
            "loan_requests": "log only, never self-approved",
            "synthetic_credits_never_expand_real_api_authority": True,
        },
        "models": "Answer:3->32tanh->32tanh actor4/value1; ASK:61->32tanh actor2/value1; answer observations and capacity identical; auxiliary ask model instantiated in controls but only learned arm updates it",
        "query_observation": "current8 raw normalized cases, current8x4 answer probabilities, credits clipped64, remaining call fraction, recent observed accuracy/income, stream fraction; no labels or future board",
        "learned_asking": "ASK/SKIP sampled from learned neural categorical policy; initial ASK probability0.05; PPO four epochs every16 packets, clip0.2 gamma0.98 GAE0.95 entropy0.01 Adam0.001; reward=(earned synthetic credits-2*granted ask)/8; benefits of tutoring affect subsequent rewards",
        "heuristic": "ask when mean four-class entropy>1.0 nats, same synthetic affordability and12-call allowance; engineered control only",
        "answer_updates": "commit sampled answers before query and feedback; receive per-action correctness reward (four classes, wrong feedback does not reveal answer); four PPO epochs each8-case packet, clip0.2 entropy0.01 value0.5 Adam0.003; add unit-weight CE on actual tutor answers in same steps when granted; no replay; exact1024 optimizer steps per arm",
        "real_tutor": "pinned gpt-5.4-mini-2026-03-17, reasoning none; structured eight labels plus optional short rule; only labels drive CE, prose logged not interpreted; no oracle correction/replacement of wrong LLM labels; labels do not retroactively fix scored packet answers",
        "real_limits": PRICING,
        "budget_matching": "identical maximum12 tutor calls and96 tutor labels for learned/heuristic; primary equal2048-case exposure; also exact4/8/12-call checkpoints reached by both, explicitly different exposure; actual spend may differ, never force spending to claim matched actual cost",
        "online_evaluation": "same predict-query-feedback-update loop throughout stream, no separate frozen deployment mode; copied/read-only held-out probes are diagnostics only and cannot train any model or drive asking; score before feedback",
        "probes": "256 balanced fresh cases per horizon; never feed results to policy; partitioned from training by latent tuple hash; report per-horizon greedy accuracy, confidence errors, on-stream income/accuracy, teacher mistakes, per-phase eligible ask probability/rate, denied asks and loan requests",
        "calibration": "one offline perfect-label tutor run per arm seed101, same budgets, no pretraining carried forward; require BOTH tutor arms final balanced probe accuracy>=0.70, no-tutor>=0.30, learned ask weights change and1..12 real-equivalent queries; reject without tuning if fails; does not require learned arm to win",
        "primary_success": "learned net income>=1.05*each control in>=2/3 seeds; learned final horizon1 and2 accuracies>=0.70 in each seed and no final macro loss>5pp versus heuristic; small descriptive pilot, not significance claim",
        "farming_control": "mandatory fixed job stream disallows selecting easy jobs; separately report easy/hard/return income and competence; income growth with hard-skill failure does not satisfy success",
        "help_reduction_claim": "only compare request propensity/rate while both cash and real allowance are available; reductions from exhausted budgets, denied requests or offered-job mix do not establish independence",
        "future_board": "record only as possible future current-job-only versus visible upcoming jobs/rewards/availability comparison, without lesson recommendations; no board/planning system in this run",
        "stops": "fixed budgets, failed offline calibration/tests, hash mismatch, malformed/truncated tutor response, uncertain count/usage/network failure, no retries, CPU1 nice15, RSS1800MiB/disk8GiB,600s per calibration or live phase; no real loans",
        "pricing_sources": [
            "https://developers.openai.com/api/docs/models/gpt-5.4-mini",
            "https://developers.openai.com/api/docs/guides/token-counting",
            "https://developers.openai.com/cookbook/articles/per_run_spending_controller_responses_api",
        ],
    }
    save(OUTPUT / "plan.json", plan)
    for seed, split in [(101, "calibration"), *[(s, "main") for s in SEEDS]]:
        packets = stream(seed, split)
        score = positional_baseline(packets)
        if score >= 0.35:
            raise RuntimeError("Packet positions reveal labels; refuse experiment")
        save(OUTPUT / f"stream-{split}-{seed}.json", packets)
        save(
            OUTPUT / f"ordering-check-{split}-{seed}.json",
            {"position_only_accuracy": score, "passed": True},
        )
    for seed, name in [(99101, "calibration"), (20261008, "main")]:
        save(
            OUTPUT / f"probe-{name}.json",
            [case(seed, i, h, "probe") for h in (1, 2) for i in range(256)],
        )
    names = [
        "scripts/run_continual_tutor.py",
        "llm/continual_budget.py",
        "llm/budgeted_teacher.py",
        "scripts/run_adaptive_curriculum.py",
        *[str(p.relative_to(ROOT)) for p in OUTPUT.glob("*.json")],
    ]
    save(
        OUTPUT / "source-data-hashes.json",
        {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in names
        },
    )


def verify() -> None:
    for name, expected in json.loads(
        (OUTPUT / "source-data-hashes.json").read_text()
    ).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected


def run(simulated: bool) -> None:
    if not simulated:
        raise RuntimeError(
            "Paid execution retired after the ordering-integrity audit; any new study "
            "must use a new approved protocol with explicit fresh accounting"
        )
    verify()
    phase = "calibration" if simulated else "real"
    if not simulated:
        preflight = json.loads((OUTPUT / "calibration-summary.json").read_text())
        checks = json.loads((OUTPUT / "offline-checks.json").read_text())
        if not preflight["passed"] or not checks["passed"]:
            raise RuntimeError("Offline preflight has not passed")
    directory = OUTPUT / phase
    directory.mkdir(exist_ok=False)
    probes = json.loads(
        (OUTPUT / f"probe-{'calibration' if simulated else 'main'}.json").read_text()
    )
    budget, teacher = None, None
    status = {"complete": False, "simulated": simulated, "runs": []}
    started = time.monotonic()
    try:
        for seed in ([101] if simulated else SEEDS):
            packets = json.loads(
                (
                    OUTPUT
                    / f"stream-{'calibration' if simulated else 'main'}-{seed}.json"
                ).read_text()
            )
            for arm in ARMS:
                result = fit(
                    arm,
                    seed,
                    packets,
                    probes,
                    directory / f"{arm}-{seed}",
                    teacher,
                    simulated=simulated,
                    started=started,
                )
                status["runs"].append(result)
                save(OUTPUT / f"{phase}-partial.json", status)
        status["complete"] = True
        if simulated:
            by_arm = {r["arm"]: r for r in status["runs"]}
            status["passed"] = (
                all(
                    by_arm[a]["evaluations"][-1]["macro_greedy_accuracy"] >= 0.70
                    for a in ("learned", "heuristic")
                )
                and by_arm["no_tutor"]["evaluations"][-1]["macro_greedy_accuracy"]
                >= 0.30
                and by_arm["learned"]["ask_initial_hash"]
                != by_arm["learned"]["ask_final_hash"]
                and 1 <= by_arm["learned"]["calls"] <= 12
            )
            if not status["passed"]:
                status["blocker"] = (
                    "Declared offline learnability/mechanism gate failed; no paid execution or tuning permitted"
                )
    except Exception as exc:
        status["stop_reason"] = str(exc)
        raise
    finally:
        if teacher is not None:
            teacher.close()
        if budget is not None:
            budget.close()
        verify()
        status.update(
            {
                "prior_ledgers_unchanged": True,
                "wall_seconds": time.monotonic() - started,
                "resources": resources(),
            }
        )
        save(OUTPUT / f"{phase}-summary.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "calibration", "real"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.phase == "prepare":
        prepare()
    else:
        run(args.phase == "calibration")
