#!/usr/bin/env python3
"""Bounded, network-disabled offline diagnosis; never constructs a tutor client."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from collections import Counter
from pathlib import Path

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from scripts import run_continual_tutor as base

OUTPUT = ROOT / "logs/offline-tutor-repair"
SEEDS = [73, 109, 211]
VARIANTS = ["corrected_blocked", "interleaved", "recent_feedback", "balanced_feedback"]


def no_network(*args, **kwargs):
    raise RuntimeError("Network prohibited in offline tutor diagnosis")


def public_rows(rows: list[dict]) -> list[dict]:
    """The sole case boundary for acting: three causal inputs, no metadata."""
    return [{key: row[key] for key in ("p", "v", "horizon")} for row in rows]


def make_stream(seed: int) -> list[list[dict]]:
    targets = np.random.default_rng(np.random.SeedSequence([625403, seed]))
    ordering = np.random.default_rng(np.random.SeedSequence([729871, seed]))
    result, used = [], set()
    for packet in range(256):
        rows = []
        for offset in range(8):
            target = int(targets.integers(4))
            index = (packet * 8 + offset) * 4 + target
            for collision in range(1000):
                row = base.case(
                    seed, index + collision * 400000, base.horizon_at(packet), "main"
                )
                key = (row["p"], row["v"], row["horizon"])
                if key not in used:
                    break
            else:
                raise RuntimeError("Unique case bound")
            used.add(key)
            rows.append(row)
        ordering.shuffle(rows)
        result.append(rows)
    return result


def commit(
    model,
    ask,
    rows,
    credits,
    calls,
    recent_accuracy,
    recent_income,
    packet,
    arm,
    answer_rng,
    ask_rng,
):
    """Commit all actions before any oracle or outcome is available."""
    x = base.features(public_rows(rows))
    with torch.no_grad():
        logits, values = model(x)
        probs = logits.softmax(-1)
        actions = torch.multinomial(probs, 1, generator=answer_rng).squeeze(-1)
        obs = base.ask_observation(
            x, probs, credits, calls, recent_accuracy, recent_income, packet
        )
        ask_logits, ask_value = ask(obs)
        ask_probs = ask_logits.softmax(-1)
        entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum(-1).mean())
        raw_ask = (
            bool(torch.multinomial(ask_probs, 1, generator=ask_rng).item())
            if arm == "learned"
            else entropy > 1.0 if arm == "heuristic" else False
        )
    return {
        "x": x,
        "actions": actions,
        "probabilities": probs,
        "old_log": logits.log_softmax(-1)[torch.arange(8), actions],
        "old_values": values,
        "obs": obs,
        "ask_logits": ask_logits,
        "ask_value": float(ask_value),
        "ask_probability": float(ask_probs[1]),
        "raw_ask": raw_ask,
        "entropy": entropy,
    }


class FeedbackMemory:
    """Retain only experienced inputs, selected actions and correctness bits."""

    def __init__(self, mode: str, seed: int):
        self.mode = mode
        self.rng = np.random.default_rng(seed + 810000)
        self.rows: dict[int, list] = {}
        self.seen: Counter = Counter()

    def add(
        self, x: torch.Tensor, actions: torch.Tensor, correct: torch.Tensor
    ) -> None:
        for inputs, action, result in zip(x, actions, correct, strict=True):
            row = (inputs.detach().clone(), int(action), bool(result))
            key = int(inputs[2] > 0) if self.mode == "balanced_feedback" else 0
            rows = self.rows.setdefault(key, [])
            self.seen[key] += 1
            if self.mode == "recent_feedback":
                rows.append(row)
                if len(rows) > 256:
                    rows.pop(0)
            elif len(rows) < 128:
                rows.append(row)
            else:
                index = int(self.rng.integers(self.seen[key]))
                if index < 128:
                    rows[index] = row

    def sample(self):
        if not self.rows:
            return None
        selected = []
        for key in sorted(self.rows):
            rows = self.rows[key]
            for index in self.rng.integers(len(rows), size=32 // len(self.rows)):
                selected.append(rows[index])
        assert len(selected) == 32
        return (
            torch.stack([r[0] for r in selected]),
            torch.tensor([r[1] for r in selected]),
            torch.tensor([r[2] for r in selected], dtype=torch.float32),
        )


def update_with_feedback(model, optimizer, decision, reward, labels, replay):
    if replay is None:
        return base.update_answers(
            model,
            optimizer,
            decision["x"],
            decision["actions"],
            decision["old_log"],
            decision["old_values"],
            reward,
            labels,
        )
    advantages = reward - decision["old_values"]
    losses = []
    replay_x, replay_actions, replay_correct = replay
    for _ in range(4):
        logits, values = model(decision["x"])
        log_probs = logits.log_softmax(-1)
        ratios = (
            log_probs[torch.arange(8), decision["actions"]] - decision["old_log"]
        ).exp()
        actor = -torch.minimum(
            ratios * advantages, ratios.clamp(0.8, 1.2) * advantages
        ).mean()
        entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
        ce = (
            F.cross_entropy(logits, torch.tensor(labels))
            if labels is not None
            else torch.zeros(())
        )
        remembered_logits, _ = model(replay_x)
        chosen_probability = remembered_logits.softmax(-1)[
            torch.arange(32), replay_actions
        ]
        feedback_loss = F.binary_cross_entropy(
            chosen_probability.clamp(1e-6, 1 - 1e-6), replay_correct
        )
        loss = (
            actor
            + 0.5 * F.mse_loss(values, reward)
            - 0.01 * entropy
            + ce
            + feedback_loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite feedback update")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    return {
        "optimizer_steps": 4,
        "mean_loss": float(np.mean(losses)),
        "tutor_labels_used": 0 if labels is None else 8,
        "replay_rows_per_step": 32,
    }


def fit(variant, arm, seed, packets, probes, started):
    directory = OUTPUT / "runs" / f"{variant}-{arm}-{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    model = base.AnswerModel()
    torch.manual_seed(seed + 90000)
    ask = base.AskModel()
    initial = base.fingerprint(model.state_dict())
    ask_initial = base.fingerprint(ask.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    ask_optimizer = torch.optim.Adam(ask.parameters(), lr=0.001)
    answer_rng = torch.Generator().manual_seed(seed + 40000)
    ask_rng = torch.Generator().manual_seed(seed + 50000)
    memory = (
        FeedbackMemory(variant, seed)
        if variant in {"recent_feedback", "balanced_feedback"}
        else None
    )
    credits, calls, income, recent_accuracy, recent_income = 4.0, 0, 0.0, 0.25, 0.0
    records, rollout, ask_updates = [], [], 0
    evaluations = [base.measure(model, optimizer, ask, ask_optimizer, probes, 0, 0)]
    with (directory / "packets.jsonl").open("x") as log:
        for packet, rows in enumerate(packets):
            telemetry = base.resources()
            if (
                time.monotonic() - started > 600
                or telemetry["peak_rss_mib"] > 1800
                or telemetry["free_disk_gib"] < 8
            ):
                raise RuntimeError("Offline resource stop")
            decision = commit(
                model,
                ask,
                public_rows(rows),
                credits,
                calls,
                recent_accuracy,
                recent_income,
                packet,
                arm,
                answer_rng,
                ask_rng,
            )
            eligible = credits >= 2 and calls < 12
            granted = decision["raw_ask"] and eligible
            before = credits
            # No outcome or label has been accessed by the decision path above.
            labels = (
                [base.answer(r["p"], r["v"], r["horizon"]) for r in public_rows(rows)]
                if granted
                else None
            )
            target = torch.tensor([r["label"] for r in rows])
            correct = decision["actions"] == target
            pay = 0.25 if rows[0]["horizon"] == 1 else 1.0
            earned = float(correct.sum()) * pay
            calls += int(granted)
            income += earned
            credits += earned - 2 * granted
            if granted:
                assert labels == target.tolist()
            # Sample only earlier feedback. Current records are added after update.
            replay = memory.sample() if memory else None
            update = update_with_feedback(
                model, optimizer, decision, correct.float() * pay, labels, replay
            )
            if memory:
                memory.add(decision["x"], decision["actions"], correct)
            reward = (earned - 2 * granted) / 8
            if arm == "learned":
                rollout.append(
                    {
                        "obs": decision["obs"].detach(),
                        "action": int(decision["raw_ask"]),
                        "log_prob": float(
                            decision["ask_logits"].log_softmax(-1)[
                                int(decision["raw_ask"])
                            ]
                        ),
                        "value": decision["ask_value"],
                        "reward": reward,
                    }
                )
            recent_accuracy = 0.9 * recent_accuracy + 0.1 * float(
                correct.float().mean()
            )
            recent_income = 0.9 * recent_income + 0.1 * earned
            if arm == "learned" and (packet + 1) % 16 == 0:
                bootstrap = 0.0
                if packet + 1 < 256:
                    with torch.no_grad():
                        next_x = base.features(public_rows(packets[packet + 1]))
                        obs = base.ask_observation(
                            next_x,
                            model(next_x)[0].softmax(-1),
                            credits,
                            calls,
                            recent_accuracy,
                            recent_income,
                            packet + 1,
                        )
                        bootstrap = float(ask(obs)[1])
                base.update_ask(ask, ask_optimizer, rollout, bootstrap)
                ask_updates += 4
                rollout.clear()
            record = {
                "packet": packet,
                "horizon": rows[0]["horizon"],
                "committed_actions": decision["actions"].tolist(),
                "correct": correct.tolist(),
                "income": earned,
                "credits_before": before,
                "credits_after": credits,
                "calls": calls,
                "raw_ask": decision["raw_ask"],
                "ask_probability": decision["ask_probability"],
                "eligible": eligible,
                "granted": granted,
                "teacher": "offline_perfect_oracle" if granted else None,
                "teacher_labels": labels,
                "real_cost_usd": 0,
                "net_reward": reward,
                "replay_rows": 0 if replay is None else 32,
                **update,
            }
            assert credits >= 0 and calls <= 12
            records.append(record)
            log.write(json.dumps(record) + "\n")
            if packet + 1 in (64, 192, 256) or granted and calls in (4, 8, 12):
                evaluation = base.measure(
                    model, optimizer, ask, ask_optimizer, probes, packet + 1, calls
                )
                evaluation["kind"] = (
                    "environment" if packet + 1 in (64, 192, 256) else "teacher_cost"
                )
                evaluations.append(evaluation)
    # A causal-input diagnostic: clamp the supplied horizon, keep p/v fixed.
    h2 = [r for r in probes if r["horizon"] == 2]
    with torch.no_grad():
        normal = model(base.features(public_rows(h2)))[0].argmax(-1)
        clamped = model(base.features([{**r, "horizon": 1} for r in public_rows(h2)]))[
            0
        ].argmax(-1)
    result = {
        "variant": variant,
        "arm": arm,
        "seed": seed,
        "answer_initial_hash": initial,
        "answer_final_hash": base.fingerprint(model.state_dict()),
        "ask_initial_hash": ask_initial,
        "ask_final_hash": base.fingerprint(ask.state_dict()),
        "answer_updates": 1024,
        "ask_updates": ask_updates,
        "calls": calls,
        "labels_used": calls * 8,
        "teacher": "offline_perfect_oracle",
        "real_calls": 0,
        "income": income,
        "net_income": income - 2 * calls,
        "credits": credits,
        "blocked_requests": sum(r["raw_ask"] and not r["eligible"] for r in records),
        "evaluations": evaluations,
        "horizon_clamp_prediction_agreement": float((normal == clamped).float().mean()),
        "resources": base.resources(),
    }
    base.save(directory / "summary.json", result)
    checkpoints = ROOT / "checkpoints/offline-tutor-repair"
    checkpoints.mkdir(exist_ok=True)
    torch.save(
        {
            "answer_model": model.state_dict(),
            "ask_model": ask.state_dict(),
            "answer_optimizer": optimizer.state_dict(),
            "ask_optimizer": ask_optimizer.state_dict(),
        },
        checkpoints / f"{variant}-{arm}-{seed}.pt",
    )
    print(
        json.dumps(
            {
                "variant": variant,
                "arm": arm,
                "seed": seed,
                "calls": calls,
                "net_income": result["net_income"],
                "final_per_horizon": [
                    r["greedy_accuracy"] for r in evaluations[-1]["per_horizon"]
                ],
            }
        ),
        flush=True,
    )
    return result


def audit(streams, probes):
    rows = []
    probe_keys = {(r["p"], r["v"], r["horizon"]) for r in probes}
    counts = {}
    for seed, packets in streams.items():
        all_rows = [r for packet in packets for r in packet]
        keys = {(r["p"], r["v"], r["horizon"]) for r in all_rows}
        assert len(keys) == 2048 and not keys & probe_keys
        assert all(
            base.answer(r["p"], r["v"], r["horizon"]) == r["label"] for r in all_rows
        )
        count = np.zeros((8, 4), dtype=int)
        for packet in packets:
            for slot, row in enumerate(packet):
                count[slot, row["label"]] += 1
        counts[seed] = count
        score = float(count.max(-1).sum() / 2048)
        balanced_fraction = (
            sum(
                Counter(r["label"] for r in packet) == Counter({i: 2 for i in range(4)})
                for packet in packets
            )
            / 256
        )
        assert score <= 0.35 and balanced_fraction < 0.5
        rows.append(
            {
                "seed": seed,
                "slot_majority_accuracy": score,
                "exact_balanced_packet_fraction": balanced_fraction,
                "train_unique_and_disjoint_from_probe": True,
            }
        )
    cross_seed = []
    for source in SEEDS:
        for target in SEEDS:
            if source == target:
                continue
            classifier = counts[source].argmax(-1)
            score = float(counts[target][np.arange(8), classifier].sum() / 2048)
            assert score <= 0.35
            cross_seed.append(
                {"fit_seed": source, "held_out_seed": target, "accuracy": score}
            )
    # The complete input resolves deterministic dynamics. Without horizon, some
    # identical p/v pairs have two contradictory targets and cannot be identified.
    paired = []
    for p in np.linspace(-1.9, 1.9, 39):
        for v in np.linspace(-1.9, 1.9, 39):
            if any(min(abs(p + v * h - b) for b in (-1, 0, 1)) < 0.15 for h in (1, 2)):
                continue
            paired.append([base.answer(float(p), float(v), h) for h in (1, 2)])
    disagreements = sum(a != b for a, b in paired)
    return {
        "passed": True,
        "streams": rows,
        "cross_seed_slot_classifier": cross_seed,
        "paired_identifiability": {
            "numeric_pairs": len(paired),
            "different_targets_across_horizons": disagreements,
            "without_horizon_max_pair_accuracy": 1 - disagreements / (2 * len(paired)),
            "with_horizon_oracle_accuracy": 1.0,
        },
        "teacher_selection": "Fixed oracle; policy chooses ASK/SKIP for current whole packet, no answer-dependent teacher or case selection",
        "future_outcomes": "Only observed later rewards enter PPO returns after those outcomes occur; no unobserved outcome enters a committed decision",
    }


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    base.save(
        OUTPUT / "plan.json",
        json.loads((ROOT / "configs/offline-tutor-repair.json").read_text()),
    )
    if (OUTPUT / "source-data-hashes.json").exists():
        raise RuntimeError("Offline preparation already frozen")
    streams = {seed: make_stream(seed) for seed in SEEDS}
    probes = [base.case(20261009, i, h, "probe") for h in (1, 2) for i in range(256)]
    integrity = audit(streams, probes)
    integrity["recorded_teacher_reuse"] = {
        "responses_reused": 0,
        "replacement": "Explicit offline perfect oracle; no cached API records read",
    }
    base.save(OUTPUT / "integrity.json", integrity)
    for seed, packets in streams.items():
        base.save(OUTPUT / f"stream-{seed}.json", packets)
        order = np.random.default_rng(seed + 712000).permutation(256).tolist()
        base.save(OUTPUT / f"interleaved-order-{seed}.json", order)
    base.save(OUTPUT / "probes.json", probes)
    names = [
        "scripts/run_offline_tutor_repair.py",
        "scripts/run_continual_tutor.py",
        "tests/test_offline_tutor_repair.py",
        *[str(p.relative_to(ROOT)) for p in OUTPUT.glob("*.json")],
    ]
    base.save(
        OUTPUT / "source-data-hashes.json",
        {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in names
        },
    )
    print(json.dumps(integrity, indent=2))


def verify():
    for name, expected in json.loads(
        (OUTPUT / "source-data-hashes.json").read_text()
    ).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name


def run():
    verify()
    if not json.loads((OUTPUT / "integrity.json").read_text())["passed"]:
        raise RuntimeError("Offline preflight failed")
    socket.create_connection = no_network
    socket.socket.connect = no_network
    torch.set_num_threads(1)
    probes = json.loads((OUTPUT / "probes.json").read_text())
    started = time.monotonic()
    result = {
        "complete": False,
        "real_calls": 0,
        "network_disabled_in_process": True,
        "runs": [],
    }
    try:
        for variant in VARIANTS:
            for seed in SEEDS:
                packets = json.loads((OUTPUT / f"stream-{seed}.json").read_text())
                if variant == "interleaved":
                    order = json.loads(
                        (OUTPUT / f"interleaved-order-{seed}.json").read_text()
                    )
                    packets = [packets[i] for i in order]
                for arm in (["no_tutor"] if variant == "interleaved" else base.ARMS):
                    result["runs"].append(
                        fit(variant, arm, seed, packets, probes, started)
                    )
                    base.save(OUTPUT / "partial.json", result)
        result["complete"] = True
    finally:
        verify()
        result.update(
            {
                "wall_seconds": time.monotonic() - started,
                "resources": base.resources(),
                "prior_ledgers_unchanged": True,
            }
        )
        base.save(OUTPUT / "summary.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        run()
