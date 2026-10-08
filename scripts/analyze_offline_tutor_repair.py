#!/usr/bin/env python3
"""Recompute offline results and gates from saved evidence, without ML imports."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/offline-tutor-repair"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + "\n")


def table(name, rows):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    plan = read(OUT / "plan.json")
    hashes = read(OUT / "source-data-hashes.json")
    for name, expected in hashes.items():
        assert sha(ROOT / name) == expected, name
    for prior in plan.get("prior_ledgers", []):
        assert sha(ROOT / prior["path"]) == prior["sha256"]
        assert read(ROOT / prior["path"])["closed"]
    summary = read(OUT / "summary.json")
    assert summary["complete"] and summary["real_calls"] == 0
    assert len(summary["runs"]) == plan["runs"] == 30
    assert read(OUT / "integrity.json")["passed"]
    results, phases, probes, cost_checkpoints = [], [], [], []
    frozen_checks, packet_checks, replay_examples = 0, 0, 0
    for run in summary["runs"]:
        variant, arm, seed = run["variant"], run["arm"], run["seed"]
        path = OUT / "runs" / f"{variant}-{arm}-{seed}"
        assert read(path / "summary.json") == run
        packets = [
            json.loads(line)
            for line in (path / "packets.jsonl").read_text().splitlines()
        ]
        stream = read(OUT / f"stream-{seed}.json")
        if variant == "interleaved":
            order = read(OUT / f"interleaved-order-{seed}.json")
            assert sorted(order) == list(range(256))
            stream = [stream[i] for i in order]
        assert len(packets) == 256
        credits, calls, income = 4.0, 0, 0.0
        for i, row in enumerate(packets):
            cases = stream[i]
            assert row["packet"] == i and row["horizon"] == cases[0]["horizon"]
            assert row["credits_before"] == credits
            assert row["eligible"] == (credits >= 2 and calls < 12)
            assert row["granted"] == (row["eligible"] and row["raw_ask"])
            assert row["correct"] == [
                a == r["label"]
                for a, r in zip(row["committed_actions"], cases, strict=True)
            ]
            pay = 0.25 if row["horizon"] == 1 else 1.0
            assert row["income"] == sum(row["correct"]) * pay
            assert row["real_cost_usd"] == 0 and row["optimizer_steps"] == 4
            if row["granted"]:
                assert row["teacher"] == "offline_perfect_oracle"
                assert row["teacher_labels"] == [r["label"] for r in cases]
                assert row["tutor_labels_used"] == 8
                calls += 1
            else:
                assert row["teacher"] is None and row["teacher_labels"] is None
                assert row["tutor_labels_used"] == 0
            assert row["replay_rows"] == (
                32 if i and variant in {"recent_feedback", "balanced_feedback"} else 0
            )
            replay_examples += row["replay_rows"] * 4
            income += row["income"]
            credits += row["income"] - row["granted"] * 2
            assert row["credits_after"] == credits >= 0
            assert row["calls"] == calls <= 12
            assert row["net_reward"] == (row["income"] - row["granted"] * 2) / 8
            packet_checks += 1
        assert income == run["income"] and income - calls * 2 == run["net_income"]
        assert run["credits"] == credits and run["calls"] == calls
        assert run["labels_used"] == calls * 8 and run["real_calls"] == 0
        assert run["answer_updates"] == 1024
        assert run["ask_updates"] == (64 if arm == "learned" else 0)
        assert run["answer_initial_hash"] != run["answer_final_hash"]
        assert (run["ask_initial_hash"] != run["ask_final_hash"]) == (arm == "learned")
        evaluations = {}
        for evaluation in run["evaluations"]:
            assert evaluation["state_hash_before"] == evaluation["state_hash_after"]
            frozen_checks += 1
            kind = evaluation.get("kind", "initial")
            if kind != "teacher_cost":
                evaluations[evaluation["packet"]] = evaluation
            else:
                cost_checkpoints.append(
                    {
                        "variant": variant,
                        "arm": arm,
                        "seed": seed,
                        "packet": evaluation["packet"],
                        "calls": evaluation["calls"],
                        "labels": evaluation["teacher_labels"],
                        "cases_seen": evaluation["cases_seen"],
                        "macro_accuracy": evaluation["macro_greedy_accuracy"],
                    }
                )
            for h in evaluation["per_horizon"]:
                probes.append(
                    {
                        "variant": variant,
                        "arm": arm,
                        "seed": seed,
                        "packet": evaluation["packet"],
                        "calls": evaluation["calls"],
                        "kind": kind,
                        **h,
                    }
                )
        assert set(evaluations) == {0, 64, 192, 256}

        def accuracy(packet, horizon, values=evaluations):
            return values[packet]["per_horizon"][horizon - 1]["greedy_accuracy"]

        a0, ab, afinal = accuracy(64, 1), accuracy(192, 1), accuracy(256, 1)
        b0, bfinal = accuracy(192, 2), accuracy(256, 2)
        conditions = {
            "active_h1_at64": a0 >= 0.85,
            "active_h2_at192": b0 >= 0.85,
            "h1_at192": ab >= 0.80,
            "both_final": min(afinal, bfinal) >= 0.80,
            "h1_drop_at_most_10pp": a0 - ab <= 0.10,
            "h2_drop_at_most_10pp": b0 - bfinal <= 0.10,
            "ask_mechanism": arm != "learned"
            or run["ask_initial_hash"] != run["ask_final_hash"]
            and 1 <= calls <= 12,
        }
        metrics = {
            "variant": variant,
            "arm": arm,
            "seed": seed,
            "calls": calls,
            "labels": calls * 8,
            "gross_income": income,
            "synthetic_fees": calls * 2,
            "net_income": run["net_income"],
            "h1_at64": a0,
            "h1_at192": ab,
            "h2_at192": b0,
            "h1_final": afinal,
            "h2_final": bfinal,
            "macro_final": (afinal + bfinal) / 2,
            "h1_drop_pp": (a0 - ab) * 100,
            "h2_drop_pp": (b0 - bfinal) * 100,
            "horizon_clamp_prediction_agreement": run[
                "horizon_clamp_prediction_agreement"
            ],
            "blocked_requests": run["blocked_requests"],
            "gate_passed": all(conditions.values()),
            "failed_conditions": ";".join(k for k, v in conditions.items() if not v),
        }
        results.append(metrics)
        for name, start, end in [
            ("first_64_packets", 0, 64),
            ("middle_128_packets", 64, 192),
            ("last_64_packets", 192, 256),
        ]:
            rows = packets[start:end]
            eligible = [r for r in rows if r["eligible"]]
            phases.append(
                {
                    "variant": variant,
                    "arm": arm,
                    "seed": seed,
                    "segment": name,
                    "calls": sum(r["granted"] for r in rows),
                    "raw_requests": sum(r["raw_ask"] for r in rows),
                    "eligible_packets": len(eligible),
                    "eligible_request_rate": (
                        mean(r["raw_ask"] for r in eligible) if eligible else None
                    ),
                    "eligible_learned_probability": (
                        mean(r["ask_probability"] for r in eligible)
                        if eligible and arm == "learned"
                        else None
                    ),
                    "blocked_requests": sum(
                        r["raw_ask"] and not r["eligible"] for r in rows
                    ),
                    "gross_income": sum(r["income"] for r in rows),
                }
            )
    assert packet_checks == 7680
    for seed in plan["seeds"]:
        assert (
            len(
                {r["answer_initial_hash"] for r in summary["runs"] if r["seed"] == seed}
            )
            == 1
        )
    groups = defaultdict(list)
    for row in results:
        groups[row["variant"], row["arm"]].append(row)
    means = []
    fields = [
        "calls",
        "gross_income",
        "synthetic_fees",
        "net_income",
        "h1_at64",
        "h1_at192",
        "h2_at192",
        "h1_final",
        "h2_final",
        "macro_final",
        "h1_drop_pp",
        "h2_drop_pp",
        "horizon_clamp_prediction_agreement",
    ]
    for (variant, arm), rows in groups.items():
        assert len(rows) == 3
        means.append(
            {
                "variant": variant,
                "arm": arm,
                **{field: mean(r[field] for r in rows) for field in fields},
            }
        )
    target = [r for r in results if r["variant"] == "balanced_feedback"]
    assert len(target) == 9
    gate = all(r["gate_passed"] for r in target)
    by_key = {(r["variant"], r["arm"], r["seed"]): r for r in results}
    utility = []
    for variant in ["corrected_blocked", "recent_feedback", "balanced_feedback"]:
        wins = sum(
            all(
                by_key[variant, "learned", seed]["net_income"]
                >= 1.05 * by_key[variant, arm, seed]["net_income"]
                for arm in ["heuristic", "no_tutor"]
            )
            for seed in plan["seeds"]
        )
        macro_guard = all(
            by_key[variant, "learned", seed]["macro_final"]
            >= by_key[variant, "heuristic", seed]["macro_final"] - 0.05
            for seed in plan["seeds"]
        )
        utility.append(
            {
                "variant": variant,
                "qualifying_income_wins": wins,
                "macro_guard": macro_guard,
                "passed": wins >= 2 and macro_guard,
            }
        )
    result = {
        "complete": True,
        "integrity_gate_passed": True,
        "learnability_retention_gate_passed": gate,
        "gate_passed_runs": sum(r["gate_passed"] for r in target),
        "gate_total_runs": 9,
        "failed_gate_runs": [r for r in target if not r["gate_passed"]],
        "utility": utility,
        "means": means,
        "frozen_evaluation_checks": frozen_checks,
        "packet_accounting_checks": packet_checks,
        "replay_examples_including_repeated_epochs": replay_examples,
        "real_calls": 0,
        "prior_ledgers_unchanged": True,
        "task_board_executed": False,
        "task_board_blocker": (
            "Preregistered <=10pp first-task loss condition failed in three balanced-feedback runs"
            if not gate
            else None
        ),
        "resources": summary["resources"],
        "wall_seconds": summary["wall_seconds"],
        "teacher": "offline_perfect_oracle; no real-response reuse",
        "source_data_manifest_sha256": sha(OUT / "source-data-hashes.json"),
        "plan_sha256": sha(OUT / "plan.json"),
    }
    save("analysis.json", result)
    table("runs.csv", results)
    table("means.csv", means)
    table("phases.csv", phases)
    table("probes.csv", probes)
    table("cost-checkpoints.csv", cost_checkpoints)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"means", "failed_gate_runs"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
