#!/usr/bin/env python3
"""Audit saved board evidence; standard library only, local or directly in ZIP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
REL = "logs/task-board-study"
ARMS = ["current_no_tutor", "board_no_tutor", "current_tutor", "board_tutor"]
SEEDS = [307, 401, 503]


class Evidence:
    def __init__(self, bundle: Path | None):
        self.archive = zipfile.ZipFile(bundle) if bundle else None
        self.names = set(self.archive.namelist()) if self.archive else set()

    def raw(self, name: str) -> bytes:
        if self.archive:
            selected = "source/" + name if "source/" + name in self.names else name
            return self.archive.read(selected)
        return (ROOT / name).read_bytes()

    def read(self, name: str):
        return json.loads(self.raw(name))

    def sha(self, name: str):
        return hashlib.sha256(self.raw(name)).hexdigest()


def save(directory: Path, name: str, value) -> None:
    (directory / name).write_text(json.dumps(value, indent=2) + "\n")


def table(directory: Path, name: str, rows: list[dict]) -> None:
    with (directory / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values: list[float]) -> dict:
    assert len(values) == 3
    # Exact 97.5% quantile for Student t with two degrees of freedom.
    critical = math.sqrt(2 * 0.95**2 / (1 - 0.95**2))
    center = mean(values)
    margin = critical * stdev(values) / math.sqrt(3)
    return {
        "paired_seed_values": values,
        "mean": center,
        "exploratory_95pct_t_interval": [center - margin, center + margin],
        "independent_units": 3,
        "note": "Three seed-level differences; not independent packet trials or a significance claim",
    }


def public_view(schedule, cycle, visible):
    contract = cycle // 32

    def pay(h):
        return 0.25 if h == 1 else 1.0

    future = []
    if visible:
        for offset in (1, 2):
            index = contract + offset
            if index < 8:
                h = schedule[index]
                future.append(
                    {"skill": h, "pay": pay(h), "available_in": index * 32 - cycle}
                )
    h = schedule[contract]
    return {
        "current": {"skill": h, "pay": pay(h), "remaining": 32 - cycle % 32},
        "past_contracts": schedule[:contract],
        "future": future,
    }


def main(bundle: Path | None = None, output: Path | None = None):
    evidence = Evidence(bundle)
    output = output or ROOT / REL
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = evidence.read(f"{REL}/source-data-hashes.json")
    for name, expected in source_manifest.items():
        assert evidence.sha(name) == expected, name
    preserved = evidence.read(f"{REL}/preserved-evidence.json")
    for name, expected in preserved.items():
        assert evidence.sha(name) == expected, name
    summary = evidence.read(f"{REL}/summary.json")
    plan = evidence.read(f"{REL}/plan.json")
    assert (
        summary["complete"]
        and summary["real_calls"] == 0
        and len(summary["runs"]) == 72
    )
    assert summary["wall_seconds"] + summary["preparation_seconds"] < 600
    assert summary["resources"]["peak_rss_mib"] < 1800
    assert summary["resources"]["output_bytes"] < 100 * 1024 * 1024
    records, returns, probes, choice_rows = [], [], [], []
    cycle_checks, packet_checks, frozen_checks, calls_total = 0, 0, 0, 0
    by_run = {}
    for run in summary["runs"]:
        arm, seed, episode = run["arm"], run["seed"], run["episode"]
        training = episode < 4
        visible = arm.startswith("board_")
        tutor = not arm.endswith("no_tutor")
        directory = f"{REL}/runs/{arm}-{seed}-{episode}"
        assert evidence.read(directory + "/summary.json") == run
        data_name = f"{REL}/data-{seed}-{episode}.json"
        assert evidence.sha(data_name) == run["data_sha256"]
        data = evidence.read(data_name)
        schedule = data["schedule"]
        rows = [
            json.loads(line)
            for line in evidence.raw(directory + "/cycles.jsonl").splitlines()
        ]
        assert len(rows) == 256 and run["schedule"] == schedule
        assert run["phase"] == ("train" if training else "assessment")
        calls, income, credits, updates = 0, 0.0, 4.0, 0
        seen = [0, 0]
        correct_paid = [0, 0]
        first_correct, first_n = [0, 0], [0, 0]
        ret_correct, ret_n = [0, 0], [0, 0]
        practice_counts, query_counts = [0, 0], [0, 0]
        denied, requests, counterfactual_l1, p_b_changes = 0, 0, [], []
        for cycle, row in enumerate(rows):
            assert row["cycle"] == cycle
            assert row["view"] == public_view(schedule, cycle, visible)
            assert row["credits_before"] == credits
            probs = row["action_probabilities"]
            assert len(probs) == 4 and abs(sum(probs) - 1) < 1e-6
            assert all(0 <= p <= 1 for p in probs)
            if not tutor:
                assert probs[2:] == [0.0, 0.0]
            action = row["chosen_action"]
            skill = action % 2 + 1
            assert row["practice_skill"] == skill
            assert row["query_requested"] == (action >= 2)
            assert row["eligible"] == (tutor and credits >= 2 and calls < 12)
            assert row["granted"] == (action >= 2 and row["eligible"])
            assert row["denied"] == (action >= 2 and not row["eligible"])
            practice_counts[skill - 1] += 1
            if row["granted"]:
                credits -= 2
                calls += 1
                query_counts[skill - 1] += 1
            requests += int(row["query_requested"])
            denied += int(row["denied"])
            for kind, expected in [
                ("practice", data["practice"][cycle][skill - 1]),
                ("paid", data["paid"][cycle]),
            ]:
                packet = row[kind]
                assert packet["examples"] == expected
                chosen = packet["committed_answers"]
                assert len(chosen) == 8 and all(
                    type(a) is int and 0 <= a <= 3 for a in chosen
                )
                correct = [
                    a == case["label"] for a, case in zip(chosen, expected, strict=True)
                ]
                assert packet["correct"] == correct
                h = expected[0]["horizon"]
                earnings = (
                    sum(correct) * (0.25 if h == 1 else 1.0) if kind == "paid" else 0
                )
                assert packet["paid_income"] == earnings
                assert packet["optimizer_steps"] == 4
                memory_before = [
                    packet["memory_seen_before"].get(str(i), 0) for i in range(2)
                ]
                assert memory_before == seen
                expected_replay = (
                    [16, 16]
                    if all(seen)
                    else [32, 0] if seen[0] else [0, 32] if seen[1] else []
                )
                assert packet["replay_skill_counts"] == expected_replay
                seen[h - 1] += 8
                labelled = kind == "practice" and row["granted"]
                assert packet["teacher_labels"] == (
                    [case["label"] for case in expected] if labelled else None
                )
                assert packet["teacher"] == (
                    "offline_perfect_oracle" if labelled else None
                )
                assert packet["tutor_labels_used"] == (8 if labelled else 0)
                if kind == "paid":
                    correct_paid[h - 1] += sum(correct)
                    contract = cycle // 32
                    if cycle % 32 < 4:
                        if h not in schedule[:contract]:
                            first_correct[h - 1] += sum(correct)
                            first_n[h - 1] += 8
                        elif contract and h != schedule[contract - 1]:
                            ret_correct[h - 1] += sum(correct)
                            ret_n[h - 1] += 8
                packet_checks += 1
            income += row["paid"]["paid_income"]
            credits += row["paid"]["paid_income"]
            assert row["income"] == row["paid"]["paid_income"]
            assert row["calls"] == calls <= 12 and row["credits_after"] == credits >= 0
            assert row["allocator_reward"] == (row["income"] - 2 * row["granted"]) / 8
            step_updates = 4 if training and (cycle + 1) % 16 == 0 else 0
            assert row["allocator_optimizer_steps"] == step_updates
            updates += step_updates
            cf = row["future_counterfactual"]
            if cf is not None:
                assert not training
                if not visible:
                    assert cf["l1_change"] == 0 and cf["probabilities"] == probs
                counterfactual_l1.append(cf["l1_change"])
                p_b_changes.append(
                    abs(
                        sum(cf["probabilities"][i] for i in (1, 3))
                        - sum(probs[i] for i in (1, 3))
                    )
                )
            cycle_checks += 1
        assert run["income"] == income and run["net_income"] == income - calls * 2
        assert run["credits"] == credits == 4 + run["net_income"]
        assert run["calls"] == calls and run["practice_counts"] == practice_counts
        assert run["query_counts"] == query_counts and sum(practice_counts) == 256
        assert sum(seen) == 4096 and run["answer_updates"] == 2048
        assert run["allocator_updates"] == updates == (64 if training else 0)
        assert run["answer_initial_hash"] != run["answer_final_hash"]
        if not training:
            assert run["allocator_initial_hash"] == run["allocator_final_hash"]
            assert run["actor_initial_hash"] == run["actor_final_hash"]
        else:
            assert run["actor_initial_hash"] != run["actor_final_hash"]
        by_run[arm, seed, episode] = run
        calls_total += calls
        evaluations = {e["packet"]: e for e in run["evaluations"]}
        assert set(evaluations) == set(range(0, 257, 32))
        for evaluation in run["evaluations"]:
            assert evaluation["state_hash_before"] == evaluation["state_hash_after"]
            assert evaluation["cases_seen"] == evaluation["packet"] * 8
            frozen_checks += 1
            for h in evaluation["per_horizon"]:
                probes.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "episode": episode,
                        "phase": run["phase"],
                        "cycle": evaluation["packet"],
                        "paid_cases_seen": evaluation["packet"] * 8,
                        "practice_cases_seen": evaluation["packet"] * 8,
                        "total_cases_seen": evaluation["packet"] * 16,
                        **h,
                    }
                )
        drops = []
        for contract, h in enumerate(schedule):
            if (
                not contract
                or h == schedule[contract - 1]
                or h not in schedule[:contract]
            ):
                continue
            earlier = max(i for i in range(contract) if schedule[i] == h)
            previous_end, start = (earlier + 1) * 32, contract * 32
            before = evaluations[previous_end]["per_horizon"][h - 1]["greedy_accuracy"]
            ready = evaluations[start]["per_horizon"][h - 1]["greedy_accuracy"]
            outcomes = [
                c for row in rows[start : start + 4] for c in row["paid"]["correct"]
            ]
            window = rows[previous_end:start]
            drop = (before - ready) * 100
            drops.append(drop)
            returns.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "episode": episode,
                    "phase": run["phase"],
                    "contract": contract,
                    "skill": h,
                    "absence_cycles": start - previous_end,
                    "practice_packets_for_returning_skill_during_absence": sum(
                        r["practice_skill"] == h for r in window
                    ),
                    "queries_for_returning_skill_during_absence": sum(
                        r["practice_skill"] == h and r["granted"] for r in window
                    ),
                    "previous_end_probe_accuracy": before,
                    "before_return_probe_accuracy": ready,
                    "maintenance_drop_pp": drop,
                    "drop_exceeds_10pp": drop > 10,
                    "return_job_correct": sum(outcomes),
                    "return_job_cases": len(outcomes),
                    "return_job_accuracy": mean(outcomes),
                }
            )
        final = evaluations[256]
        record = {
            "arm": arm,
            "seed": seed,
            "episode": episode,
            "phase": run["phase"],
            "calls": calls,
            "labels": calls * 8,
            "gross_income": income,
            "synthetic_fees": calls * 2,
            "net_income": run["net_income"],
            "paid_accuracy": sum(correct_paid) / 2048,
            "A_paid_accuracy": correct_paid[0] / 1024,
            "B_paid_accuracy": correct_paid[1] / 1024,
            "return_correct": sum(ret_correct),
            "return_cases": sum(ret_n),
            "return_accuracy": sum(ret_correct) / sum(ret_n),
            "A_return_correct": ret_correct[0],
            "A_return_cases": ret_n[0],
            "B_return_correct": ret_correct[1],
            "B_return_cases": ret_n[1],
            "first_appearance_accuracy": sum(first_correct) / sum(first_n),
            "cold_start_accuracy": first_correct[schedule[0] - 1]
            / first_n[schedule[0] - 1],
            "first_later_paid_skill_accuracy": first_correct[2 - schedule[0]]
            / first_n[2 - schedule[0]],
            "final_macro": final["macro_greedy_accuracy"],
            "A_final": final["per_horizon"][0]["greedy_accuracy"],
            "B_final": final["per_horizon"][1]["greedy_accuracy"],
            "practice_B_fraction": practice_counts[1] / 256,
            "raw_query_requests": requests,
            "denied_query_requests": denied,
            "counterfactual_states": len(counterfactual_l1),
            "mean_counterfactual_action_l1": (
                mean(counterfactual_l1) if counterfactual_l1 else 0
            ),
            "mean_counterfactual_practice_B_change": (
                mean(p_b_changes) if p_b_changes else 0
            ),
            "max_maintenance_drop_pp": max(drops),
            "maintenance_drops_over_10pp": sum(d > 10 for d in drops),
        }
        records.append(record)
        for segment, start, end in [
            ("first_quarter", 0, 64),
            ("middle", 64, 192),
            ("last_quarter", 192, 256),
        ]:
            subset = rows[start:end]
            eligible = [r for r in subset if r["eligible"]]
            choice_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "episode": episode,
                    "phase": run["phase"],
                    "segment": segment,
                    "practice_B_fraction": mean(
                        r["practice_skill"] == 2 for r in subset
                    ),
                    "query_probability": mean(
                        sum(r["action_probabilities"][2:]) for r in subset
                    ),
                    "eligible_cycles": len(eligible),
                    "eligible_request_rate": (
                        mean(r["query_requested"] for r in eligible)
                        if eligible
                        else None
                    ),
                    "calls": sum(r["granted"] for r in subset),
                    "denied": sum(r["denied"] for r in subset),
                }
            )
    assert cycle_checks == 18432 and packet_checks == 36864 and frozen_checks == 648
    assert calls_total <= 432
    for seed in SEEDS:
        for episode in range(6):
            assert (
                len({by_run[a, seed, episode]["answer_initial_hash"] for a in ARMS})
                == 1
            )
            assert len({by_run[a, seed, episode]["data_sha256"] for a in ARMS}) == 1
        assert len({by_run[a, seed, 0]["allocator_initial_hash"] for a in ARMS}) == 1
        for arm in ARMS:
            for episode in range(1, 6):
                assert (
                    by_run[arm, seed, episode]["allocator_initial_hash"]
                    == by_run[arm, seed, episode - 1]["allocator_final_hash"]
                )
    seed_metrics = []
    for arm in ARMS:
        for seed in SEEDS:
            rows = [
                r
                for r in records
                if r["arm"] == arm and r["seed"] == seed and r["phase"] == "assessment"
            ]
            assert len(rows) == 2
            row = {"arm": arm, "seed": seed}
            fields = [
                "net_income",
                "gross_income",
                "synthetic_fees",
                "calls",
                "paid_accuracy",
                "first_appearance_accuracy",
                "cold_start_accuracy",
                "first_later_paid_skill_accuracy",
                "final_macro",
                "A_final",
                "B_final",
                "practice_B_fraction",
                "mean_counterfactual_action_l1",
                "mean_counterfactual_practice_B_change",
            ]
            row.update({k: mean(r[k] for r in rows) for k in fields})
            row["return_accuracy"] = sum(r["return_correct"] for r in rows) / sum(
                r["return_cases"] for r in rows
            )
            for h in ("A", "B"):
                row[f"{h}_return_accuracy"] = sum(
                    r[f"{h}_return_correct"] for r in rows
                ) / sum(r[f"{h}_return_cases"] for r in rows)
            seed_metrics.append(row)
    by_seed = {(r["arm"], r["seed"]): r for r in seed_metrics}
    means = [
        {
            "arm": arm,
            **{
                field: mean(r[field] for r in seed_metrics if r["arm"] == arm)
                for field in seed_metrics[0]
                if field not in {"arm", "seed"}
            },
        }
        for arm in ARMS
    ]
    contrasts = []
    for name, treatment, control in [
        ("board_without_tutor", "board_no_tutor", "current_no_tutor"),
        ("board_with_tutor", "board_tutor", "current_tutor"),
        ("tutor_current_only", "current_tutor", "current_no_tutor"),
        ("tutor_with_board", "board_tutor", "board_no_tutor"),
    ]:
        contrasts.append(
            {
                "name": name,
                "treatment": treatment,
                "control": control,
                "return_accuracy_pp": interval(
                    [
                        (
                            by_seed[treatment, s]["return_accuracy"]
                            - by_seed[control, s]["return_accuracy"]
                        )
                        * 100
                        for s in SEEDS
                    ]
                ),
                "net_credits_per_stream": interval(
                    [
                        by_seed[treatment, s]["net_income"]
                        - by_seed[control, s]["net_income"]
                        for s in SEEDS
                    ]
                ),
            }
        )
    primary = contrasts[0]
    qualifying = sum(
        delta >= 5 for delta in primary["return_accuracy_pp"]["paired_seed_values"]
    )
    cash_guard = primary["net_credits_per_stream"]["mean"] >= 0
    skill_guard = all(
        mean(by_seed["board_no_tutor", s][f"{h}_return_accuracy"] for s in SEEDS)
        >= 0.80
        for h in ("A", "B")
    )
    tutor_gates = []
    for prefix in ("current", "board"):
        wins = sum(
            by_seed[prefix + "_tutor", s]["net_income"]
            >= 1.05 * by_seed[prefix + "_no_tutor", s]["net_income"]
            for s in SEEDS
        )
        guard = all(
            by_seed[prefix + "_tutor", s]["final_macro"]
            >= by_seed[prefix + "_no_tutor", s]["final_macro"] - 0.05
            for s in SEEDS
        )
        tutor_gates.append(
            {
                "board_condition": prefix,
                "income_wins_of_three": wins,
                "macro_guard": guard,
                "passed": wins >= 2 and guard,
            }
        )
    assessed_returns = [r for r in returns if r["phase"] == "assessment"]
    result = {
        "complete": True,
        "integrity_passed": True,
        "approved_protocol_sha256": plan["proposal_sha256"],
        "source_manifest_sha256": evidence.sha(f"{REL}/source-data-hashes.json"),
        "prior_files_unchanged": len(preserved),
        "streams": 72,
        "training_streams": 48,
        "assessment_streams": 24,
        "cycle_checks": cycle_checks,
        "packet_checks": packet_checks,
        "frozen_probe_checks": frozen_checks,
        "continuous_answer_updates_per_assessment_stream": 2048,
        "legacy_probe_cases_seen_scope": "paid cases only; probes.csv explicitly reports paid, practice and total exposure",
        "assessment_allocator_frozen": True,
        "paired_data_and_initialization_verified": True,
        "simulated_queries": calls_total,
        "assessment_simulated_queries": sum(
            r["calls"] for r in records if r["phase"] == "assessment"
        ),
        "real_calls": 0,
        "means": means,
        "contrasts": contrasts,
        "primary_preparation_signal": {
            "passed": qualifying >= 2 and cash_guard and skill_guard,
            "seeds_with_at_least_5pp_gain": qualifying,
            "net_income_guard": cash_guard,
            "per_skill_80pct_guard": skill_guard,
        },
        "tutor_utility": tutor_gates,
        "headroom_pp_in_current_only_no_tutor_by_seed": {
            str(s): (1 - by_seed["current_no_tutor", s]["return_accuracy"]) * 100
            for s in SEEDS
        },
        "assessment_return_events": len(assessed_returns),
        "assessment_maintenance_drops_over_10pp": sum(
            r["drop_exceeds_10pp"] for r in assessed_returns
        ),
        "assessment_max_maintenance_drop_pp": max(
            r["maintenance_drop_pp"] for r in assessed_returns
        ),
        "resources": summary["resources"],
        "execution_seconds": summary["wall_seconds"],
        "preparation_seconds": summary["preparation_seconds"],
        "new_reservations_usd": "0",
        "real_tutor_extension_executed": False,
        "old_retention_gate_verdict": "failed, unchanged; this separately approved study treats maintenance as an outcome",
    }
    save(output, "analysis.json", result)
    table(output, "streams.csv", records)
    table(output, "assessment-seeds.csv", seed_metrics)
    table(output, "assessment-means.csv", means)
    table(output, "return-events.csv", returns)
    table(output, "probes.csv", probes)
    table(output, "choices.csv", choice_rows)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"means", "contrasts"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    main(args.bundle, args.output_dir)
