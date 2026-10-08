"""Analyze saved curriculum evidence; no learner, teacher, browser or API runs."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs/harder-curriculum"


def read(path):
    return json.loads(path.read_text())


def checkpoint(run, kind, count):
    key = "teacher_labels" if kind == "teacher_cost" else "cases_seen"
    return next(c for c in run["checkpoints"] if c["kind"] == kind and c[key] == count)


def rate(rows, field):
    return sum(row[field] for row in rows) / len(rows) if rows else None


def main():
    plan = read(OUTPUT / "plan.json")
    status = read(OUTPUT / "status.json")
    assert status["complete"]
    runs = read(OUTPUT / "summaries.json")
    assert len(runs) == 9
    calibration = read(OUTPUT / "calibration-summary.json")
    assert calibration["selected"] == status["selected_candidate"]
    assert all(
        hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
        for name, expected in plan["ledger_hashes"].items()
    )
    costs = [
        {c["teacher_labels"] for c in run["checkpoints"] if c["kind"] == "teacher_cost"}
        for run in runs
    ]
    common_costs = sorted(set.intersection(*costs))
    highest_common = max(common_costs) if common_costs else None
    summaries, task_rows, frontier = [], [], []
    frozen_checks = 0
    for run in runs:
        arm, seed = run["arm"], run["seed"]
        directory = OUTPUT / "runs" / f"{arm}-{seed}"
        cases = [
            json.loads(line)
            for line in (directory / "cases.jsonl").read_text().splitlines()
        ]
        blocks = [
            json.loads(line)
            for line in (directory / "blocks.jsonl").read_text().splitlines()
        ]
        main_cases = [c for c in cases if c["phase"] == "curriculum"]
        assert len(cases) == 6400 and len(main_cases) == 6144
        assert (
            sum(c["asked"] for c in cases)
            == run["teacher_labels_including_retention"]
            == run["optimizer_steps"]
        )
        assert all(c["asked"] == (c["confidence"] < 0.9) for c in cases)
        assert all(c["asked"] or c["positive_loss_reduction"] == 0 for c in cases)
        for cp in run["checkpoints"]:
            assert cp["result"]["state_hash_before"] == cp["result"]["state_hash_after"]
            assert cp["teacher_labels"] == cp["optimizer_steps"]
            frozen_checks += 1
            frontier.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "kind": cp["kind"],
                    "cases_seen": cp["cases_seen"],
                    "teacher_labels": cp["teacher_labels"],
                    "macro_accuracy": cp["result"]["macro_accuracy"],
                    "balanced_help_rate": cp["result"]["balanced_help_rate"],
                    "balanced_confident_error": cp["result"][
                        "balanced_confident_error"
                    ],
                }
            )
        warm = checkpoint(run, "environment", 192)
        final = checkpoint(run, "environment", 6144)
        post = checkpoint(run, "after_retention", 6400)
        acquired = [
            t["task"]
            for t in final["result"]["per_task"]
            if t["task"] != 5 and t["teacher_free_accuracy"] >= 0.85
        ]
        drops = [
            final["result"]["per_task"][t]["teacher_free_accuracy"]
            - post["result"]["per_task"][t]["teacher_free_accuracy"]
            for t in acquired
        ]
        help_drop = (
            warm["result"]["balanced_help_rate"] - final["result"]["balanced_help_rate"]
        )
        first, last = main_cases[:1536], main_cases[-1536:]
        for task in range(6):
            before = warm["result"]["per_task"][task]
            after = final["result"]["per_task"][task]
            actual = [c for c in main_cases if c["example"]["task"] == task]
            task_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "task": task,
                    "practice_cases": len(actual),
                    "requested_labels": sum(c["asked"] for c in actual),
                    "first_quarter_request_rate": rate(
                        [c for c in first if c["example"]["task"] == task], "asked"
                    ),
                    "last_quarter_request_rate": rate(
                        [c for c in last if c["example"]["task"] == task], "asked"
                    ),
                    "post_warmup_accuracy": before["teacher_free_accuracy"],
                    "final_accuracy": after["teacher_free_accuracy"],
                    "peak_environment_checkpoint_accuracy": max(
                        cp["result"]["per_task"][task]["teacher_free_accuracy"]
                        for cp in run["checkpoints"]
                        if cp["kind"] == "environment"
                    ),
                    "peak_to_final_accuracy_drop": max(
                        cp["result"]["per_task"][task]["teacher_free_accuracy"]
                        for cp in run["checkpoints"]
                        if cp["kind"] == "environment"
                    )
                    - after["teacher_free_accuracy"],
                    "post_warmup_fixed_probe_help": before["would_request_help"],
                    "final_fixed_probe_help": after["would_request_help"],
                    "final_confident_error": after["confident_error_fraction"],
                    "after_retention_accuracy": post["result"]["per_task"][task][
                        "teacher_free_accuracy"
                    ],
                }
            )
        adaptive_blocks = [
            b for b in blocks if b["phase"] == "curriculum" and b["block"] >= 6
        ]
        at_cost = (
            checkpoint(run, "teacher_cost", highest_common) if highest_common else None
        )
        summaries.append(
            {
                "arm": arm,
                "seed": seed,
                "final_macro_accuracy": final["result"]["macro_accuracy"],
                "final_worst_task_accuracy": min(
                    t["teacher_free_accuracy"] for t in final["result"]["per_task"]
                ),
                "curriculum_teacher_labels": final["teacher_labels"],
                "total_teacher_labels": run["teacher_labels_including_retention"],
                "post_warmup_balanced_help": warm["result"]["balanced_help_rate"],
                "final_balanced_help": final["result"]["balanced_help_rate"],
                "balanced_help_drop": help_drop,
                "final_balanced_autonomous_correct": final["result"][
                    "balanced_autonomous_correct"
                ],
                "final_balanced_confident_error": final["result"][
                    "balanced_confident_error"
                ],
                "less_help_with_competence": help_drop >= 0.20
                and final["result"]["macro_accuracy"] >= 0.85
                and final["result"]["balanced_confident_error"] <= 0.05,
                "training_query_rate_first_quarter": rate(first, "asked"),
                "training_query_rate_last_quarter": rate(last, "asked"),
                "wide_margin_practice_share_first_quarter": sum(
                    c["example"]["task"] % 2 == 0 for c in first
                )
                / len(first),
                "wide_margin_practice_share_last_quarter": sum(
                    c["example"]["task"] % 2 == 0 for c in last
                )
                / len(last),
                "mean_selection_total_variation_from_uniform": statistics.mean(
                    sum(abs(p - 1 / 6) for p in b["selection_probabilities"]) / 2
                    for b in adaptive_blocks
                ),
                "matched_label_cost": highest_common,
                "accuracy_at_matched_label_cost": (
                    at_cost["result"]["macro_accuracy"] if at_cost else None
                ),
                "practice_cases_at_matched_label_cost": (
                    at_cost["cases_seen"] if at_cost else None
                ),
                "retention_additional_labels_and_updates": post["teacher_labels"]
                - final["teacher_labels"],
                "retention_acquired_nonfocus_tasks": acquired,
                "retention_acquired_tasks_still_above_threshold": sum(
                    post["result"]["per_task"][t]["teacher_free_accuracy"] >= 0.85
                    for t in acquired
                ),
                "retention_mean_accuracy_drop_on_acquired": (
                    statistics.mean(drops) if drops else None
                ),
                "retention_max_accuracy_drop_on_acquired": (
                    max(drops) if drops else None
                ),
                "retention_macro_accuracy": post["result"]["macro_accuracy"],
            }
        )
    paired = []
    for seed in plan["seeds"]:
        by_arm = {r["arm"]: r for r in summaries if r["seed"] == seed}
        assert len({r["initial_weight_hash"] for r in runs if r["seed"] == seed}) == 1
        row = {"seed": seed}
        for other in ["random", "fixed"]:
            row["final_accuracy_advantage_over_" + other] = (
                by_arm["adaptive"]["final_macro_accuracy"]
                - by_arm[other]["final_macro_accuracy"]
            )
            row["matched_cost_advantage_over_" + other] = (
                by_arm["adaptive"]["accuracy_at_matched_label_cost"]
                - by_arm[other]["accuracy_at_matched_label_cost"]
            )
        row["beats_both_by_two_points_at_final"] = all(
            row["final_accuracy_advantage_over_" + arm] >= 0.02
            for arm in ["random", "fixed"]
        )
        row["beats_both_by_two_points_at_matched_cost"] = all(
            row["matched_cost_advantage_over_" + arm] >= 0.02
            for arm in ["random", "fixed"]
        )
        paired.append(row)
    decision = (
        sum(r["beats_both_by_two_points_at_final"] for r in paired) >= 2
        and sum(r["beats_both_by_two_points_at_matched_cost"] for r in paired) >= 2
    )
    numeric = [
        k
        for k, v in summaries[0].items()
        if isinstance(v, (float, int)) and not isinstance(v, bool) and k != "seed"
    ]
    means = {
        arm: {
            k: statistics.mean(
                r[k] for r in summaries if r["arm"] == arm and r[k] is not None
            )
            for k in numeric
        }
        for arm in plan["arms"]
    }
    audit = {
        "status": status,
        "learnability": read(OUTPUT / "learnability.json"),
        "common_teacher_cost_milestones": common_costs,
        "highest_common_teacher_cost": highest_common,
        "per_run": summaries,
        "per_task": task_rows,
        "paired_comparisons": paired,
        "predeclared_adaptive_advantage_supported": decision,
        "means": means,
        "frozen_evaluation_snapshots": frozen_checks,
        "main_simulated_labels": sum(r["curriculum_teacher_labels"] for r in summaries),
        "total_simulated_labels_including_retention": sum(
            r["total_teacher_labels"] for r in summaries
        ),
        "calibration_simulated_labels_separate": sum(
            r["total_oracle_labels"] for r in calibration["calibration_runs"]
        ),
        "calibration": calibration,
        "api_cost_usd": 0,
    }
    (OUTPUT / "analysis.json").write_text(json.dumps(audit, indent=2) + "\n")
    for name, rows in [
        ("frontier.csv", frontier),
        ("per-run.csv", summaries),
        ("per-task.csv", task_rows),
    ]:
        with (OUTPUT / name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(
        json.dumps(
            {
                k: audit[k]
                for k in [
                    "highest_common_teacher_cost",
                    "per_run",
                    "paired_comparisons",
                    "predeclared_adaptive_advantage_supported",
                    "means",
                    "frozen_evaluation_snapshots",
                    "main_simulated_labels",
                    "total_simulated_labels_including_retention",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
