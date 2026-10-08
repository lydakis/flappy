"""Summarize saved recovery logs without running a browser, learner or API."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs/cpu-browser-recovery"


def read(path):
    return json.loads(path.read_text())


def score(result):
    rows = [row for ep in result["episodes"] for row in ep["browser_trials"]]
    clicks = [r for r in rows if r["action"]["name"] == "click"]
    return {
        "correct": sum(r["correct"] for r in rows),
        "decisions": len(rows),
        "accuracy": result["choice_accuracy"],
        "left_correct": sum(r["correct"] for r in rows if r["cue"] == "left"),
        "left_decisions": sum(r["cue"] == "left" for r in rows),
        "right_correct": sum(r["correct"] for r in rows if r["cue"] == "right"),
        "right_decisions": sum(r["cue"] == "right" for r in rows),
        "clicks": len(clicks),
        "missed_clicks": sum(not r["click_delivered"] for r in clicks),
        "action_counts": dict(
            Counter(
                r["action"]["name"] + (r["action"].get("selector") or "") for r in rows
            )
        ),
        "learning_hash_unchanged": result["learning_hash_before"]
        == result["learning_hash_after"],
    }


def main():
    status = read(OUTPUT / "status.json")
    plan = read(OUTPUT / "plan.json")
    evaluations = []
    training = []
    samples = []
    frozen = 0
    for seed in plan["seeds"]:
        for arm, transitions in [("fresh", 0), ("random", 0)] + [
            ("trained", n) for n in plan["evaluation_at_transitions"][1:]
        ]:
            filename = (
                f"{arm}-{seed}.json"
                if arm != "trained"
                else f"trained-{seed}-{transitions}.json"
            )
            path = OUTPUT / "results" / filename
            if not path.exists():
                continue
            result = read(path)
            row = {
                "seed": seed,
                "arm": arm,
                "transitions": transitions,
                **score(result),
            }
            assert row["missed_clicks"] == 0 and row["learning_hash_unchanged"]
            frozen += result["learning_hash_before"] is not None
            evaluations.append(row)
        path = OUTPUT / f"results/training-{seed}.jsonl"
        if not path.exists():
            continue
        episodes = [json.loads(line) for line in path.read_text().splitlines()]
        rows = [r for ep in episodes for r in ep["result"]["browser_trials"]]
        assert all(r["click_delivered"] for r in rows)
        assert all(
            ep["result"]["original_result"]["learner_failures"] == 0 for ep in episodes
        )
        training.append(
            {
                "seed": seed,
                "episodes": len(episodes),
                "transitions": len(rows),
                "correct": sum(r["correct"] for r in rows),
                "first_16_episode_accuracy": sum(
                    ep["result"]["correct_choices"] for ep in episodes[:16]
                )
                / (min(16, len(episodes)) * 24),
                "last_16_episode_accuracy": sum(
                    ep["result"]["correct_choices"] for ep in episodes[-16:]
                )
                / (min(16, len(episodes)) * 24),
                "clicks": sum(r["action"]["name"] == "click" for r in rows),
                "missed_clicks": 0,
            }
        )
        samples.extend(s for ep in episodes for s in ep["resource_samples"])
    last = OUTPUT / "last-resource-samples.json"
    if last.exists():
        samples.extend(read(last))
    assert all(status["ledger_hashes_unchanged"].values())
    assert all(status["source_hashes_unchanged"].values())
    ledger_checks = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
        for name, expected in plan["ledger_hashes"].items()
    }
    assert all(ledger_checks.values())
    result = {
        "status": status,
        "training": training,
        "evaluations": evaluations,
        "unchanged_frozen_snapshots": frozen,
        "resources": {
            "max_parent_peak_rss_mib": max(s["peak_rss_mib"] for s in samples),
            "max_process_tree_mib": max(s["browser_tree_mib"] for s in samples),
            "min_disk_gib": min(s["free_disk_gib"] for s in samples),
        },
        "closed_ledgers_still_unchanged": ledger_checks,
    }
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = [
        "seed",
        "arm",
        "transitions",
        "correct",
        "decisions",
        "accuracy",
        "left_correct",
        "left_decisions",
        "right_correct",
        "right_decisions",
        "clicks",
        "missed_clicks",
    ]
    with (OUTPUT / "evaluation.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(evaluations)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
