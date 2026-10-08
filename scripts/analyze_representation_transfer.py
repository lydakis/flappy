"""Audit saved transfer calibration; never train, query a teacher or use a network."""

import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs/representation-transfer"


def read(path: Path) -> dict | list:
    return json.loads(path.read_text())


def main() -> None:
    plan = read(OUTPUT / "plan.json")
    calibration = read(OUTPUT / "calibration-summary.json")
    assert calibration["complete"]
    assert calibration["passed"] is False
    assert not (OUTPUT / "main").exists()
    runs = calibration["runs"]
    assert {r["arm"] for r in runs} == set(plan["domains"])
    assert len({r["initial_weight_hash"] for r in runs}) == 1
    assert all(r["parameter_count"] == plan["parameter_count"] for r in runs)
    assert all(r["initial_weight_hash"] != r["final_weight_hash"] for r in runs)
    for name, expected in read(OUTPUT / "source-data-hashes.json").items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    for name, expected in plan["prior_hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    tuples = defaultdict(set)
    for path in OUTPUT.glob("latent-*.json"):
        split = path.name.split("-")[1]
        cases = read(path)
        keys = {(c["position_milli"], c["velocity_milli"], c["steps"]) for c in cases}
        assert len(keys) == len(cases)
        tuples[split].update(keys)
        assert sum(c["label"] for c in cases) * 2 == len(cases)
        assert sum(
            (c["velocity_milli"] > 0) == bool(c["label"]) for c in cases
        ) * 2 == len(cases)
        assert sum(
            (c["position_milli"] > 0) == bool(c["label"]) for c in cases
        ) * 2 == len(cases)
    for first, left in tuples.items():
        for second, right in tuples.items():
            if first != second:
                assert not left.intersection(right)
    probe = read(OUTPUT / "calibration/evaluation-inputs.json")
    group_labels = defaultdict(list)
    for e in probe:
        group_labels[(e["domain"], e["group"], e["template"])].append(e["label"])
        assert e["latent_id"] not in e["input"]
    assert all(sum(labels) * 2 == len(labels) for labels in group_labels.values())
    frozen = 0
    rows = []
    checkpoint_dir = ROOT / "checkpoints/representation-transfer"
    checkpoint_dir.mkdir(exist_ok=True)
    for run in runs:
        assert run["teacher_labels"] == 12288 and run["updates"] == 384
        assert run["domain_exposure"] == {
            d: 12288 if d == run["arm"] else 0 for d in plan["domains"]
        }
        directory = OUTPUT / "calibration" / f"{run['arm']}-{run['seed']}"
        batches = [
            json.loads(line)
            for line in (directory / "batches.jsonl").read_text().splitlines()
        ]
        assert len(batches) == 384
        latent_ids = [
            identifier for batch in batches for identifier in batch["latent_ids"]
        ]
        assert len(set(latent_ids)) == len(latent_ids) == 12288
        for i, batch in enumerate(batches):
            assert batch["examples_seen"] == batch["teacher_labels"] == (i + 1) * 32
            assert batch["updates"] == i + 1
            assert len(batch["labels"]) == 32
        for e in run["evaluations"]:
            assert e["state_hash_before"] == e["state_hash_after"]
            frozen += 1
            for m in e["metrics"]:
                rows.append(
                    {
                        "source_training_domain": run["arm"],
                        "seed": run["seed"],
                        "total_training_examples": e["examples_seen"],
                        "total_teacher_labels": e["labels_used"],
                        "target_training_exposure": (
                            e["examples_seen"] if m["domain"] == run["arm"] else 0
                        ),
                        **m,
                    }
                )
        # Keep final snapshots in the repository's standard checkpoint location.
        source = directory / "checkpoint.pt"
        target = checkpoint_dir / f"calibration-{run['arm']}-{run['seed']}.pt"
        if source.exists():
            assert not target.exists()
            shutil.move(source, target)
        assert target.is_file()
    with (OUTPUT / "calibration-matrix-and-frontier.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "stopped_at_predeclared_calibration_gate",
        "main_comparison_run": False,
        "mixed_training_run": False,
        "main_question_answered": False,
        "main_final_probe_evaluated": False,
        "calibration_runs": len(runs),
        "calibration_examples_and_teacher_labels": sum(
            r["teacher_labels"] for r in runs
        ),
        "calibration_updates": sum(r["updates"] for r in runs),
        "source_gate_results": [
            {
                "domain": r["arm"],
                "seed": r["seed"],
                "source_seen_accuracy": next(
                    m["accuracy"]
                    for m in r["evaluations"][-1]["metrics"]
                    if m["domain"] == r["arm"] and m["group"] == "seen_template"
                ),
                "source_heldout_accuracy": next(
                    m["accuracy"]
                    for m in r["evaluations"][-1]["metrics"]
                    if m["domain"] == r["arm"] and m["group"] == "heldout_template"
                ),
            }
            for r in runs
        ],
        "matrix_and_frontier": rows,
        "checks": {
            "passed": True,
            "frozen_evaluations": frozen,
            "same_initial_weights_and_capacity": True,
            "all_models_changed": True,
            "exact_label_update_accounting": True,
            "no_repeated_training_latents_within_learner": True,
            "four_latent_split_universes_disjoint": True,
            "probe_labels_balanced_per_template": True,
            "position_sign_and_velocity_sign_baselines_exactly_50_percent": True,
            "source_data_prior_evidence_hashes_unchanged": True,
        },
        "resources": calibration["resources"],
        "calibration_wall_seconds": calibration["wall_seconds"],
        "new_paid_calls": 0,
    }
    (OUTPUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "matrix_and_frontier"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
