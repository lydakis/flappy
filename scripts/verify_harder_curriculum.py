"""Read-only checks of saved harder-menu traces and preserved earlier evidence."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs/harder-curriculum"


def read(path):
    return json.loads(path.read_text())


def main():
    plan = read(OUTPUT / "plan.json")
    runs = read(OUTPUT / "summaries.json")
    calibration = read(OUTPUT / "calibration-summary.json")
    assert len(calibration["calibration_runs"]) <= 6
    assert calibration["selected"] == "quadratic"
    assert [r["seed"] for r in calibration["calibration_runs"]] == [101, 103]
    assert all(r["qualification"]["passed"] for r in calibration["calibration_runs"])
    examples, matching_observations = {}, 0
    for run in runs:
        directory = OUTPUT / "runs" / f"{run['arm']}-{run['seed']}"
        cases = [
            json.loads(s) for s in (directory / "cases.jsonl").read_text().splitlines()
        ]
        for row in cases:
            example = row["example"]
            if example["id"] in examples:
                assert example == examples[example["id"]]
                matching_observations += 1
            examples[example["id"]] = example
        assert run["initial_weight_hash"] != run["final_weight_hash"]
        assert len(cases) == 6400
        assert sum(c["asked"] for c in cases) == run["optimizer_steps"]
    for seed in plan["seeds"]:
        warmups = [
            next(
                c
                for c in r["checkpoints"]
                if c["kind"] == "environment" and c["cases_seen"] == 192
            )
            for r in runs
            if r["seed"] == seed
        ]
        assert warmups[0] == warmups[1] == warmups[2]
    for name, expected in read(OUTPUT / "source-hashes.json").items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    for name, expected in plan["ledger_hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
        assert read(ROOT / name)["closed"]
    result = {
        "passed": True,
        "matching_case_observations_across_arms": matching_observations,
        "identical_post_warmup_states_per_seed": True,
        "all_nine_models_changed": True,
        "query_update_accounting_exact": True,
        "source_and_probe_hashes_unchanged": True,
        "both_prior_api_ledgers_closed_and_unchanged": True,
        "calibration_cases": len(calibration["calibration_runs"]) * 4800,
        "calibration_labels": sum(
            r["total_oracle_labels"] for r in calibration["calibration_runs"]
        ),
    }
    (OUTPUT / "trace-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
