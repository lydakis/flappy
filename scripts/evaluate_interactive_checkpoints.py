#!/usr/bin/env python3
"""Frozen teacher-free reevaluation after the browser click fix; never trains."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from playwright.sync_api import sync_playwright

from scripts.run_interactive_teacher import (
    OUTPUT,
    SEEDS,
    LocalBrowserCue,
    NoTeacher,
    build,
    evaluate,
    save,
)
from scripts.run_local_learning import resources


def main() -> None:
    destination = OUTPUT / "post-fix-evaluation"
    destination.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    summaries = []
    started = time.monotonic()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 600, "height": 400})
            env = LocalBrowserCue(page)
            for seed in SEEDS:
                for arm in ["fresh", "random", "no_teacher", "live_teacher"]:
                    agent = build(env, seed, NoTeacher(), arm, arm == "random")
                    checkpoint = None
                    if arm in {"no_teacher", "live_teacher"}:
                        checkpoint = (
                            ROOT / f"checkpoints/interactive-teacher/{arm}-{seed}.pt"
                        )
                        checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                        agent.learner.load(str(checkpoint))
                    result = evaluate(agent)
                    rows = [
                        r for ep in result["episodes"] for r in ep["browser_trials"]
                    ]
                    clicks = [r for r in rows if r["action"]["name"] == "click"]
                    missed = sum(
                        r["actual_choice"] != r["action"]["selector"].lstrip("#")
                        for r in clicks
                    )
                    if missed:
                        raise RuntimeError("Click delivery still unreliable")
                    if checkpoint:
                        assert (
                            hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                            == checksum
                        )
                    save(destination / f"{arm}-{seed}.json", result)
                    summary = {
                        "seed": seed,
                        "arm": arm,
                        "correct": sum(r["correct"] for r in rows),
                        "decisions": len(rows),
                        "choice_accuracy": result["choice_accuracy"],
                        "clicks": len(clicks),
                        "missed_clicks": missed,
                        "checkpoint_sha256": checksum if checkpoint else None,
                        "learning_state_unchanged": result["learning_hash_before"]
                        == result["learning_hash_after"],
                        "resources": resources(),
                        "wall_seconds": time.monotonic() - started,
                    }
                    summaries.append(summary)
                    save(destination / "summary.json", summaries)
                    print(json.dumps(summary), flush=True)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
