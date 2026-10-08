#!/usr/bin/env python3
"""No API or learning: check original browser click delivery across trial renders."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright

from envs.browsergym_client import PlannerAction
from scripts.run_interactive_teacher import LocalBrowserCue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the existing diagnostic result")
    source = ROOT / "logs/interactive-teacher/results/training-no_teacher-7.json"
    actions = [
        PlannerAction(**r["action"])
        for r in json.loads(source.read_text())["browser_trials"]
    ]
    records = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 600, "height": 400})
        env = LocalBrowserCue(page)
        for repeat in range(5):
            env.reset()
            for action in actions:
                env.step(action)
                row = dict(env.rows[-1])
                row["repeat"] = repeat
                row["click_delivered"] = action.name != "click" or row[
                    "actual_choice"
                ] == action.selector.lstrip("#")
                records.append(row)
        browser.close()
    result = {
        "actions": len(records),
        "clicks": sum(r["action"]["name"] == "click" for r in records),
        "missed_clicks": sum(not r["click_delivered"] for r in records),
        "records": records,
    }
    output = args.output
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))
    if result["missed_clicks"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
