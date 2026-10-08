#!/usr/bin/env python3
"""Replay-only focus/actionability diagnosis. No teacher or learner is constructed."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright

from envs.browsergym_client import BrowserGymEnvWrapper, PlannerAction
from scripts.run_interactive_teacher import LocalBrowserCue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the existing diagnostic result")
    original = BrowserGymEnvWrapper._planner_action_to_browser_action
    source = ROOT / "logs/interactive-teacher/results/training-no_teacher-7.json"
    actions = [
        PlannerAction(**r["action"])
        for r in json.loads(source.read_text())["browser_trials"]
    ]
    results = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        for mode in ["baseline", "bring_to_front", "normal_actionability"]:
            page = browser.new_page(viewport={"width": 600, "height": 400})
            env = LocalBrowserCue(page)
            records = []

            def translated(self, action, selected_mode=mode):
                source = original(self, action)
                # Recreate the paid run's forced-click baseline after the fix.
                normal = source.replace("timeout=5000, force=True", "timeout=5000")
                if selected_mode != "normal_actionability" and action.name == "click":
                    return normal.replace(
                        "elem.click(timeout=5000)",
                        "elem.click(timeout=5000, force=True)",
                    )
                return normal

            BrowserGymEnvWrapper._planner_action_to_browser_action = translated
            for repeat in range(5):
                env.reset()
                for action in actions:
                    before = page.evaluate(
                        "({focused: document.hasFocus(), active: document.activeElement?.tagName, id: document.activeElement?.id})"
                    )
                    if mode == "bring_to_front":
                        page.bring_to_front()
                    env.step(action)
                    row = dict(env.rows[-1])
                    row["focus_before"] = before
                    row["repeat"] = repeat
                    row["missed"] = action.name == "click" and row[
                        "actual_choice"
                    ] != action.selector.lstrip("#")
                    records.append(row)
            result = {
                "mode": mode,
                "clicks": sum(r["action"]["name"] == "click" for r in records),
                "missed_clicks": sum(r["missed"] for r in records),
                "records": records,
            }
            results.append(result)
            print(
                json.dumps({k: v for k, v in result.items() if k != "records"}),
                flush=True,
            )
            page.close()
        browser.close()
    BrowserGymEnvWrapper._planner_action_to_browser_action = original
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
