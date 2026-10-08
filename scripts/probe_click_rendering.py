#!/usr/bin/env python3
"""Local, no-learning differential probe for forced-click rendering races."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright

from envs.browsergym_client import BrowserGymEnvWrapper, make_planner_action
from scripts.run_interactive_teacher import LocalBrowserCue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the existing diagnostic result")
    results = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 600, "height": 400})
        env = LocalBrowserCue(page)
        for mode in [
            "original_force",
            "post_click_wait",
            "actionability_wait",
            "pre_click_two_frames",
        ]:
            failed = 0
            for trial in range(80):
                env.t = trial % 24
                env._render()
                source = BrowserGymEnvWrapper._planner_action_to_browser_action(
                    env, make_planner_action("click", selector="#left")
                )
                source = source.replace("timeout=5000, force=True", "timeout=5000")
                source = source.replace(
                    "elem.click(timeout=5000)", "elem.click(timeout=5000, force=True)"
                )
                if mode == "actionability_wait":
                    source = source.replace("force=True", "force=False")
                elif mode == "pre_click_two_frames":
                    page.evaluate(
                        "() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"
                    )
                exec(source, {"page": page})  # noqa: S102 -- fixed local action catalog
                if mode == "post_click_wait":
                    page.wait_for_timeout(50)
                failed += page.evaluate("window.lastChoice") != "left"
            results.append({"mode": mode, "attempts": 80, "missed_clicks": failed})
            print(json.dumps(results[-1]), flush=True)
        browser.close()
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
