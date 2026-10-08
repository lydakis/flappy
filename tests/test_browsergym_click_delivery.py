"""Opt-in local Chromium regression; never uses a teacher, credential or learner."""

import os

import pytest

from envs.browsergym_client import make_planner_action


@pytest.mark.skipif(
    os.environ.get("RUN_LOCAL_BROWSER_TESTS") != "1",
    reason="Requires the existing isolated Chromium installation",
)
def test_click_delivered_after_navigation_and_dom_replacement():
    from playwright.sync_api import sync_playwright

    from scripts.run_interactive_teacher import LocalBrowserCue

    # Prefix of a failing real trajectory. Forced clicks lost the final click
    # even though the button was visible and Playwright returned successfully.
    sequence = [
        ("click", {"selector": "#left"}),
        ("click", {"selector": "#right"}),
        ("click", {"selector": "#right"}),
        ("press", {"key": "Tab"}),
        ("scroll", {"direction": "up"}),
        ("scroll", {"direction": "up"}),
        ("scroll", {"direction": "up"}),
        ("wait", {"wait_ms": 200}),
        ("scroll", {"direction": "up"}),
        ("click", {"selector": "#left"}),
        ("press", {"key": "Tab"}),
        ("press", {"key": "Tab"}),
        ("press", {"key": "Enter"}),
        ("click", {"selector": "#left"}),
    ]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 600, "height": 400})
            env = LocalBrowserCue(page)
            failures = []
            for repeat in range(5):
                env.reset()
                for index, (name, kwargs) in enumerate(sequence):
                    env.step(make_planner_action(name, **kwargs))
                    if name == "click":
                        expected = kwargs["selector"].lstrip("#")
                        actual = env.rows[-1]["actual_choice"]
                        if actual != expected:
                            failures.append((repeat, index, expected, actual))
            assert not failures, f"Undelivered clicks: {failures}"
        finally:
            browser.close()
