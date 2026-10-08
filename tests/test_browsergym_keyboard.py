from types import SimpleNamespace
from unittest.mock import create_autospec

from playwright.sync_api import Keyboard

from envs.browsergym_client import BrowserGymEnvWrapper, make_planner_action


def test_generated_keyboard_press_matches_playwright_signature():
    wrapper = BrowserGymEnvWrapper.__new__(BrowserGymEnvWrapper)
    wrapper.navigation_timeout = 5.0
    keyboard = create_autospec(Keyboard, instance=True)
    snippet = wrapper._planner_action_to_browser_action(
        make_planner_action("press", key="Tab")
    )
    # Execute our generated, fixed-fixture action against the real API signature.
    exec(snippet, {"page": SimpleNamespace(keyboard=keyboard)})  # noqa: S102
    keyboard.press.assert_called_once_with("Tab")
