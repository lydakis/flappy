"""BrowserGym environment wrapper for FLAPPY.

This module standardises interaction with MiniWoB++ tasks exposed through
BrowserGym. It converts between the high-level tool actions used by the
planner and the low-level environment API, while also handling DOM
observations and timeouts.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - handled during runtime
    gym = None

try:
    import browsergym
    import browsergym.miniwob  # noqa: F401
except ImportError:  # pragma: no cover - handled during runtime
    browsergym = None

BrowserAction = Any  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

DEFAULT_ACTION_TIMEOUT = 5.0


@dataclasses.dataclass
class PlannerAction:
    """Represents a high-level tool call issued by the planner."""

    name: str
    selector: Optional[str] = None
    text: Optional[str] = None
    key: Optional[str] = None
    direction: Optional[str] = None
    wait_ms: Optional[int] = None


class BrowserGymEnvWrapper(gym.Env if gym else object):
    """Lightweight wrapper around BrowserGym MiniWoB++ environments."""

    def __init__(
        self,
        env_id: str,
        headless: bool = True,
        max_episode_steps: Optional[int] = None,
        navigation_timeout: float = DEFAULT_ACTION_TIMEOUT,
        observation_mode: str = "dom_text",
    ) -> None:
        if gym is None or browsergym is None:  # pragma: no cover - runtime guard
            raise RuntimeError(
                "BrowserGymEnvWrapper requires gymnasium and browsergym. "
                "Please install the project dependencies."
            )

        miniwob_url = os.environ.get("MINIWOB_URL")
        if miniwob_url and not miniwob_url.endswith("/"):
            os.environ["MINIWOB_URL"] = miniwob_url + "/"

        self.env_id = env_id
        self.headless = headless
        self.max_episode_steps = max_episode_steps
        self.navigation_timeout = navigation_timeout
        self.observation_mode = observation_mode
        self.env = self._make_env()
        self.observation_space = getattr(self.env, "observation_space", None)
        self.action_space = getattr(self.env, "action_space", None)
        self._step_count = 0
        self._last_observation: Optional[Dict[str, Any]] = None

    def _make_env(self) -> gym.Env:
        """Instantiate the BrowserGym environment with supplied options."""
        logger.info("Creating BrowserGym environment %s", self.env_id)
        kwargs: Dict[str, Any] = {
            "headless": self.headless,
            "action_mapping": None,
        }
        if self.max_episode_steps is not None:
            kwargs["max_episode_steps"] = self.max_episode_steps
        env = gym.make(self.env_id, **kwargs)
        return env

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        return_info: bool = False,
    ) -> Any:
        """Reset the underlying environment and clear counters."""
        self._step_count = 0
        result = self.env.reset(seed=seed, options=options)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        self._last_observation = obs
        return (obs, info) if return_info else obs

    def step(
        self, action: BrowserAction | PlannerAction
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Convert planner action to BrowserGym action and step environment."""
        if isinstance(action, PlannerAction):
            env_action = self._planner_action_to_browser_action(action)
        else:
            env_action = action
        start = time.perf_counter()
        obs, reward, terminated, truncated, info = self.env.step(env_action)  # type: ignore[arg-type]
        latency = time.perf_counter() - start
        info.setdefault("flappy/latency_sec", latency)
        task_info = info.get("task_info", {}) or {}
        episode_reward = float(task_info.get("RAW_REWARD_GLOBAL", reward))
        success = bool(task_info.get("DONE_GLOBAL") and episode_reward > 0.0)
        info.setdefault("episode_reward", episode_reward)
        info.setdefault("success", success)
        self._step_count += 1
        self._last_observation = obs
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        self.env.close()

    def encode_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Return a dictionary friendly for LLM prompts and RL features."""
        dom_text = obs.get("dom_text") or obs.get("text")
        if isinstance(dom_text, bytes):
            dom_text = dom_text.decode("utf-8")
        goal = obs.get("goal") or obs.get("instruction") or ""
        if not dom_text:
            dom_object = obs.get("dom_object")
            if dom_object:
                try:
                    dom_text = json.dumps(dom_object)
                except TypeError:
                    dom_text = str(dom_object)
        combined = dom_text or ""
        if goal:
            combined = f"Goal: {goal}\n{combined}"
        return {
            "dom_text": combined,
            "url": obs.get("url", ""),
            "timestamp": time.time(),
        }

    @staticmethod
    def tool_schemas() -> Dict[str, Dict[str, Any]]:
        """JSON Schema definitions for planner tool calls."""
        return {
            "click": {
                "type": "object",
                "properties": {"selector": {"type": "string"}},
                "required": ["selector"],
            },
            "type": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["selector", "text"],
            },
            "press": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
            "scroll": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["up", "down"]}
                },
                "required": ["direction"],
            },
            "wait": {
                "type": "object",
                "properties": {"ms": {"type": "integer", "minimum": 0}},
                "required": ["ms"],
            },
            "back": {"type": "object", "properties": {}, "required": []},
            "save_note": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        }

    def _planner_action_to_browser_action(self, action: PlannerAction) -> BrowserAction:
        """Translate planner actions into executable BrowserGym python snippets."""
        timeout_ms = int((self.navigation_timeout or DEFAULT_ACTION_TIMEOUT) * 1000)
        name = action.name
        if name == "click":
            if not action.selector:
                raise ValueError("click action requires a selector")
            selector = json.dumps(action.selector)
            return (
                f"elem = page.wait_for_selector({selector}, state='visible', timeout={timeout_ms})\n"
                "if elem is None:\n"
                f"    raise ValueError('Selector not found: {action.selector}')\n"
                "elem.click(timeout=5000, force=True)\n"
            )
        if name == "type":
            if not action.selector:
                raise ValueError("type action requires a selector")
            selector = json.dumps(action.selector)
            text = json.dumps(action.text or "")
            return (
                f"elem = page.wait_for_selector({selector}, state='visible', timeout={timeout_ms})\n"
                "if elem is None:\n"
                f"    raise ValueError('Selector not found: {action.selector}')\n"
                "elem.click(timeout=5000)\n"
                f"elem.fill({text}, timeout=5000)\n"
            )
        if name == "press":
            if action.selector:
                selector = json.dumps(action.selector)
                focus_snippet = (
                    f"elem = page.wait_for_selector({selector}, timeout={timeout_ms})\n"
                    "if elem is None:\n"
                    f"    raise ValueError('Selector not found: {action.selector}')\n"
                    "elem.focus()\n"
                )
            else:
                focus_snippet = ""
            key = json.dumps(action.key or "")
            if not action.key:
                raise ValueError("press action requires a key")
            return focus_snippet + f"page.keyboard.press({key}, timeout=5000)\n"
        if name == "scroll":
            direction = (action.direction or "down").lower()
            delta = 400 if direction == "down" else -400
            return f"page.mouse.wheel(0, {delta})\n"
        if name == "wait":
            wait_ms = int(action.wait_ms or 0)
            return f"page.wait_for_timeout({max(wait_ms, 0)})\n"
        if name == "back":
            return "page.go_back(wait_until='load')\n"
        if name == "save_note":
            return "pass\n"
        raise ValueError(f"Unsupported planner action: {name}")

    def last_observation(self) -> Optional[Dict[str, Any]]:
        """Return the last raw observation."""
        return self._last_observation


def make_planner_action(name: str, **kwargs: Any) -> PlannerAction:
    """Helper to construct planner actions with validation."""
    return PlannerAction(name=name, **kwargs)
