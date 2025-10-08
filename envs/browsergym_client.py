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
import time
from typing import Any, Dict, Optional, Tuple

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - handled during runtime
    gym = None

try:
    import browsergym
    from browsergym.core.action import BrowserAction
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
            "observation_mode": self.observation_mode,
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
        self, action: PlannerAction
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Convert planner action to BrowserGym action and step environment."""
        env_action = self._planner_action_to_browser_action(action)
        start = time.perf_counter()
        obs, reward, terminated, truncated, info = self.env.step(env_action)  # type: ignore[arg-type]
        latency = time.perf_counter() - start
        info.setdefault("flappy/latency_sec", latency)
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
        return {
            "dom_text": dom_text or "",
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
        """Map planner-friendly actions into BrowserGym actions."""
        name = action.name
        payload: Dict[str, Any]
        if name == "click":
            payload = {"action_type": "click", "selector": action.selector}
        elif name == "type":
            payload = {
                "action_type": "type",
                "selector": action.selector,
                "text": action.text or "",
            }
        elif name == "press":
            payload = {"action_type": "press", "key": action.key}
        elif name == "scroll":
            payload = {"action_type": "scroll", "direction": action.direction}
        elif name == "wait":
            payload = {"action_type": "wait", "milliseconds": action.wait_ms or 0}
        elif name == "back":
            payload = {"action_type": "history", "direction": "back"}
        elif name == "save_note":
            payload = {"action_type": "noop", "note": action.text}
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported planner action: {name}")

        payload.setdefault("timeout", self.navigation_timeout)
        logger.debug("Planner action %s mapped to %s", name, json.dumps(payload))
        return payload  # type: ignore[return-value]

    def last_observation(self) -> Optional[Dict[str, Any]]:
        """Return the last raw observation."""
        return self._last_observation


def make_planner_action(name: str, **kwargs: Any) -> PlannerAction:
    """Helper to construct planner actions with validation."""
    return PlannerAction(name=name, **kwargs)
