"""Tests for BrowserGym environment wrapper behavior."""

from envs.browsergym_client import (
    BrowserGymEnvWrapper,
    DEFAULT_ACTION_TIMEOUT,
)


class _FakeEnv:
    """Minimal stand-in for a Gym-style environment."""

    def __init__(self) -> None:
        self.last_action = None

    def step(self, action):  # type: ignore[no-untyped-def]
        self.last_action = action
        observation = {"value": 42}
        reward = 1.0
        terminated = False
        truncated = False
        info: dict = {}
        return observation, reward, terminated, truncated, info


def test_step_accepts_native_action() -> None:
    """Numeric actions should bypass planner conversion."""

    wrapper = object.__new__(BrowserGymEnvWrapper)
    wrapper.env = _FakeEnv()  # type: ignore[attr-defined]
    wrapper.navigation_timeout = DEFAULT_ACTION_TIMEOUT
    wrapper._step_count = 0
    wrapper._last_observation = None

    obs, reward, terminated, truncated, info = wrapper.step(7)  # type: ignore[arg-type]

    assert wrapper.env.last_action == 7  # type: ignore[attr-defined]
    assert obs == {"value": 42}
    assert reward == 1.0
    assert not terminated
    assert not truncated
    assert "flappy/latency_sec" in info
