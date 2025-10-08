"""Pure RL baseline agent (PPO)."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:  # pragma: no cover - optional dependency
    PPO = None  # type: ignore[assignment]
    DummyVecEnv = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class PureRLAgent:
    """Wrap Stable-Baselines PPO for extrinsic-only training."""

    def __init__(
        self,
        env_fn: Callable[[], Any],
        *,
        ppo_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if PPO is None:
            raise RuntimeError(
                "Stable-Baselines3 not installed. Please install requirements."
            )
        self.vec_env = DummyVecEnv([env_fn])
        kwargs = {"policy": "MlpPolicy", "verbose": 1}
        if ppo_kwargs:
            kwargs.update(ppo_kwargs)
        self.model = PPO(env=self.vec_env, **kwargs)

    def learn(self, total_timesteps: int, **kwargs: Any) -> None:
        logger.info("Training pure PPO agent for %d steps", total_timesteps)
        self.model.learn(total_timesteps=total_timesteps, **kwargs)

    def run_episode(self, task_id: str) -> Dict[str, float]:
        env = self.vec_env.envs[0]
        obs, info = env.reset(return_info=True)
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        return {"success": float(info.get("success", False)), "reward": total_reward, "steps": steps}

    def evaluate(self, env_fn: Callable[[], Any], episodes: int = 10) -> Dict[str, float]:
        env = env_fn()
        returns = []
        for _ in range(episodes):
            obs, info = env.reset(return_info=True)
            done = False
            total_reward = 0.0
            while not done:
                action, _ = self.model.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
            returns.append(total_reward)
        env.close()
        return {"mean_return": sum(returns) / len(returns)}
