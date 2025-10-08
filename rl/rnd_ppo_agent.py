"""PPO + Random Network Distillation learner."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper
except ImportError:  # pragma: no cover - optional dependency
    PPO = None  # type: ignore[assignment]
    DummyVecEnv = None  # type: ignore[assignment]
    VecEnv = None  # type: ignore[assignment]
    VecEnvWrapper = object  # type: ignore[assignment]

from rl.context import SubgoalEncoder
from rl.features import DomTextHasher
from rl.policy import MaskedCategoricalPolicy, PolicyConfig

logger = logging.getLogger(__name__)


@dataclass
class RNDConfig:
    embedding_dim: int = 1024
    predictor_lr: float = 1e-4
    intrinsic_weight: float = 1.0
    extrinsic_weight: float = 1.0
    normalize_rewards: bool = True


class RNDModule(nn.Module):
    """Target and predictor networks for RND intrinsic reward."""

    def __init__(self, input_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)


class RNDRewardVecWrapper(VecEnvWrapper):
    """VecEnv wrapper injecting intrinsic reward from RND."""

    def __init__(
        self,
        venv: VecEnv,
        rnd_module: RNDModule,
        optimizer: torch.optim.Optimizer,
        feature_extractor: DomTextHasher,
        config: RNDConfig,
    ) -> None:
        super().__init__(venv)
        self.rnd_module = rnd_module
        self.optimizer = optimizer
        self.feature_extractor = feature_extractor
        self.config = config
        self._running_mean = 0.0
        self._running_std = 1.0
        self._count = 1e-4

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        intrinsic = self._compute_intrinsic(obs)
        if self.config.normalize_rewards:
            self._update_running_stats(intrinsic)
            intrinsic = (intrinsic - self._running_mean) / (self._running_std + 1e-8)
        combined = (
            self.config.extrinsic_weight * rewards
            + self.config.intrinsic_weight * intrinsic
        )
        for idx, info in enumerate(infos):
            info["flappy/intrinsic_reward"] = float(intrinsic[idx])
        return obs, combined, dones, infos

    def _compute_intrinsic(self, observations: np.ndarray) -> np.ndarray:
        features = self._encode_observations(observations)
        tensor = torch.from_numpy(features).float()
        with torch.no_grad():
            target = self.rnd_module.target(tensor)
        prediction = self.rnd_module.predictor(tensor)
        loss = F.mse_loss(prediction, target, reduction="mean")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        errors = (prediction.detach() - target).pow(2).mean(dim=1)
        return errors.numpy()

    def _encode_observations(self, observations: np.ndarray) -> np.ndarray:
        feature_list = []
        iterable = observations if isinstance(observations, (list, tuple)) else [observations]
        for obs in iterable:
            if isinstance(obs, dict):
                feature_list.append(self.feature_extractor.encode(obs))
            else:
                feature_list.append(np.asarray(obs, dtype=np.float32))
        return np.stack(feature_list, axis=0)

    def _update_running_stats(self, intrinsic: np.ndarray) -> None:
        batch_mean = intrinsic.mean()
        batch_std = intrinsic.std()
        total_count = self._count + intrinsic.size
        delta = batch_mean - self._running_mean
        new_mean = self._running_mean + delta * intrinsic.size / total_count
        m_a = self._running_std**2 * self._count
        m_b = batch_std**2 * intrinsic.size
        m2 = m_a + m_b + delta**2 * self._count * intrinsic.size / total_count
        self._running_mean = new_mean
        self._running_std = np.sqrt(m2 / total_count)
        self._count = total_count


class PPORNDLearner:
    """High-level interface around PPO + RND intrinsic reward."""

    def __init__(
        self,
        env_fn: Callable[[], Any],
        *,
        rnd_config: Optional[RNDConfig] = None,
        ppo_kwargs: Optional[Dict[str, Any]] = None,
        feature_dim: int = 2048,
        subgoal_dim: int = 256,
        max_actions: int = 32,
    ) -> None:
        if torch is None or PPO is None:  # pragma: no cover - runtime guard
            raise RuntimeError(
                "PPORNDLearner requires PyTorch and Stable-Baselines3. "
                "Install the project requirements."
            )
        self.rnd_config = rnd_config or RNDConfig(embedding_dim=feature_dim)
        self.feature_extractor = DomTextHasher(dim=feature_dim)
        self.vec_env = DummyVecEnv([env_fn])
        rnd_module = RNDModule(feature_dim)
        optimizer = torch.optim.Adam(
            rnd_module.predictor.parameters(), lr=self.rnd_config.predictor_lr
        )
        self.vec_env = RNDRewardVecWrapper(
            self.vec_env, rnd_module, optimizer, self.feature_extractor, self.rnd_config
        )
        default_kwargs = dict(
            policy="MlpPolicy",
            verbose=1,
            tensorboard_log=None,
        )
        if ppo_kwargs:
            default_kwargs.update(ppo_kwargs)
        self.model = PPO(env=self.vec_env, **default_kwargs)
        self.subgoal_encoder = SubgoalEncoder(dim=subgoal_dim)
        self.context_policy = MaskedCategoricalPolicy(
            PolicyConfig(
                state_dim=feature_dim,
                subgoal_dim=subgoal_dim,
                action_dim=max_actions,
                hidden_dim=128,
            )
        )
        self.max_actions = max_actions

    def learn(self, total_timesteps: int, **kwargs: Any) -> None:
        logger.info("Starting PPO+RND training for %d steps", total_timesteps)
        self.model.learn(total_timesteps=total_timesteps, **kwargs)

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = PPO.load(path, env=self.vec_env)

    def predict(self, observation: Any, deterministic: bool = False):
        return self.model.predict(observation, deterministic=deterministic)

    def sample_action(self, features: Any, mask: Optional[np.ndarray] = None) -> int:
        raise NotImplementedError("Use sample_action_with_context instead.")

    def sample_action_with_context(
        self,
        state_vec: np.ndarray,
        subgoal_vec: np.ndarray,
        mask: Optional[np.ndarray],
        action_count: int,
        deterministic: bool = False,
    ) -> int:
        mask_full = np.ones(self.max_actions, dtype=np.float32)
        if action_count < self.max_actions:
            mask_full[action_count:] = 0.0
        if mask is not None:
            mask_full[: len(mask)] *= mask.astype(np.float32)
        if mask_full[:action_count].sum() == 0.0:
            mask_full[:action_count] = 1.0
        action_idx = self.context_policy.sample(
            state_vec=state_vec,
            subgoal_vec=subgoal_vec,
            mask=mask_full,
            deterministic=deterministic,
        )
        if action_idx >= action_count:
            valid_indices = np.nonzero(mask_full[:action_count] > 0)[0]
            if len(valid_indices) == 0:
                action_idx = np.random.randint(0, action_count)
            else:
                action_idx = int(np.random.choice(valid_indices))
        return action_idx
