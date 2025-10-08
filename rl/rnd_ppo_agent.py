"""PPO + Random Network Distillation learner with coach context."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - runtime guard
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from rl.policy import MaskedCategoricalPolicy, PolicyConfig

logger = logging.getLogger(__name__)


@dataclass
class RNDConfig:
    embedding_dim: int = 2048
    predictor_lr: float = 1e-4
    intrinsic_weight: float = 1.0
    extrinsic_weight: float = 1.0
    normalize_rewards: bool = True


@dataclass
class LearnerConfig:
    feature_dim: int = 2048
    subgoal_dim: int = 256
    hidden_dim: int = 128
    max_actions: int = 32
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    rollout_size: int = 2048
    policy_epochs: int = 4
    minibatch_size: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    clip_range: float = 0.2


@dataclass
class SampleOutput:
    action: int
    log_prob: float
    value: float
    mask: np.ndarray
    entropy: float


@dataclass
class Transition:
    state: np.ndarray
    subgoal: np.ndarray
    mask: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float
    intrinsic: float
    done: bool
    next_state: np.ndarray
    next_subgoal: np.ndarray


class RNDModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
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


class ValueNetwork(nn.Module):
    def __init__(self, config: LearnerConfig) -> None:
        super().__init__()
        input_dim = config.feature_dim + config.subgoal_dim
        self.base = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(config.hidden_dim, 1)

    def forward(self, state: torch.Tensor, subgoal: torch.Tensor) -> torch.Tensor:
        latent = self.base(torch.cat([state, subgoal], dim=-1))
        return self.head(latent)


class PPORNDLearner:
    """Custom PPO learner with RND intrinsic reward."""

    def __init__(
        self,
        *,
        learner_config: Optional[LearnerConfig] = None,
        rnd_config: Optional[RNDConfig] = None,
    ) -> None:
        if torch is None or nn is None:
            raise RuntimeError("PPORNDLearner requires PyTorch. Please install torch.")

        self.config = learner_config or LearnerConfig()
        self.rnd_config = rnd_config or RNDConfig(embedding_dim=self.config.feature_dim)
        self.policy = MaskedCategoricalPolicy(
            PolicyConfig(
                state_dim=self.config.feature_dim,
                subgoal_dim=self.config.subgoal_dim,
                hidden_dim=self.config.hidden_dim,
                action_dim=self.config.max_actions,
            )
        )
        self.value_net = ValueNetwork(self.config)
        self.rnd_module = RNDModule(self.config.feature_dim)
        self.rnd_optimizer = torch.optim.Adam(
            self.rnd_module.predictor.parameters(), lr=self.rnd_config.predictor_lr
        )
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=self.config.learning_rate,
        )
        self.rollout: List[Transition] = []
        self.total_steps = 0
        self._running_mean = 0.0
        self._running_var = 1.0
        self._count = 1e-4

    def sample_action_with_context(
        self,
        state_vec: np.ndarray,
        subgoal_vec: np.ndarray,
        mask: Optional[np.ndarray],
        action_count: int,
        deterministic: bool = False,
    ) -> SampleOutput:
        mask_full = np.ones(self.config.max_actions, dtype=np.float32)
        if action_count < self.config.max_actions:
            mask_full[action_count:] = 0.0
        if mask is not None:
            mask_full[: len(mask)] *= mask.astype(np.float32)
        if mask_full[:action_count].sum() == 0.0:
            mask_full[:action_count] = 1.0

        action = self.policy.sample(state_vec, subgoal_vec, mask_full, deterministic)

        state = torch.from_numpy(state_vec.astype(np.float32)).unsqueeze(0)
        subgoal = torch.from_numpy(subgoal_vec.astype(np.float32)).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_full.astype(np.float32)).unsqueeze(0)
        logits = self.policy.forward(state, subgoal, mask_tensor)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        log_prob = float(dist.log_prob(torch.tensor([action])).item())
        value = float(self.value_net(state, subgoal).item())
        entropy = float(dist.entropy().mean().item())

        return SampleOutput(
            action=action,
            log_prob=log_prob,
            value=value,
            mask=mask_full,
            entropy=entropy,
        )

    def compute_intrinsic(self, next_state: np.ndarray) -> float:
        state_tensor = torch.from_numpy(next_state.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            target = self.rnd_module.target(state_tensor)
        prediction = self.rnd_module.predictor(state_tensor)
        loss = F.mse_loss(prediction, target.detach())
        self.rnd_optimizer.zero_grad()
        loss.backward()
        self.rnd_optimizer.step()
        error = (prediction.detach() - target.detach()).pow(2).mean().item()
        if not self.rnd_config.normalize_rewards:
            return error
        self._update_running_stats(error)
        return (error - self._running_mean) / (np.sqrt(self._running_var) + 1e-8)

    def _update_running_stats(self, value: float) -> None:
        self._count += 1
        delta = value - self._running_mean
        self._running_mean += delta / self._count
        delta2 = value - self._running_mean
        self._running_var += delta * delta2

    def observe_transition(
        self,
        *,
        state: np.ndarray,
        subgoal: np.ndarray,
        sample: SampleOutput,
        reward: float,
        intrinsic: float,
        done: bool,
        next_state: np.ndarray,
        next_subgoal: np.ndarray,
    ) -> None:
        transition = Transition(
            state=state.astype(np.float32),
            subgoal=subgoal.astype(np.float32),
            mask=sample.mask.astype(np.float32),
            action=sample.action,
            log_prob=sample.log_prob,
            value=sample.value,
            reward=float(reward),
            intrinsic=float(intrinsic),
            done=done,
            next_state=next_state.astype(np.float32),
            next_subgoal=next_subgoal.astype(np.float32),
        )
        self.rollout.append(transition)
        self.total_steps += 1
        if len(self.rollout) >= self.config.rollout_size:
            self._update_policy()
            self.rollout.clear()

    def _update_policy(self) -> None:
        if not self.rollout:
            return
        batch = self.rollout
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        intrinsic = np.array([t.intrinsic for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        values = np.array([t.value for t in batch], dtype=np.float32)

        combined_reward = (
            self.rnd_config.extrinsic_weight * rewards
            + self.rnd_config.intrinsic_weight * intrinsic
        )
        next_values = []
        for t in batch:
            state_tensor = torch.from_numpy(t.next_state).unsqueeze(0)
            subgoal_tensor = torch.from_numpy(t.next_subgoal).unsqueeze(0)
            with torch.no_grad():
                nv = self.value_net(state_tensor, subgoal_tensor).item()
            next_values.append(nv)
        next_values = np.array(next_values, dtype=np.float32)

        returns = np.zeros_like(values)
        advantages = np.zeros_like(values)
        gae = 0.0
        for step in reversed(range(len(batch))):
            delta = (
                combined_reward[step]
                + self.config.gamma * next_values[step] * (1.0 - dones[step])
                - values[step]
            )
            gae = delta + self.config.gamma * self.config.gae_lambda * (1.0 - dones[step]) * gae
            advantages[step] = gae
            returns[step] = gae + values[step]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.from_numpy(np.stack([t.state for t in batch])).float()
        subgoals = torch.from_numpy(np.stack([t.subgoal for t in batch])).float()
        masks = torch.from_numpy(np.stack([t.mask for t in batch])).float()
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64)
        old_log_probs = torch.tensor([t.log_prob for t in batch], dtype=torch.float32)
        returns_t = torch.from_numpy(returns).float()
        adv_t = torch.from_numpy(advantages).float()

        dataset_size = len(batch)
        indices = np.arange(dataset_size)
        for _ in range(self.config.policy_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_idx = indices[start:end]
                mb_states = states[mb_idx]
                mb_subgoals = subgoals[mb_idx]
                mb_masks = masks[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log = old_log_probs[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_adv = adv_t[mb_idx]

                logits = self.policy.forward(mb_states, mb_subgoals, mb_masks)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                log_probs = dist.log_prob(mb_actions)
                ratio = torch.exp(log_probs - mb_old_log)
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                values_pred = self.value_net(mb_states, mb_subgoals).squeeze(-1)
                value_loss = F.mse_loss(values_pred, mb_returns)
                entropy = dist.entropy().mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()), 1.0
                )
                self.optimizer.step()

        logger.debug("PPO update completed on %d samples", dataset_size)

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "value": self.value_net.state_dict(),
                "rnd": self.rnd_module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "rnd_opt": self.rnd_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy"])
        self.value_net.load_state_dict(checkpoint["value"])
        self.rnd_module.load_state_dict(checkpoint["rnd"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.rnd_optimizer.load_state_dict(checkpoint["rnd_opt"])

    @property
    def max_actions(self) -> int:
        return self.config.max_actions
