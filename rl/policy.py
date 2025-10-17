"""Coach-conditioned policy utilities for masked action selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PolicyConfig:
    state_dim: int
    subgoal_dim: int
    hidden_dim: int = 128
    action_dim: int = 4
    use_mask_head: bool = False


class MaskedCategoricalPolicy(nn.Module):
    """Lightweight policy mapping (state, subgoal) -> action logits and optional mask."""

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = config.state_dim + config.subgoal_dim
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.mask_head = (
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.action_dim),
            )
            if config.use_mask_head
            else None
        )

    def _compute_latent(
        self,
        state_vec: torch.Tensor,
        subgoal_vec: torch.Tensor,
    ) -> torch.Tensor:
        joined = torch.cat([state_vec, subgoal_vec], dim=-1)
        return self.feature_net(joined)

    def forward(
        self,
        state_vec: torch.Tensor,
        subgoal_vec: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent = self._compute_latent(state_vec, subgoal_vec)
        logits = self.action_head(latent)
        if mask is not None:
            logits = logits + torch.log(mask.clamp(min=1e-6))
        return logits

    @torch.no_grad()
    def sample(
        self,
        state_vec: np.ndarray,
        subgoal_vec: np.ndarray,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> int:
        state = torch.from_numpy(state_vec.astype(np.float32)).unsqueeze(0)
        subgoal = torch.from_numpy(subgoal_vec.astype(np.float32)).unsqueeze(0)
        mask_tensor = (
            torch.from_numpy(mask.astype(np.float32)).unsqueeze(0) if mask is not None else None
        )
        logits = self.forward(state, subgoal, mask_tensor)
        if deterministic:
            return int(torch.argmax(logits, dim=-1).item())
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        return int(dist.sample().item())

    @torch.no_grad()
    def predict_mask(
        self,
        state_vec: np.ndarray,
        subgoal_vec: np.ndarray,
    ) -> Optional[np.ndarray]:
        if self.mask_head is None:
            return None
        state = torch.from_numpy(state_vec.astype(np.float32)).unsqueeze(0)
        subgoal = torch.from_numpy(subgoal_vec.astype(np.float32)).unsqueeze(0)
        latent = self._compute_latent(state, subgoal)
        logits = self.mask_head(latent)
        mask = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        return mask.astype(np.float32, copy=False)
