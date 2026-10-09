"""GRPO pieces: group-normalized advantages and the per-token policy loss.

With one on-policy gradient step per sampled group the PPO ratio is exactly 1, so
clipping never binds and the objective reduces to advantage-weighted
log-likelihood plus a KL penalty toward the frozen base model (adapter off).
"""

from __future__ import annotations

import math


def group_advantages(rewards: list[float], eps: float = 1e-6) -> list[float] | None:
    """(r - mean) / std within one group; None when every reward is equal."""
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    if std < eps:
        return None
    return [(r - mean) / (std + eps) for r in rewards]


def grpo_loss(logp, ref_logp, advantages, mask, kl_coef: float):
    """Per-sequence token-mean of ``-A * logp + kl_coef * KL``, averaged over the group.

    ``logp``/``ref_logp``/``mask`` are [group, tokens] tensors over completion
    tokens; ``advantages`` is [group]. KL uses the non-negative k3 estimator.
    """
    log_ratio = ref_logp - logp
    kl = log_ratio.exp() - log_ratio - 1
    per_token = -advantages[:, None] * logp + kl_coef * kl
    per_seq = (per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return per_seq.mean(), (kl * mask).sum() / mask.sum().clamp(min=1)
