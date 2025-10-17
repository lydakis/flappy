"""Common interface dataclasses shared between driver, coach, and learner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class MaskDelta:
    """Coach-issued mask delta specifying allowed or blocked patterns."""

    allow: list[str] = field(default_factory=list)
    block: list[str] = field(default_factory=list)


@dataclass
class GoalContext:
    """Current goal text and embedding used by the driver."""

    text: str = ""
    embedding: Optional[np.ndarray] = None


@dataclass
class MaskDecision:
    """Final mask decision after combining coach and policy masks."""

    final: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    source: str = "none"
    coach: Optional[np.ndarray] = None
    policy: Optional[np.ndarray] = None
    guardrail_applied: bool = False

    def as_tuple(self) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], str]:
        return self.final, self.coach, self.policy, self.source


def _trim(mask: Optional[np.ndarray], action_count: int) -> Optional[np.ndarray]:
    if mask is None:
        return None
    if mask.ndim != 1:
        mask = mask.reshape(-1)
    if len(mask) >= action_count:
        return mask[:action_count].astype(np.float32, copy=False)
    padded = np.ones(action_count, dtype=np.float32)
    padded[: len(mask)] = mask.astype(np.float32, copy=False)
    return padded


def combine_masks(
    policy_mask: Optional[Sequence[float]],
    coach_mask: Optional[Sequence[float]],
    action_count: int,
    *,
    epsilon: float = 1e-3,
) -> MaskDecision:
    """Merge policy- and coach-proposed masks into a final decision."""
    policy_arr = (
        np.asarray(policy_mask, dtype=np.float32).reshape(-1) if policy_mask is not None else None
    )
    coach_arr = (
        np.asarray(coach_mask, dtype=np.float32).reshape(-1) if coach_mask is not None else None
    )
    trimmed_policy = _trim(policy_arr, action_count)
    trimmed_coach = _trim(coach_arr, action_count)

    final = np.ones(action_count, dtype=np.float32)
    source_components: list[str] = []
    if trimmed_policy is not None:
        final *= trimmed_policy
        source_components.append("policy")
    if trimmed_coach is not None:
        final *= trimmed_coach
        source_components.append("coach")
    source = "+".join(source_components) if source_components else "none"

    if np.all(final < epsilon):
        final[:] = 1.0
        source = "fallback"

    return MaskDecision(
        final=final,
        source=source,
        coach=trimmed_coach,
        policy=trimmed_policy,
        guardrail_applied=False,
    )
