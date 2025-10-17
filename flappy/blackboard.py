"""Shared blackboard protocol for coach⇄driver communication."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from flappy.interfaces import MaskDelta


@dataclass
class AffordanceHint:
    """Lightweight description of an action affordance surfaced to the coach."""

    idx: int
    text: str
    score: float = 0.0


@dataclass
class DriverEvent:
    """Structured driver-side event."""

    label: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriverToCoachMessage:
    """Signals emitted by the driver for low-frequency coaching."""

    surprisal: float = 0.0
    affordances: List[AffordanceHint] = field(default_factory=list)
    events: List[DriverEvent] = field(default_factory=list)
    suggested_query: Optional[str] = None


@dataclass
class CoachToDriverMessage:
    """Coach directives consumed by the driver."""

    goal: str = ""
    mask_delta: MaskDelta = field(default_factory=MaskDelta)
    plan_sketch: Optional[str] = None
    notes_request: Optional[str] = None


@dataclass
class BlackboardState:
    """Mutable shared state exchanged between driver and coach."""

    driver_to_coach: DriverToCoachMessage = field(default_factory=DriverToCoachMessage)
    coach_to_driver: CoachToDriverMessage = field(default_factory=CoachToDriverMessage)

    def clear(self) -> None:
        """Reset both directions to defaults."""
        self.driver_to_coach = DriverToCoachMessage()
        self.coach_to_driver = CoachToDriverMessage()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot."""
        return dataclasses.asdict(self)

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Return the blackboard as JSON for prompt injection or logging."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

