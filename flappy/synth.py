"""Plan sketch synthesiser scaffolding.

The synthesiser fills holes in DSL sketches using DOM-derived affordances.
For v0.2 we provide a placeholder interface so future implementations
can plug in heuristic search, LLM ranking, or learned policies.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from flappy.dsl import DSLNode, DSLVerb


@dataclass
class Sketch:
    """Represents a plan with optional holes to be instantiated."""

    root: DSLNode
    holes: List[str]


@dataclass
class CandidatePlan:
    """Synthesised plan proposal with a simple score."""

    root: DSLNode
    score: float = 0.0
    rationale: Optional[str] = None


class PlanSynthesiser:
    """Placeholder enumerator for filling DSL sketches.

    The current implementation returns the original sketch untouched.
    Later versions will search over DOM elements / macros to bind free
    variables.
    """

    def enumerate(
        self,
        sketch: Sketch,
        *,
        max_candidates: int = 3,
        context: Optional[dict] = None,
    ) -> Iterable[CandidatePlan]:
        selectors = self._extract_selectors(context)
        plan_copy = copy.deepcopy(sketch.root)
        if self._plan_compatible(plan_copy, selectors):
            yield CandidatePlan(root=plan_copy, score=1.0, rationale="selectors-bound")
        elif not selectors:
            # If no selector information is available, fall back to the raw sketch.
            yield CandidatePlan(root=plan_copy, score=0.0, rationale="identity-no-selectors")

    def _extract_selectors(self, context: Optional[dict]) -> Sequence[str]:
        if not context:
            return []
        selectors = context.get("selectors")
        if isinstance(selectors, (list, tuple)):
            return selectors
        return []

    def _plan_compatible(self, node: DSLNode, selectors: Sequence[str]) -> bool:
        if node.verb in {DSLVerb.CLICK, DSLVerb.TYPE} and node.args:
            selector = self._normalise_selector(node.args[0])
            if selector and selectors and selector not in selectors:
                return False
        for child in node.children:
            if not self._plan_compatible(child, selectors):
                return False
        return True

    @staticmethod
    def _normalise_selector(raw_selector: str) -> str:
        token = raw_selector.split(",", 1)[0].strip()
        token = token.rstrip(")").strip()
        token = token.strip("\"'")
        return token


__all__ = ["Sketch", "CandidatePlan", "PlanSynthesiser"]
