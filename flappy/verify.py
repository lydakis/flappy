"""Verification scaffolding for plan execution signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from flappy.dsl import DSLNode, DSLVerb

Predicate = Callable[[Dict[str, object]], bool]


@dataclass
class VerificationResult:
    """Outcome of checking a plan or step."""

    ok: bool
    reason: Optional[str] = None
    rewards: Dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.rewards is None:
            self.rewards = {}


class PlanVerifier:
    """Minimal verifier placeholder enforcing trivial checks."""

    def __init__(self, *, predicates: Optional[Dict[str, Predicate]] = None) -> None:
        self.predicates = predicates or {}

    def register_predicate(self, name: str, predicate: Predicate) -> None:
        self.predicates[name] = predicate

    def verify(
        self,
        node: DSLNode,
        trace: Iterable[Dict[str, object]],
        *,
        context: Optional[dict] = None,
    ) -> VerificationResult:
        """Ensure that selectors referenced in the plan exist in the context."""
        del trace  # unused for now
        selectors = self._extract_selectors(context)
        missing = self._find_missing_selectors(node, selectors)
        if missing:
            reason = f"missing selectors: {sorted(missing)}"
            return VerificationResult(ok=False, reason=reason)
        return VerificationResult(ok=True, reason="selectors-present")

    def _extract_selectors(self, context: Optional[dict]) -> Sequence[str]:
        if not context:
            return []
        selectors = context.get("selectors")
        if isinstance(selectors, (list, tuple)):
            return selectors
        return []

    def _find_missing_selectors(self, node: DSLNode, selectors: Sequence[str]) -> set[str]:
        missing: set[str] = set()
        if node.verb in {DSLVerb.CLICK, DSLVerb.TYPE} and node.args:
            selector = self._normalise_selector(node.args[0])
            if selector and selectors and selector not in selectors:
                missing.add(selector)
        for child in node.children:
            missing.update(self._find_missing_selectors(child, selectors))
        return missing

    @staticmethod
    def _normalise_selector(raw_selector: str) -> str:
        token = raw_selector.split(",", 1)[0].strip()
        token = token.rstrip(")").strip()
        token = token.strip("\"'")
        return token


__all__ = ["PlanVerifier", "VerificationResult", "Predicate"]
