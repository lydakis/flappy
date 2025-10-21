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
        normalized_selectors = {
            selector.lower() for selector in selectors if isinstance(selector, str)
        }

        missing: set[str] = set()

        def _visit(current: DSLNode) -> None:
            if current.verb in {DSLVerb.CLICK, DSLVerb.TYPE} and current.args:
                selector = current.args[0]
                if isinstance(selector, str) and selector:
                    if selector.lower() not in normalized_selectors:
                        missing.add(selector)
                elif selector:
                    # Non-string selectors cannot be matched against the context.
                    missing.add(str(selector))
            for child in current.children:
                _visit(child)

        _visit(node)
        return missing


__all__ = ["PlanVerifier", "VerificationResult", "Predicate"]
