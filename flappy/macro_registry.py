"""Macro registry scaffolding for runtime skill promotion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from flappy.dsl import DSLNode


@dataclass
class Macro:
    """Represents a reusable macro composed of DSL nodes."""

    name: str
    params: Dict[str, str] = field(default_factory=dict)
    body: List[DSLNode] = field(default_factory=list)
    tier: str = "ephemeral"  # ephemeral | site | library
    usage_count: int = 0
    success_count: int = 0


class MacroRegistry:
    """Mutable registry for proposed and promoted macros (stub)."""

    def __init__(self) -> None:
        self._macros: Dict[str, Macro] = {}

    def propose(self, macro: Macro) -> None:
        self._macros[macro.name] = macro

    def promote(self, name: str, tier: str) -> None:
        if name in self._macros:
            self._macros[name].tier = tier

    def demote(self, name: str) -> None:
        if name in self._macros:
            self._macros[name].tier = "ephemeral"

    def record_success(self, name: str, success: bool) -> None:
        macro = self._macros.get(name)
        if macro is None:
            return
        macro.usage_count += 1
        if success:
            macro.success_count += 1

    def list_for(self, tier: Optional[str] = None) -> List[Macro]:
        if tier is None:
            return list(self._macros.values())
        return [macro for macro in self._macros.values() if macro.tier == tier]

    def stats(self) -> Dict[str, Dict[str, int]]:
        return {
            name: {
                "usage": macro.usage_count,
                "success": macro.success_count,
            }
            for name, macro in self._macros.items()
        }


__all__ = ["Macro", "MacroRegistry"]
