"""Tiny plan DSL primitives for FLAPPY v0.2.

This module intentionally provides a lightweight abstraction only.
Concrete executors, compilers, and optimisation passes will live in
upcoming iterations; for now we expose dataclasses and basic parsing
helpers so other components (coach, synthesiser, verifier) can share a
common representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Optional


class DSLVerb(str, Enum):
    """Canonical kernel verbs supported by the DSL."""

    CLICK = "Click"
    TYPE = "Type"
    WAIT = "Wait"
    EXTRACT_ARTICLE = "ExtractArticle"
    WRITE_NOTE = "WriteNote"
    IF = "If"
    ELSE = "Else"
    FOR_EACH = "ForEach"
    QUERY = "Query"
    EXISTS = "Exists"


@dataclass
class DSLNode:
    """Base node in the DSL AST."""

    verb: DSLVerb
    args: List[str] = field(default_factory=list)
    children: List["DSLNode"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "verb": self.verb.value,
            "args": list(self.args),
            "children": [child.to_dict() for child in self.children],
        }

    @staticmethod
    def from_dict(payload: dict) -> "DSLNode":
        verb = DSLVerb(payload.get("verb", DSLVerb.CLICK.value))
        args = payload.get("args") or []
        children_payloads = payload.get("children") or []
        return DSLNode(
            verb=verb,
            args=list(args),
            children=[DSLNode.from_dict(child) for child in children_payloads],
        )


def render_plan(root: DSLNode, indent: int = 0) -> str:
    """Pretty-print a plan tree for logging/debugging."""
    prefix = "  " * indent
    line = f"{prefix}{root.verb.value}"
    if root.args:
        line += f"({', '.join(root.args)})"
    rendered = [line]
    for child in root.children:
        rendered.append(render_plan(child, indent + 1))
    return "\n".join(rendered)


def flatten_verbs(root: DSLNode) -> List[DSLVerb]:
    """Return a depth-first list of verbs for quick heuristics."""
    verbs = [root.verb]
    for child in root.children:
        verbs.extend(flatten_verbs(child))
    return verbs


def count_nodes(root: DSLNode) -> int:
    """Count the number of nodes in the plan tree."""
    total = 1
    for child in root.children:
        total += count_nodes(child)
    return total


def iter_nodes(root: DSLNode) -> Iterable[DSLNode]:
    """Yield nodes depth-first."""
    yield root
    for child in root.children:
        yield from iter_nodes(child)


def is_kernel_only(root: DSLNode) -> bool:
    """Return True if the plan contains only kernel primitives."""
    kernel_verbs = {
        DSLVerb.CLICK,
        DSLVerb.TYPE,
        DSLVerb.WAIT,
        DSLVerb.EXTRACT_ARTICLE,
        DSLVerb.WRITE_NOTE,
    }
    return all(node.verb in kernel_verbs for node in iter_nodes(root))


def make_leaf(verb: DSLVerb, *args: str) -> DSLNode:
    """Convenience helper for constructing a leaf node."""
    return DSLNode(verb=verb, args=list(args))


def make_control_flow(
    verb: DSLVerb,
    condition: Optional[str],
    body: List[DSLNode],
    else_body: Optional[List[DSLNode]] = None,
) -> DSLNode:
    """Build a control-flow node such as If/ForEach with optional children."""
    args: List[str] = []
    if condition:
        args.append(condition)
    node = DSLNode(verb=verb, args=args, children=list(body))
    if else_body:
        node.children.extend(else_body)
    return node


__all__ = [
    "DSLNode",
    "DSLVerb",
    "render_plan",
    "flatten_verbs",
    "count_nodes",
    "iter_nodes",
    "is_kernel_only",
    "make_leaf",
    "make_control_flow",
]
