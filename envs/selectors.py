"""Selector utilities for BrowserGym DOM interactions."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class SelectorCandidate:
    selector: str
    score: float


def normalize_label(text: str) -> str:
    """Normalize DOM labels to aid matching."""
    lowered = text.strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def stable_selector(tag: str, attributes: dict[str, str]) -> str:
    """Compose a CSS selector that resists DOM drift."""
    parts: List[str] = [tag]
    for attr in ("id", "name", "aria-label", "placeholder", "role", "value"):
        value = attributes.get(attr)
        if value:
            sanitized = css_escape(value)
            parts.append(f'[{attr}="{sanitized}"]')
    return "".join(parts)


def css_escape(value: str) -> str:
    """Escape CSS special characters."""
    safe = re.sub(r'(["\\])', r"\\\1", value)
    return safe


def hash_selector(selector: str) -> str:
    """Create a short fingerprint useful for logging."""
    digest = hashlib.sha1(selector.encode("utf-8")).hexdigest()[:8]
    return digest


def rank_candidates(candidates: Iterable[SelectorCandidate]) -> List[SelectorCandidate]:
    """Order selector candidates descending by score."""
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def best_selector(candidates: Iterable[SelectorCandidate]) -> Optional[str]:
    ranked = rank_candidates(candidates)
    return ranked[0].selector if ranked else None


INTERACTIVE_TAGS = {"button", "input", "a", "textarea", "select", "option"}
ATTR_PATTERN = re.compile(
    r'(?P<attr>id|name|aria-label|placeholder|data-testid|role)="(?P<value>[^"]+)"'
)
TAG_PATTERN = re.compile(
    r"<(?P<tag>[a-zA-Z0-9]+)(?P<body>[^>]*)>", re.MULTILINE | re.IGNORECASE
)


def extract_interactive_selectors(dom_text: str, max_candidates: int = 12) -> List[str]:
    """Return heuristic CSS selectors for interactive elements."""
    candidates: List[SelectorCandidate] = []
    seen: set[str] = set()
    for match in TAG_PATTERN.finditer(dom_text):
        tag = match.group("tag").lower()
        if tag not in INTERACTIVE_TAGS:
            continue
        attr_body = match.group("body")
        attrs = _parse_attributes(attr_body)
        selector = stable_selector(tag, attrs)
        if selector in seen:
            continue
        score = _score_attributes(attrs)
        candidates.append(SelectorCandidate(selector=selector, score=score))
        seen.add(selector)
        if len(candidates) >= max_candidates:
            break
    return [candidate.selector for candidate in rank_candidates(candidates)]


def _parse_attributes(attr_body: str) -> Dict[str, str]:
    attributes: Dict[str, str] = {}
    for attr_match in ATTR_PATTERN.finditer(attr_body):
        attr = attr_match.group("attr")
        value = attr_match.group("value")
        attributes[attr] = value
    return attributes


def _score_attributes(attributes: Dict[str, str]) -> float:
    score = 0.0
    if "id" in attributes:
        score += 3.0
    if "name" in attributes:
        score += 2.0
    if "aria-label" in attributes or "placeholder" in attributes:
        score += 1.5
    if "data-testid" in attributes:
        score += 1.0
    return score or 0.5
