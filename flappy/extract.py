"""Document extraction helpers and metadata dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ExtractedDocument:
    """Structured representation of a cleaned web article."""

    url: str
    title: str
    byline: Optional[str]
    published_at: Optional[datetime]
    text: str
    embedding: Optional[list[float]] = None


class DocumentExtractor:
    """Thin wrapper around readability-like extractors (stub)."""

    def __init__(self, *, strategy: str = "readability") -> None:
        self.strategy = strategy

    def extract(self, html: str, url: str) -> ExtractedDocument:
        """Return a placeholder document until a real parser is wired."""
        return ExtractedDocument(
            url=url,
            title="(untitled)",
            byline=None,
            published_at=None,
            text=html,
            embedding=None,
        )


__all__ = ["ExtractedDocument", "DocumentExtractor"]
