"""Typed episodic memory with basic vector index scaffolding."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class Note:
    """Typed note entry for the knowledge store."""

    id: str
    url: str
    title: str
    snippet: str
    note_type: str = "text"
    units: Optional[str] = None
    span: Optional[List[int]] = None
    embedding: Optional[List[float]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


class NoteStore:
    """JSONL-backed note store; vector index to be added later."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, note: Note) -> None:
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(note.to_dict()) + "\n")

    def load(self) -> List[Note]:
        if not self.path.exists():
            return []
        notes: List[Note] = []
        with self.path.open("r", encoding="utf-8") as fp:
            for line in fp:
                record = json.loads(line)
                ts = record.get("timestamp")
                timestamp = datetime.fromisoformat(ts) if ts else datetime.utcnow()
                notes.append(
                    Note(
                        id=record["id"],
                        url=record["url"],
                        title=record.get("title", ""),
                        snippet=record.get("snippet", ""),
                        note_type=record.get("note_type", "text"),
                        units=record.get("units"),
                        span=record.get("span"),
                        embedding=record.get("embedding"),
                        timestamp=timestamp,
                        confidence=float(record.get("confidence", 0.0)),
                    )
                )
        return notes

    def search(self, query: str, *, limit: int = 5) -> List[Note]:
        """Placeholder retrieval using naive substring matching."""
        results: List[Note] = []
        for note in self.load():
            haystack = " ".join([note.title, note.snippet]).lower()
            if query.lower() in haystack:
                results.append(note)
            if len(results) >= limit:
                break
        return results


__all__ = ["Note", "NoteStore"]
