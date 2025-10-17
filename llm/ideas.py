"""Idea data structures and JSONL-backed store."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


IDEA_SCORE_KEYS = ("novelty", "coherence", "usefulness")


def _utcnow_iso() -> str:
    return datetime.utcnow().isoformat()


@dataclass
class Idea:
    """Structured representation of a synthesized idea."""

    id: str
    hypothesis: str
    scores: Dict[str, float]
    accepted: bool
    source_ids: List[str] = field(default_factory=list)
    task_scope: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    prompt_versions: Dict[str, str] = field(default_factory=dict)
    timestamp: str = field(default_factory=_utcnow_iso)
    hash: Optional[str] = None
    cost: Dict[str, float] = field(default_factory=dict)
    justification: str = ""

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload.setdefault("id", str(uuid.uuid4()))
        payload.setdefault("timestamp", _utcnow_iso())
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Idea":
        raw_scores = data.get("scores", {}) or {}
        scores = {str(k): float(v) for k, v in raw_scores.items()}
        for key in IDEA_SCORE_KEYS:
            scores.setdefault(key, 0.0)
        return cls(
            id=str(data.get("id") or uuid.uuid4()),
            hypothesis=str(data.get("hypothesis", "")),
            scores=scores,
            accepted=bool(data.get("accepted", False)),
            source_ids=[str(item) for item in data.get("source_ids", [])],
            task_scope=str(data.get("task_scope")) if data.get("task_scope") else None,
            tags=[str(tag) for tag in data.get("tags", [])],
            prompt_versions={str(k): str(v) for k, v in data.get("prompt_versions", {}).items()},
            timestamp=str(data.get("timestamp", _utcnow_iso())),
            hash=str(data.get("hash")) if data.get("hash") else None,
            cost={str(k): float(v) for k, v in data.get("cost", {}).items()},
            justification=str(data.get("justification", "")),
        )


class IdeaStore:
    """Append-only JSONL-backed idea store with naive search and dedup helpers."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, idea: Idea) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(idea.to_dict()) + "\n")

    def load(self) -> List[Idea]:
        if not self.path.exists():
            return []
        ideas: List[Idea] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ideas.append(Idea.from_dict(payload))
        return ideas

    def hashes(self) -> Iterable[str]:
        for idea in self.load():
            if idea.hash:
                yield idea.hash

    def contains_hash(self, idea_hash: str) -> bool:
        return any(existing == idea_hash for existing in self.hashes())

    def search(self, query: str, *, limit: int = 5, task_scope: Optional[str] = None) -> List[Idea]:
        query_lower = query.lower()
        results: List[Idea] = []
        for idea in self.load():
            if task_scope is not None and idea.task_scope not in {task_scope, None}:
                continue
            haystack = " ".join([idea.hypothesis, " ".join(idea.tags)]).lower()
            if query_lower in haystack:
                results.append(idea)
            if len(results) >= limit:
                break
        return results


__all__ = ["Idea", "IdeaStore", "IDEA_SCORE_KEYS"]
