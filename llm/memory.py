"""Episodic memory storage for Reflexion-style feedback."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class MemoryEntry:
    task_id: str
    episode_id: str
    success: bool
    notes: str
    selectors_used: List[str]
    failure_modes: List[str]
    subgoal: str | None = None
    mask_allow: List[str] = field(default_factory=list)
    mask_block: List[str] = field(default_factory=list)


class JsonlMemoryStore:
    """Append-only JSONL episodic memory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: MemoryEntry) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(asdict(entry), handle)
            handle.write("\n")

    def load(self) -> List[MemoryEntry]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            return [MemoryEntry(**json.loads(line)) for line in handle if line.strip()]


def retrieve_top_k(
    entries: Iterable[MemoryEntry],
    task_id: str,
    query: str,
    k: int = 3,
) -> List[MemoryEntry]:
    """Rank memory entries via simple BM25-style scoring."""
    candidates = [entry for entry in entries if entry.task_id == task_id]
    if not candidates:
        return []

    query_tokens = _tokenize(query)
    doc_freq: Dict[str, int] = {}
    for entry in candidates:
        seen = set()
        for term in _tokenize(entry.notes):
            if term not in seen:
                doc_freq[term] = doc_freq.get(term, 0) + 1
                seen.add(term)

    def score(entry: MemoryEntry) -> float:
        score_val = 0.0
        words = _tokenize(entry.notes)
        for term in query_tokens:
            tf = words.count(term)
            if tf == 0:
                continue
            df = doc_freq.get(term, 1)
            idf = math.log((len(candidates) - df + 0.5) / (df + 0.5) + 1)
            score_val += idf * ((tf * (1.2 + 1)) / (tf + 1.2 * (1 - 0.75 + 0.75)))
        return score_val

    ranked = sorted(candidates, key=score, reverse=True)
    return ranked[:k]


def _tokenize(text: str) -> List[str]:
    return [token for token in text.lower().split() if token]


def load_memory(path: str | Path) -> JsonlMemoryStore:
    return JsonlMemoryStore(path)
