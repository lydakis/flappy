"""Retrieval-augmented answer scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from flappy.memory import Note


@dataclass
class Answer:
    """Structured answer with inline citations."""

    text: str
    citations: List[str]
    confidence: float = 0.0


class SimpleRAG:
    """Stub RAG answerer stitching notes into a response."""

    def __init__(self, *, max_notes: int = 3) -> None:
        self.max_notes = max_notes

    def answer(self, question: str, notes: List[Note]) -> Answer:
        del question  # unused in the stub
        top = notes[: self.max_notes]
        if not top:
            return Answer(text="No relevant notes yet.", citations=[], confidence=0.0)
        sentences = [note.snippet for note in top]
        citations = [note.url for note in top]
        return Answer(text=" ".join(sentences), citations=citations, confidence=0.1)


__all__ = ["Answer", "SimpleRAG"]
