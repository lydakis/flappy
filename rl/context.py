"""Context utilities for coach-driven RL policies."""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


class SubgoalEncoder:
    """Hashing-trick encoder for textual subgoals."""

    def __init__(self, dim: int = 256, seed: int = 7) -> None:
        self.dim = dim
        self.seed = seed

    def encode(self, text: str) -> np.ndarray:
        tokens = text.lower().split()
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in self._ngrams(tokens):
            digest = hashlib.sha1(f"{self.seed}:{token}".encode("utf-8")).hexdigest()
            index = int(digest, 16) % self.dim
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def _ngrams(self, tokens: Iterable[str], n_min: int = 1, n_max: int = 2):
        tokens = list(tokens)
        for n in range(n_min, n_max + 1):
            for i in range(len(tokens) - n + 1):
                yield " ".join(tokens[i : i + n])
