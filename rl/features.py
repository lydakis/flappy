"""DOM feature encoders for RL policies."""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, List

import numpy as np


class DomTextHasher:
    """Hashing trick encoder for DOM text observations."""

    def __init__(self, dim: int = 2048, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.dim = dim
        self.ngram_range = ngram_range

    def encode(self, observation: Dict[str, str]) -> np.ndarray:
        text = observation.get("dom_text", "")
        tokens = _tokenize(text)
        features = np.zeros(self.dim, dtype=np.float32)
        for ngram in _generate_ngrams(tokens, self.ngram_range):
            index = int(hashlib.sha1(" ".join(ngram).encode("utf-8")).hexdigest(), 16)
            features[index % self.dim] += 1.0
        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm
        return features


def _tokenize(text: str) -> List[str]:
    return [tok for tok in text.lower().split() if tok]


def _generate_ngrams(tokens: Iterable[str], ngram_range: tuple[int, int]):
    tokens = list(tokens)
    n_min, n_max = ngram_range
    for n in range(n_min, n_max + 1):
        for i in range(len(tokens) - n + 1):
            yield tokens[i : i + n]
