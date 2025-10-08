"""Lightweight buffers for logging exploration statistics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Iterator, List, Tuple


@dataclass
class Transition:
    state: bytes
    action: bytes
    reward: float
    done: bool


class EpisodeLogBuffer:
    """FIFO buffer for logging recent transitions."""

    def __init__(self, capacity: int = 5_000) -> None:
        self.capacity = capacity
        self._items: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self._items.append(transition)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Transition]:
        return iter(self._items)

    def to_list(self) -> List[Transition]:
        return list(self._items)
