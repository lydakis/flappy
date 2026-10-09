"""JSONL event log plus optional TensorBoard scalars for one run."""

from __future__ import annotations

import json
from pathlib import Path


class RunLog:
    """Append-only run log. ``RunLog(None)`` keeps events in memory only."""

    def __init__(self, directory: Path | None, *, tensorboard: bool = True):
        self.events: list[dict] = []
        self.directory = directory
        self._stream = None
        self._writer = None
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=False)
            self._stream = (directory / "events.jsonl").open("a")
            if tensorboard:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(str(directory / "tb"))

    def event(self, record: dict) -> None:
        self.events.append(record)
        if self._stream is not None:
            self._stream.write(json.dumps(record, sort_keys=True) + "\n")
            self._stream.flush()

    def scalars(self, step: int, values: dict[str, float]) -> None:
        self.event({"type": "scalars", "tick": step, **values})
        if self._writer is not None:
            for key, value in values.items():
                self._writer.add_scalar(key, value, step)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
        if self._writer is not None:
            self._writer.close()
