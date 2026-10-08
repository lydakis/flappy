"""A new interactive ledger linked to the immutable, closed prior experiment."""

from __future__ import annotations

import hashlib
import json
import os
from decimal import Decimal
from pathlib import Path

from llm.budgeted_teacher import PER_ATTEMPT, PRICING, BudgetStop, SharedBudget, _now

TOTAL_CEILING = Decimal("5.00")
STOP_BELOW = Decimal("4.90")
MAX_ATTEMPTS = 10
LIVE_PRICING = {
    **PRICING,
    "max_attempts": MAX_ATTEMPTS,
    "ceiling_usd": str(TOTAL_CEILING),
}


class LinkedBudget(SharedBudget):
    """Retain every previous reservation; fail closed if the prior ledger changes."""

    def __init__(self, path: Path, previous: Path):
        super().__init__(path)
        self.previous = previous.resolve()

    def _prior_link(self) -> dict:
        prior = SharedBudget(self.previous).snapshot()
        reserved = sum(Decimal(a["reserved_usd"]) for a in prior["attempts"])
        if not prior["closed"] or not reserved.is_finite() or reserved < 0:
            raise BudgetStop("Prior budget is not the expected closed ledger")
        return {
            "path": str(self.previous),
            "sha256": hashlib.sha256(self.previous.read_bytes()).hexdigest(),
            "reserved_usd": str(reserved),
            "total_ceiling_usd": str(TOTAL_CEILING),
            "stop_below_usd": str(STOP_BELOW),
        }

    def initialize(self) -> None:
        link = self._prior_link()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked(), self.path.open("x") as stream:
            json.dump(
                {
                    "pricing": LIVE_PRICING,
                    "prior": link,
                    "attempts": [],
                    "closed": False,
                },
                stream,
                indent=2,
            )
            stream.flush()
            os.fsync(stream.fileno())

    def _read(self) -> dict:
        try:
            state = json.loads(self.path.read_text())
            if (
                state["pricing"] != LIVE_PRICING
                or state["prior"] != self._prior_link()
                or type(state["closed"]) is not bool
                or not isinstance(state["attempts"], list)
                or len(state["attempts"]) > MAX_ATTEMPTS
            ):
                raise ValueError("invalid ledger")
            for index, attempt in enumerate(state["attempts"]):
                if (
                    attempt["id"] != index
                    or attempt["reserved_usd"] != str(PER_ATTEMPT)
                    or attempt["status"] not in {"reserved", "complete", "failed"}
                ):
                    raise ValueError("invalid reservation")
            return state
        except (OSError, ValueError, TypeError, KeyError):
            raise BudgetStop(
                "Linked budget missing, corrupt, or prior ledger changed"
            ) from None

    def reserve(self, purpose: str) -> int:
        if purpose not in {"advice", "reflection", "demonstrations"}:
            raise BudgetStop("Unknown teacher purpose")
        with self._locked():
            state = self._read()
            attempts = state["attempts"]
            if state["closed"] or any(a["status"] != "complete" for a in attempts):
                raise BudgetStop("Interactive budget closed or unresolved attempt")
            if (
                len(attempts) >= MAX_ATTEMPTS
                or Decimal(state["prior"]["reserved_usd"])
                + (len(attempts) + 1) * PER_ATTEMPT
                > STOP_BELOW
            ):
                raise BudgetStop("Cumulative API reservation limit reached")
            attempt_id = len(attempts)
            attempts.append(
                {
                    "id": attempt_id,
                    "purpose": purpose,
                    "reserved_usd": str(PER_ATTEMPT),
                    "status": "reserved",
                    "reserved_at": _now(),
                }
            )
            self._write(state)
            return attempt_id
