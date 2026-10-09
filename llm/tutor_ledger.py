"""Fail-closed ledger and request path for the curious-student tutor.

Follows the ``SharedBudget``/``ContinualBudget`` pattern: every request reserves a
worst-case charge before sending, reservations are never refunded, an ambiguous
outcome closes the ledger, and a missing or altered ledger never resets itself.
The ledger caps cumulative reservations at $5 and each run at its own allowance.
"""

from __future__ import annotations

import json
import os
import re
from decimal import Decimal
from pathlib import Path

from llm.budgeted_teacher import BudgetedTeacher, BudgetStop, SharedBudget, _now

# Verified against the developers.openai.com model pages (gpt-6-luna on 2026-10-09).
PRICES = {
    "gpt-6-luna": ("0.10", "0.50", "none"),
    "gpt-5.4-mini-2026-03-17": ("0.75", "4.50", "none"),
    "gpt-5-mini-2025-08-07": ("0.25", "2.00", "minimal"),
}
DEFAULT_MODEL = "gpt-6-luna"
MAX_PROMPT_BYTES = 4000
PURPOSES = {"hint", "worked_example", "explanation"}
RUN_ID_RE = re.compile(r"[a-z0-9_.-]{1,64}")


def pricing_for(model: str) -> dict:
    if model not in PRICES:
        raise BudgetStop("Tutor model has no verified pricing")
    input_usd, output_usd, effort = PRICES[model]
    return {
        "model": model,
        "verified_date": "2026-10-08",
        "input_usd_per_million": input_usd,
        "output_usd_per_million": output_usd,
        "reasoning_effort": effort,
        # ASCII prompts are capped in bytes; one token never covers less than a
        # byte, and the margin covers message framing.
        "max_prompt_bytes": MAX_PROMPT_BYTES,
        "max_input_tokens": MAX_PROMPT_BYTES + 96,
        "max_output_tokens": 768,
        "reservation_multiplier": "1.10",
        "total_ceiling_usd": "5.00",
    }


def per_attempt(pricing: dict) -> Decimal:
    return (
        (
            Decimal(pricing["input_usd_per_million"]) * pricing["max_input_tokens"]
            + Decimal(pricing["output_usd_per_million"]) * pricing["max_output_tokens"]
        )
        / Decimal(1_000_000)
        * Decimal(pricing["reservation_multiplier"])
    )


class TutorLedger(SharedBudget):
    """Locked ledger with a $5 cumulative ceiling and per-run allowances."""

    def __init__(self, path: Path, model: str = DEFAULT_MODEL):
        super().__init__(path)
        self.pricing = pricing_for(model)
        self.per_attempt = per_attempt(self.pricing)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked(), self.path.open("x") as stream:
            json.dump(
                {"pricing": self.pricing, "attempts": [], "closed": False},
                stream,
                indent=2,
            )
            stream.flush()
            os.fsync(stream.fileno())

    def _read(self) -> dict:
        try:
            state = json.loads(self.path.read_text())
            if state["pricing"] != self.pricing or type(state["closed"]) is not bool:
                raise ValueError("incompatible ledger")
            attempts = state["attempts"]
            if not isinstance(attempts, list):
                raise TypeError("invalid attempts")
            for index, row in enumerate(attempts):
                if (
                    row["id"] != index
                    or row["reserved_usd"] != str(self.per_attempt)
                    or not RUN_ID_RE.fullmatch(row["run_id"])
                    or row["purpose"] not in PURPOSES
                    or row["status"] not in {"reserved", "complete", "failed"}
                ):
                    raise ValueError("invalid reservation")
            if len(attempts) * self.per_attempt > Decimal(
                self.pricing["total_ceiling_usd"]
            ):
                raise ValueError("ceiling exceeded")
            return state
        except (OSError, ValueError, KeyError, TypeError):
            raise BudgetStop("Tutor ledger missing, corrupt or incompatible") from None

    def reserve(self, run_id: str, purpose: str, run_cap_usd: Decimal) -> int:
        if not RUN_ID_RE.fullmatch(run_id) or purpose not in PURPOSES:
            raise BudgetStop("Unknown tutor run or purpose")
        with self._locked():
            state = self._read()
            attempts = state["attempts"]
            if state["closed"] or any(a["status"] != "complete" for a in attempts):
                raise BudgetStop("Tutor budget closed or unresolved attempt")
            run_count = sum(a["run_id"] == run_id for a in attempts)
            if (len(attempts) + 1) * self.per_attempt > Decimal(
                self.pricing["total_ceiling_usd"]
            ) or (run_count + 1) * self.per_attempt > run_cap_usd:
                raise BudgetStop("Tutor total or per-run allowance reached")
            attempt_id = len(attempts)
            attempts.append(
                {
                    "id": attempt_id,
                    "run_id": run_id,
                    "purpose": purpose,
                    "reserved_usd": str(self.per_attempt),
                    "status": "reserved",
                    "reserved_at": _now(),
                }
            )
            self._write(state)
            return attempt_id

    def finish(self, attempt_id: int, usage: dict | None) -> None:
        with self._locked():
            state = self._read()
            row = state["attempts"][attempt_id]
            if row["status"] != "reserved":
                raise BudgetStop("Attempt already finalized")
            valid = isinstance(usage, dict) and all(
                type(usage.get(name)) is int and 0 <= usage[name] <= bound
                for name, bound in (
                    ("input_tokens", self.pricing["max_input_tokens"]),
                    ("output_tokens", self.pricing["max_output_tokens"]),
                    ("reasoning_tokens", self.pricing["max_output_tokens"]),
                )
            )
            valid = valid and usage["reasoning_tokens"] <= usage["output_tokens"]
            row["finished_at"] = _now()
            if valid:
                row["status"] = "complete"
                row["usage"] = usage
                row["estimated_usd_no_cache_discount"] = str(
                    (
                        Decimal(self.pricing["input_usd_per_million"])
                        * usage["input_tokens"]
                        + Decimal(self.pricing["output_usd_per_million"])
                        * usage["output_tokens"]
                    )
                    / Decimal(1_000_000)
                )
            else:
                row["status"] = "failed"
                state["closed"] = True
            self._write(state)
            if not valid:
                raise BudgetStop(
                    "Uncertain tutor usage; reservation kept, ledger closed"
                )

    def totals(self, run_id: str | None = None) -> dict:
        """Reserved and estimated spend, overall or for one run."""
        attempts = [
            a
            for a in self.snapshot()["attempts"]
            if run_id is None or a["run_id"] == run_id
        ]
        return {
            "calls": len(attempts),
            "reserved_usd": float(sum(Decimal(a["reserved_usd"]) for a in attempts)),
            "estimated_usd": float(
                sum(
                    Decimal(a.get("estimated_usd_no_cache_discount", "0"))
                    for a in attempts
                )
            ),
        }


class LedgerTutorClient(BudgetedTeacher):
    """One narrow Responses request per tutor call, reserved before sending.

    Build with ``LedgerTutorClient.from_key_file(ledger, path)``; then call
    ``bind_run`` once before ``request``.
    """

    run_id: str = ""
    run_cap_usd: Decimal = Decimal(0)

    def bind_run(self, run_id: str, run_cap_usd: float) -> None:
        cap = Decimal(str(run_cap_usd))
        if not RUN_ID_RE.fullmatch(run_id) or not Decimal(0) < cap <= Decimal(5):
            raise BudgetStop("Invalid run id or run allowance")
        self.run_id, self.run_cap_usd = run_id, cap

    def request(self, prompt: str, purpose: str) -> str | None:
        """Return tutor text, or None if the reply was cut off (usage still billed)."""
        ledger: TutorLedger = self.budget  # type: ignore[assignment]
        pricing = ledger.pricing
        if (
            not self.run_id
            or not isinstance(prompt, str)
            or not prompt.isascii()
            or len(prompt.encode()) > pricing["max_prompt_bytes"]
            or (self._secret and self._secret in prompt)
        ):
            raise BudgetStop("Invalid tutor request")
        attempt = ledger.reserve(self.run_id, purpose, self.run_cap_usd)
        try:
            response = self._client.responses.create(
                model=pricing["model"],
                input=prompt,
                reasoning={"effort": pricing["reasoning_effort"]},
                max_output_tokens=pricing["max_output_tokens"],
                service_tier="default",
                store=False,
            )
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "reasoning_tokens": response.usage.output_tokens_details.reasoning_tokens,
            }
        # HTTP/auth/prompt exception contents must never reach output.
        except Exception:  # noqa: BLE001
            try:
                ledger.finish(attempt, None)
            except BudgetStop:
                pass
            ledger.close()
            raise BudgetStop(
                "Tutor request failed; no retry; reservation kept"
            ) from None
        ledger.finish(attempt, usage)
        if not re.fullmatch(
            re.escape(pricing["model"]) + r"(-\d{4}-\d{2}-\d{2})?", str(response.model)
        ):
            ledger.close()
            raise BudgetStop("Tutor response from unexpected model; ledger closed")
        if response.status == "incomplete":
            return None
        text = response.output_text
        if (
            response.status != "completed"
            or not isinstance(text, str)
            or len(text.encode()) > 8192
            or (self._secret and self._secret in text)
        ):
            ledger.close()
            raise BudgetStop("Tutor response validation failed; ledger closed")
        return text
