"""New capped tutor ledger; retain explicitly supplied closed ledgers byte-for-byte."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

from llm.budgeted_teacher import BudgetedTeacher, BudgetStop, SharedBudget, _now

MODEL = "gpt-5.4-mini-2026-03-17"
PRICING = {
    "model": MODEL,
    "verified_date": "2026-10-08",
    "source": "https://developers.openai.com/api/docs/models/gpt-5.4-mini",
    "input_usd_per_million": "0.75",
    "output_usd_per_million": "4.50",
    "max_input_tokens": 2048,
    "max_output_tokens": 512,
    "input_reservation_multiplier": 2,
    "input_multiplier_reason": "Reserve a second full input charge for counting overhead; do not assume a refunded/free preflight",
    "reservation_multiplier": "1.10",
    "max_attempts": 72,
    "max_attempts_per_run": 12,
    "total_ceiling_usd": "5.00",
    "stop_below_usd": "4.90",
    "old_holds_released": False,
}
PER_ATTEMPT = (
    (Decimal("0.75") * 2048 * 2 + Decimal("4.50") * 512)
    / Decimal(1000000)
    * Decimal("1.10")
)
RUN_IDS = {f"{arm}-{seed}" for arm in ("learned", "heuristic") for seed in (7, 19, 43)}


def reconciliation(root: Path, expected_priors: dict[str, str]) -> dict:
    rows = []
    for relative, expected in expected_priors.items():
        path = root / relative
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise BudgetStop("Prior ledger changed")
        prior = json.loads(raw)
        if not prior["closed"] or any(
            a["status"] != "complete" for a in prior["attempts"]
        ):
            raise BudgetStop("Prior ledger not closed and complete")
        rows.append(
            {
                "path": relative,
                "sha256": expected,
                "closed": True,
                "completed_attempts": len(prior["attempts"]),
                "reserved_usd": str(
                    sum(Decimal(a["reserved_usd"]) for a in prior["attempts"])
                ),
                "usage_estimate_usd_not_verified_billing": str(
                    sum(
                        Decimal(a["estimated_usd_no_cache_discount"])
                        for a in prior["attempts"]
                    )
                ),
            }
        )
    prior_reserved = sum((Decimal(r["reserved_usd"]) for r in rows), Decimal(0))
    if not prior_reserved.is_finite() or prior_reserved < 0:
        raise BudgetStop("Invalid prior reservations")
    return {
        "ledgers": rows,
        "prior_reserved_usd": str(prior_reserved),
        "prior_usage_estimate_usd_not_verified_billing": str(
            sum(
                (Decimal(r["usage_estimate_usd_not_verified_billing"]) for r in rows),
                Decimal(0),
            )
        ),
        "billing_verified": False,
        "holds_released_usd": "0",
        "remaining_to_original_ceiling_usd": str(Decimal(5) - prior_reserved),
        "remaining_to_stop_guard_usd": str(Decimal("4.90") - prior_reserved),
        "new_max_reserved_usd": str(PER_ATTEMPT * 72),
        "maximum_cumulative_reserved_usd": str(prior_reserved + PER_ATTEMPT * 72),
    }


class ContinualBudget(SharedBudget):
    def __init__(self, path: Path, root: Path, *, expected_priors: dict[str, str]):
        super().__init__(path)
        self.root = root
        self.expected_priors = dict(expected_priors)

    def initialize(self) -> None:
        prior = reconciliation(self.root, self.expected_priors)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked(), self.path.open("x") as stream:
            json.dump(
                {"pricing": PRICING, "prior": prior, "attempts": [], "closed": False},
                stream,
                indent=2,
            )
            stream.flush()
            import os

            os.fsync(stream.fileno())

    def _read(self) -> dict:
        try:
            state = json.loads(self.path.read_text())
            if (
                state["pricing"] != PRICING
                or state["prior"] != reconciliation(self.root, self.expected_priors)
                or type(state["closed"]) is not bool
            ):
                raise ValueError("Bad ledger metadata")
            attempts = state["attempts"]
            if not isinstance(attempts, list) or len(attempts) > 72:
                raise ValueError("Bad attempts")
            for index, row in enumerate(attempts):
                if (
                    row["id"] != index
                    or row["reserved_usd"] != str(PER_ATTEMPT)
                    or row["run_id"] not in RUN_IDS
                    or row["status"] not in {"reserved", "complete", "failed"}
                ):
                    raise ValueError("Bad reservation")
            if any(sum(a["run_id"] == run for a in attempts) > 12 for run in RUN_IDS):
                raise ValueError("Run cap exceeded")
            return state
        except (OSError, KeyError, ValueError, TypeError):
            raise BudgetStop(
                "Continual ledger missing, corrupt or incompatible"
            ) from None

    def reserve(self, run_id: str) -> int:
        if run_id not in RUN_IDS:
            raise BudgetStop("Unknown funded run")
        with self._locked():
            state = self._read()
            attempts = state["attempts"]
            if state["closed"] or any(a["status"] != "complete" for a in attempts):
                raise BudgetStop("Budget closed or unresolved attempt")
            if (
                len(attempts) >= 72
                or sum(a["run_id"] == run_id for a in attempts) >= 12
                or Decimal(state["prior"]["prior_reserved_usd"])
                + (len(attempts) + 1) * PER_ATTEMPT
                > Decimal("4.90")
            ):
                raise BudgetStop("Original cumulative or per-run allowance reached")
            attempt = len(attempts)
            attempts.append(
                {
                    "id": attempt,
                    "run_id": run_id,
                    "purpose": "online_tutor_labels",
                    "reserved_usd": str(PER_ATTEMPT),
                    "status": "reserved",
                    "reserved_at": _now(),
                }
            )
            self._write(state)
            return attempt

    def finish(self, attempt_id: int, usage: dict | None) -> None:
        with self._locked():
            state = self._read()
            row = state["attempts"][attempt_id]
            if row["status"] != "reserved":
                raise BudgetStop("Attempt already finished")
            valid = isinstance(usage, dict) and all(
                type(usage.get(k)) is int and 0 <= usage[k] <= limit
                for k, limit in [
                    ("counted_input_tokens", 2048),
                    ("input_tokens", 2048),
                    ("output_tokens", 512),
                    ("reasoning_tokens", 512),
                ]
            )
            valid = (
                valid
                and usage["reasoning_tokens"] <= usage["output_tokens"]
                and usage["input_tokens"] == usage["counted_input_tokens"]
            )
            row["finished_at"] = _now()
            row["status"] = "complete" if valid else "failed"
            if valid:
                row["usage"] = usage
                row["generation_estimate_usd_not_verified_billing"] = str(
                    (
                        Decimal("0.75") * usage["input_tokens"]
                        + Decimal("4.50") * usage["output_tokens"]
                    )
                    / Decimal(1000000)
                )
            else:
                state["closed"] = True
            self._write(state)
            if not valid:
                raise BudgetStop(
                    "Uncertain tutor usage; reservation retained and budget closed"
                )


class ContinualTeacher(BudgetedTeacher):
    def request(self, prompt: str, run_id: str) -> tuple[str, int]:
        if (
            not isinstance(prompt, str)
            or not prompt.isascii()
            or len(prompt.encode()) > 6000
            or (self._secret and self._secret in prompt)
        ):
            raise BudgetStop("Invalid tutor request")
        attempt = self.budget.reserve(run_id)
        try:
            request = {"model": MODEL, "input": prompt, "reasoning": {"effort": "none"}}
            count = self._client.responses.input_tokens.count(**request).input_tokens
            if type(count) is not int or not 0 <= count <= 2048:
                raise BudgetStop("Tutor input token cap exceeded")
            response = self._client.responses.create(
                **request, max_output_tokens=512, service_tier="default", store=False
            )
            usage = {
                "counted_input_tokens": count,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "reasoning_tokens": response.usage.output_tokens_details.reasoning_tokens,
            }
            self.budget.finish(attempt, usage)
        # HTTP/auth/prompt exception contents must never reach output.
        except Exception:  # noqa: BLE001
            try:
                self.budget.finish(attempt, None)
            except BudgetStop:
                pass
            self.budget.close()
            raise BudgetStop(
                "Tutor request/count/usage failed; no retry; reservation retained"
            ) from None
        text = response.output_text
        if (
            response.model != MODEL
            or response.status != "completed"
            or response.service_tier != "default"
            or not isinstance(text, str)
            or len(text.encode()) > 8192
            or (self._secret and self._secret in text)
        ):
            self.budget.close()
            raise BudgetStop("Tutor response validation failed; budget closed")
        return text, attempt
