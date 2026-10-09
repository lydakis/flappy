"""Fail-closed, local-only accounting for the bounded teacher experiment.

This deliberately does not change the repository's general API client. Every
experiment advice/reflection/demo path must receive this one shared client.
Reservations are never refunded, even on an ambiguous network failure.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import stat
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

MODEL = "gpt-5.4-mini-2026-03-17"
PRICING = {
    "model": MODEL,
    "verified_date": "2026-10-08",
    "source": "https://developers.openai.com/api/docs/models/gpt-5.4-mini",
    "input_usd_per_million": "0.75",
    "output_usd_per_million": "4.50",
    "context_tokens": 400_000,
    "max_output_tokens": 2048,
    "max_input_bytes": 24_000,
    "max_attempts": 4,
    "ceiling_usd": "1.50",
    # Cover even the documented 10% regional-processing uplift conservatively.
    "reservation_multiplier": "1.10",
}
PER_ATTEMPT = (
    (
        Decimal(PRICING["input_usd_per_million"]) * PRICING["context_tokens"]
        + Decimal(PRICING["output_usd_per_million"]) * PRICING["max_output_tokens"]
    )
    / Decimal(1_000_000)
    * Decimal(PRICING["reservation_multiplier"])
)


class BudgetStop(RuntimeError):
    """A safe error containing no server response, prompt, or credential."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SharedBudget:
    """Cross-process locked ledger; missing/corrupt state never resets itself."""

    def __init__(self, path: Path):
        self.path = path
        self.lock_path = path.with_suffix(".lock")

    def initialize(self) -> None:
        """Explicit one-time initialization; refuses to replace existing state."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked(), self.path.open("x") as stream:
            json.dump({"pricing": PRICING, "attempts": [], "closed": False}, stream)
            stream.flush()
            os.fsync(stream.fileno())

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self.lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self) -> dict:
        try:
            result = json.loads(self.path.read_text())
            if result["pricing"] != PRICING or type(result["closed"]) is not bool:
                raise ValueError("incompatible ledger")
            attempts = result["attempts"]
            if (
                not isinstance(attempts, list)
                or len(attempts) > PRICING["max_attempts"]
            ):
                raise ValueError("invalid attempts")
            for index, attempt in enumerate(attempts):
                if (
                    attempt["id"] != index
                    or attempt["reserved_usd"] != str(PER_ATTEMPT)
                    or attempt["status"] not in {"reserved", "complete", "failed"}
                ):
                    raise ValueError("invalid reservation")
            return result
        except (OSError, ValueError, KeyError, TypeError):
            raise BudgetStop(
                "Budget ledger missing, corrupt, or incompatible"
            ) from None

    def _write(self, state: dict) -> None:
        temp = self.path.with_suffix(".tmp")
        with temp.open("w") as stream:
            json.dump(state, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, self.path)
        fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def snapshot(self) -> dict:
        with self._locked():
            return self._read()

    def reserve(self, purpose: str) -> int:
        if purpose not in {"advice", "reflection", "demonstrations"}:
            raise BudgetStop("Unknown teacher purpose")
        with self._locked():
            state = self._read()
            attempts = state["attempts"]
            # An incomplete prior attempt is ambiguous; do not send another.
            if state["closed"] or any(a["status"] != "complete" for a in attempts):
                raise BudgetStop("Teacher budget closed or unresolved attempt")
            if len(attempts) >= PRICING["max_attempts"] or (
                len(attempts) + 1
            ) * PER_ATTEMPT > Decimal(PRICING["ceiling_usd"]):
                raise BudgetStop("Teacher attempt/spending limit reached")
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

    def finish(self, attempt_id: int, usage: dict | None) -> None:
        with self._locked():
            state = self._read()
            attempt = state["attempts"][attempt_id]
            if attempt["status"] != "reserved":
                raise BudgetStop("Attempt already finalized")
            valid = isinstance(usage, dict) and all(
                type(usage.get(name)) is int and 0 <= usage[name] <= bound
                for name, bound in (
                    ("input_tokens", PRICING["context_tokens"]),
                    ("output_tokens", PRICING["max_output_tokens"]),
                    ("reasoning_tokens", PRICING["max_output_tokens"]),
                )
            )
            valid = valid and usage["reasoning_tokens"] <= usage["output_tokens"]
            attempt["finished_at"] = _now()
            if not valid:
                attempt["status"] = "failed"
                state["closed"] = True
            else:
                attempt["status"] = "complete"
                attempt["usage"] = usage
                # No cached-token discount: conservative billing estimate.
                cost = (
                    Decimal(PRICING["input_usd_per_million"]) * usage["input_tokens"]
                    + Decimal(PRICING["output_usd_per_million"])
                    * usage["output_tokens"]
                ) / Decimal(1_000_000)
                attempt["estimated_usd_no_cache_discount"] = str(cost)
            self._write(state)
            if not valid:
                raise BudgetStop("Teacher attempt failed; reservation retained")

    def close(self) -> None:
        with self._locked():
            state = self._read()
            state["closed"] = True
            self._write(state)


def validate_key(value: str) -> str:
    """Shape-check an API key without echoing it."""
    if not re.fullmatch(r"sk-[A-Za-z0-9_-]{20,512}", value):
        raise BudgetStop("Credential absent, duplicate, placeholder, or invalid")
    if any(word in value.lower() for word in ("placeholder", "replace", "your_key")):
        raise BudgetStop("Credential is a placeholder")
    return value


def take_env_key(name: str = "OPENAI_API_KEY") -> str:
    """Read a key injected into this job's environment and remove it from the
    environment so no child process inherits it."""
    value = os.environ.pop(name, None)
    if value is None:
        raise BudgetStop("Credential absent from the job environment")
    return validate_key(value.strip())


def read_authorized_key(path: Path) -> str:
    """Parse only the approved untracked regular file, without shell evaluation."""
    tracked = subprocess.run(
        ["git", "-C", str(path.parent), "ls-files", "--error-unmatch", path.name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if tracked.returncode != 1:
        raise BudgetStop("Credential tracking check did not confirm untracked file")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd) as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 16_384:
                raise BudgetStop("Credential file must be a small regular file")
            lines = stream.read().splitlines()
        values = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("export "):
                stripped = stripped[7:].lstrip()
            name, separator, value = stripped.partition("=")
            if separator and name.strip() == "OPENAI_API_KEY":
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                values.append(value)
        if len(values) != 1:
            raise BudgetStop("Credential absent, duplicate, placeholder, or invalid")
        return validate_key(values[0])
    except BudgetStop:
        raise
    except (OSError, ValueError, UnicodeError):
        raise BudgetStop("Credential could not be parsed safely") from None


def _refuse_ambient_auth() -> None:
    # No debug HTTP/header logging, dotenv discovery, redirects or retries.
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    if "OPENAI_CUSTOM_HEADERS" in os.environ:
        raise BudgetStop(
            "Ambient API headers present; refusing ambiguous authentication"
        )


class BudgetedTeacher:
    """One narrow Responses request path shared by demos, advice and reflection."""

    def __init__(self, budget: SharedBudget, client, secret: str = ""):
        self.budget = budget
        self._client = client
        self._secret = secret

    @classmethod
    def from_key_file(cls, budget: SharedBudget, path: Path) -> BudgetedTeacher:
        """Build a client without sending a request or loading other env files."""
        _refuse_ambient_auth()
        return cls.from_key(budget, read_authorized_key(path))

    @classmethod
    def from_key(cls, budget: SharedBudget, key: str) -> BudgetedTeacher:
        """Build a client for an already validated key; sends nothing."""
        _refuse_ambient_auth()
        import httpx2
        from openai import OpenAI

        for name in ("openai", "httpx", "httpx2", "httpcore"):
            logging.getLogger(name).setLevel(logging.CRITICAL + 1)
        client = OpenAI(
            api_key=key,
            admin_api_key="",
            webhook_secret="",
            organization="",
            project="",
            base_url="https://api.openai.com/v1",
            max_retries=0,
            timeout=90.0,
            http_client=httpx2.Client(follow_redirects=False, trust_env=False),
        )
        return cls(budget, client, key)

    def invoke_text(self, messages: list[dict], *, purpose: str = "advice") -> str:
        if not messages or len(messages) > 4:
            raise BudgetStop("Invalid message count")
        for message in messages:
            if (
                set(message) != {"role", "content"}
                or message["role"] not in {"system", "developer", "user"}
                or not isinstance(message["content"], str)
                or (self._secret and self._secret in message["content"])
            ):
                raise BudgetStop("Invalid or unsafe teacher input")
        size = len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
        if size > PRICING["max_input_bytes"]:
            raise BudgetStop("Teacher input byte limit exceeded")
        attempt_id = self.budget.reserve(purpose)
        try:
            response = self._client.responses.create(
                model=MODEL,
                input=messages,
                max_output_tokens=PRICING["max_output_tokens"],
                reasoning={"effort": "low"},
                service_tier="default",
                store=False,
                # No tools, files, previous responses, conversations or background work.
            )
        except Exception:  # noqa: BLE001 - suppress SDK errors containing request data.
            try:
                self.budget.finish(attempt_id, None)
            except BudgetStop:
                pass
            raise BudgetStop("Teacher request failed; reservation retained") from None
        try:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "reasoning_tokens": response.usage.output_tokens_details.reasoning_tokens,
            }
        except (AttributeError, TypeError):
            try:
                self.budget.finish(attempt_id, None)
            except BudgetStop:
                pass
            raise BudgetStop("Teacher usage missing") from None
        self.budget.finish(attempt_id, usage)
        text = response.output_text
        if (
            response.model != MODEL
            or response.status != "completed"
            or not isinstance(text, str)
            or len(text.encode("utf-8")) > 32_768
            or (self._secret and self._secret in text)
        ):
            self.budget.close()
            raise BudgetStop("Teacher response invalid/incomplete; stopping")
        return text

    def reflect(self, messages: list[dict]) -> str:
        return self.invoke_text(messages, purpose="reflection")

    def close(self) -> None:
        self._client.close()
        self._secret = ""
