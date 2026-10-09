"""The tutor ledger reserves before sending, caps spend and fails closed. Offline."""

import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from llm.budgeted_teacher import BudgetStop
from llm.tutor_ledger import DEFAULT_MODEL, LedgerTutorClient, TutorLedger

SECRET = "sk-" + "x" * 40


def response(model=DEFAULT_MODEL, status="completed", text="Hint.\nANSWER: 4"):
    return SimpleNamespace(
        model=model,
        status=status,
        output_text=text,
        usage=SimpleNamespace(
            input_tokens=120,
            output_tokens=40,
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )


class FakeResponses:
    def __init__(self, ledger, result=None, fail=False):
        self.ledger, self.result, self.fail, self.calls = ledger, result, fail, []

    def create(self, **kwargs):
        assert self.ledger.snapshot()["attempts"][-1]["status"] == "reserved"
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError(f"transport error echoing {SECRET}")
        return self.result or response()


def client(ledger, run_cap=0.05, **kwargs):
    fake = FakeResponses(ledger, **kwargs)
    tutor = LedgerTutorClient(ledger, SimpleNamespace(responses=fake), SECRET)
    tutor.bind_run("progress-s0-test", run_cap)
    return tutor, fake


@pytest.fixture
def ledger(tmp_path):
    result = TutorLedger(tmp_path / "tutor-ledger.json")
    result.initialize()
    return result


def test_reserves_before_send_and_stops_at_the_run_allowance(ledger):
    tutor, fake = client(ledger, run_cap=float(ledger.per_attempt * 3))
    for _ in range(3):
        assert tutor.request("What is 2 + 2?", "hint") == "Hint.\nANSWER: 4"
    with pytest.raises(BudgetStop, match="allowance"):
        tutor.request("What is 2 + 2?", "hint")
    assert len(fake.calls) == 3
    assert fake.calls[0]["store"] is False and fake.calls[0]["max_output_tokens"] == 768
    totals = ledger.totals("progress-s0-test")
    assert totals["calls"] == 3 and 0 < totals["estimated_usd"] < totals["reserved_usd"]


def test_total_ceiling_is_five_dollars_across_runs(ledger):
    state = ledger.snapshot()
    allowed = int(Decimal("5.00") / ledger.per_attempt)
    state["attempts"] = [
        {
            "id": i,
            "run_id": f"old-{i % 3}",
            "purpose": "hint",
            "reserved_usd": str(ledger.per_attempt),
            "status": "complete",
        }
        for i in range(allowed)
    ]
    ledger._write(state)
    tutor, fake = client(ledger, run_cap=5.0)
    with pytest.raises(BudgetStop, match="allowance"):
        tutor.request("hi", "hint")
    assert not fake.calls


def test_transport_failure_closes_ledger_without_leaking_details(ledger):
    tutor, _ = client(ledger, fail=True)
    with pytest.raises(BudgetStop) as info:
        tutor.request("hi", "explanation")
    assert SECRET not in str(info.value) and info.value.__cause__ is None
    state = ledger.snapshot()
    assert state["closed"] and state["attempts"][0]["status"] == "failed"
    with pytest.raises(BudgetStop, match="closed"):
        tutor.request("hi", "hint")


def test_incomplete_reply_is_billed_but_returns_none(ledger):
    tutor, _ = client(ledger, result=response(status="incomplete"))
    assert tutor.request("hi", "worked_example") is None
    assert ledger.snapshot()["attempts"][0]["status"] == "complete"
    assert not ledger.snapshot()["closed"]


@pytest.mark.parametrize(
    "prompt", ["x" * 4001, "café", f"leak {SECRET}"], ids=["long", "ascii", "secret"]
)
def test_unsafe_prompts_are_refused_before_reserving(ledger, prompt):
    tutor, fake = client(ledger)
    with pytest.raises(BudgetStop):
        tutor.request(prompt, "hint")
    assert not fake.calls and not ledger.snapshot()["attempts"]


def test_tampered_or_mismatched_ledger_fails_closed(ledger, tmp_path):
    with pytest.raises(BudgetStop):
        TutorLedger(ledger.path, "gpt-5-mini-2025-08-07").snapshot()
    state = json.loads(ledger.path.read_text())
    state["pricing"]["total_ceiling_usd"] = "500.00"
    ledger.path.write_text(json.dumps(state))
    with pytest.raises(BudgetStop, match="corrupt"):
        ledger.snapshot()
    with pytest.raises(FileExistsError):
        ledger.initialize()
    with pytest.raises(BudgetStop):
        TutorLedger(tmp_path / "missing.json").snapshot()
