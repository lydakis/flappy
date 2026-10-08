"""All tests are offline and use invented credentials/fake transport."""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from types import SimpleNamespace

import pytest

from llm.budgeted_teacher import (
    MODEL,
    PER_ATTEMPT,
    PRICING,
    BudgetedTeacher,
    BudgetStop,
    SharedBudget,
    read_authorized_key,
)

MESSAGES = [{"role": "user", "content": "Synthetic fixture"}]
USAGE = {"input_tokens": 100, "output_tokens": 30, "reasoning_tokens": 10}


def response():
    return SimpleNamespace(
        model=MODEL,
        status="completed",
        output_text="SUBGOAL: choose eligible item\nPLAN: check stock then price",
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=30,
            output_tokens_details=SimpleNamespace(reasoning_tokens=10),
        ),
    )


class FakeClient:
    def __init__(self, budget, fail=False, result=None):
        self.responses = self
        self.budget = budget
        self.fail = fail
        self.result = result
        self.calls = []

    def create(self, **kwargs):
        assert self.budget.snapshot()["attempts"][-1]["status"] == "reserved"
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("Invented sensitive exception body must be suppressed")
        return self.result or response()


@pytest.fixture
def budget(tmp_path):
    result = SharedBudget(tmp_path / "budget.json")
    result.initialize()
    return result


def test_budget_reserves_before_send_and_shared_paths_exhaust(budget):
    from llm.coach import Coach

    fake = FakeClient(budget)
    teacher = BudgetedTeacher(budget, fake)
    coach = Coach(teacher)
    coach.advise(
        task_id="fixture", dom_summary="", recent_actions=[], inventory=[], notes=""
    )
    coach.reflect("fixture", [])
    teacher.reflect(MESSAGES)
    teacher.invoke_text(MESSAGES, purpose="demonstrations")
    assert len(fake.calls) == 4
    with pytest.raises(BudgetStop, match="limit"):
        teacher.invoke_text(MESSAGES)
    assert coach.reflect("fixture", []) == ""
    assert len(fake.calls) == 4
    assert PER_ATTEMPT * 4 < Decimal("1.50") < Decimal(5)
    for call in fake.calls:
        assert call["max_output_tokens"] == 2048
        assert call["model"] == MODEL
        assert call["store"] is False
        assert call["service_tier"] == "default"
        assert "tools" not in call and "previous_response_id" not in call
    snapshot = budget.snapshot()
    assert snapshot["attempts"][0]["estimated_usd_no_cache_discount"] == "0.00021"


def test_network_failure_is_not_retried_and_burns_reservation(budget):
    fake = FakeClient(budget, fail=True)
    teacher = BudgetedTeacher(budget, fake)
    with pytest.raises(BudgetStop) as caught:
        teacher.invoke_text(MESSAGES)
    assert caught.value.__suppress_context__
    assert "sensitive" not in str(caught.value)
    with pytest.raises(BudgetStop):
        BudgetedTeacher(SharedBudget(budget.path), fake).invoke_text(MESSAGES)
    assert len(fake.calls) == 1
    assert budget.snapshot()["attempts"][0]["reserved_usd"] == str(PER_ATTEMPT)


def test_crash_pending_attempt_cannot_resume_or_reset(budget):
    budget.reserve("reflection")
    with pytest.raises(BudgetStop):
        SharedBudget(budget.path).reserve("advice")
    with pytest.raises(FileExistsError):
        budget.initialize()


def test_corrupt_missing_and_changed_price_ledgers_fail_closed(budget):
    budget.path.write_text("{")
    with pytest.raises(BudgetStop):
        budget.reserve("advice")
    budget.path.unlink()
    with pytest.raises(BudgetStop):
        budget.reserve("advice")
    budget.initialize()
    state = budget.snapshot()
    state["pricing"]["input_usd_per_million"] = "0"
    budget.path.write_text(json.dumps(state))
    with pytest.raises(BudgetStop):
        budget.reserve("advice")


def test_concurrent_clients_cannot_double_reserve(budget):
    def reserve(_):
        try:
            return SharedBudget(budget.path).reserve("advice")
        except BudgetStop:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(reserve, range(4)))
    assert results.count(0) == 1
    assert results.count(None) == 3


@pytest.mark.parametrize(
    "invalid",
    [None, {}, {**USAGE, "output_tokens": 2049}, {**USAGE, "input_tokens": -1}],
)
def test_bad_usage_stops_future_requests(budget, invalid):
    attempt = budget.reserve("advice")
    with pytest.raises(BudgetStop):
        budget.finish(attempt, invalid)
    with pytest.raises(BudgetStop):
        budget.reserve("advice")


def test_input_cap_and_credential_guard_precede_reservation(budget):
    fake = FakeClient(budget)
    teacher = BudgetedTeacher(budget, fake, secret="invented-fixture-secret")
    for content in ["x" * (PRICING["max_input_bytes"] + 1), "invented-fixture-secret"]:
        with pytest.raises(BudgetStop):
            teacher.invoke_text([{"role": "user", "content": content}])
    assert fake.calls == []
    assert budget.snapshot()["attempts"] == []


@pytest.mark.parametrize(
    "change",
    [
        {"status": "incomplete"},
        {"model": "different-model"},
        {"output_text": "invented-fixture-secret"},
    ],
)
def test_invalid_response_closes_budget(budget, change):
    result = response()
    for name, value in change.items():
        setattr(result, name, value)
    teacher = BudgetedTeacher(
        budget, FakeClient(budget, result=result), secret="invented-fixture-secret"
    )
    with pytest.raises(BudgetStop):
        teacher.invoke_text(MESSAGES)
    assert budget.snapshot()["closed"]


def test_key_parser_never_executes_env_text_and_rejects_tracked(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = tmp_path / ".env.local"
    key = "sk-" + "x" * 38
    path.write_text(f'OPENAI_API_KEY="{key}"\nUNRELATED=$(touch SHOULD_NOT_EXIST)\n')
    assert read_authorized_key(path) == key
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()
    subprocess.run(["git", "-C", str(tmp_path), "add", path.name], check=True)
    with pytest.raises(BudgetStop, match="tracking"):
        read_authorized_key(path)


def test_key_parser_rejects_symlinks_duplicates_and_interpolation(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = tmp_path / ".env.local"
    for text in [
        "OPENAI_API_KEY=${OTHER}",
        "OPENAI_API_KEY=sk-short",
        "OPENAI_API_KEY=x\nOPENAI_API_KEY=y",
    ]:
        path.write_text(text)
        with pytest.raises(BudgetStop):
            read_authorized_key(path)
    path.unlink()
    path.symlink_to(tmp_path / "missing")
    with pytest.raises(BudgetStop):
        read_authorized_key(path)


def test_real_sdk_does_not_retry_500_or_follow_redirects(budget, monkeypatch):
    import httpx2

    import llm.budgeted_teacher as module

    monkeypatch.delenv("OPENAI_CUSTOM_HEADERS", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://invalid.example")
    monkeypatch.setattr(
        module, "read_authorized_key", lambda path: "sk-invented-fixture"
    )
    original_client = httpx2.Client
    calls = []

    def handle(request):
        calls.append(request)
        assert str(request.url) == "https://api.openai.com/v1/responses"
        return httpx2.Response(500, json={"error": {"message": "fixture failure"}})

    class FakeHTTPClient(original_client):
        def __init__(self, **kwargs):
            assert kwargs == {"follow_redirects": False, "trust_env": False}
            super().__init__(transport=httpx2.MockTransport(handle), **kwargs)

    monkeypatch.setattr(httpx2, "Client", FakeHTTPClient)
    teacher = BudgetedTeacher.from_key_file(budget, budget.path)
    try:
        with pytest.raises(BudgetStop):
            teacher.invoke_text(MESSAGES)
        assert len(calls) == 1
        assert teacher._client.max_retries == 0
        assert budget.snapshot()["closed"]
    finally:
        teacher.close()


def test_ambient_auth_headers_block_before_key_read(budget, monkeypatch):
    import llm.budgeted_teacher as module

    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: invented")

    def forbidden_read(path):
        raise AssertionError("must not read credential")

    monkeypatch.setattr(module, "read_authorized_key", forbidden_read)
    with pytest.raises(BudgetStop, match="Ambient"):
        BudgetedTeacher.from_key_file(budget, budget.path)
