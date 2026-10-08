"""No-network accounting and learning-integrity checks for the continual pilot."""

import copy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from llm.budgeted_teacher import BudgetStop
from llm.continual_budget import (
    MODEL,
    PER_ATTEMPT,
    RUN_IDS,
    ContinualBudget,
    ContinualTeacher,
)
from scripts.run_continual_tutor import (
    AnswerModel,
    AskModel,
    answer,
    ask_observation,
    case,
    features,
    fingerprint,
    measure,
    parse_answers,
    prompt_for,
    stream,
    update_answers,
    update_ask,
)


@pytest.fixture
def budget(tmp_path):
    expected = {}
    for name in ("prior-a.json", "prior-b.json"):
        target = tmp_path / name
        target.write_text(
            json.dumps(
                {
                    "closed": True,
                    "attempts": [
                        {
                            "status": "complete",
                            "reserved_usd": "2.00",
                            "estimated_usd_no_cache_discount": "0.01",
                        }
                    ],
                }
            )
        )
        expected[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    result = ContinualBudget(
        tmp_path / "new-budget.json", tmp_path, expected_priors=expected
    )
    result.initialize()
    return result


def good_usage():
    return {
        "counted_input_tokens": 100,
        "input_tokens": 100,
        "output_tokens": 20,
        "reasoning_tokens": 0,
    }


def test_new_ledger_preserves_old_holds_and_caps_all_six_runs(budget):
    old = {name: (budget.root / name).read_bytes() for name in budget.expected_priors}
    for run in sorted(RUN_IDS):
        for _ in range(12):
            attempt = budget.reserve(run)
            budget.finish(attempt, good_usage())
        with pytest.raises(BudgetStop):
            budget.reserve(run)
    state = budget.snapshot()
    assert len(state["attempts"]) == 72
    assert Decimal(state["prior"]["prior_reserved_usd"]) + PER_ATTEMPT * 72 == Decimal(
        "4.42577920"
    )
    assert all((budget.root / name).read_bytes() == raw for name, raw in old.items())
    assert state["prior"]["holds_released_usd"] == "0"


def test_pending_failure_and_changed_prior_cannot_be_bypassed(budget):
    first = budget.reserve("learned-7")
    with pytest.raises(BudgetStop):
        budget.reserve("heuristic-7")
    with pytest.raises(BudgetStop):
        budget.finish(first, None)
    with pytest.raises(BudgetStop):
        budget.reserve("learned-19")
    assert budget.snapshot()["closed"]
    prior = budget.root / next(iter(budget.expected_priors))
    prior.write_text("{}")
    with pytest.raises(BudgetStop):
        budget.snapshot()


def test_concurrent_clients_cannot_send_two_unsettled_requests(budget):
    def reserve(_):
        try:
            return ContinualBudget(
                budget.path, budget.root, expected_priors=budget.expected_priors
            ).reserve("learned-7")
        except BudgetStop:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(reserve, range(4)))
    assert sum(v is not None for v in values) == 1


class FakeClient:
    def __init__(self, count=100, fail=False):
        self.count_value, self.fail = count, fail
        self.count_calls, self.generation_calls = 0, 0
        self.responses = SimpleNamespace(
            input_tokens=SimpleNamespace(count=self.count), create=self.create
        )

    def count(self, **kwargs):
        assert kwargs["model"] == MODEL and kwargs["reasoning"] == {"effort": "none"}
        self.count_calls += 1
        return SimpleNamespace(input_tokens=self.count_value)

    def create(self, **kwargs):
        self.generation_calls += 1
        assert (
            kwargs["max_output_tokens"] == 512
            and kwargs["service_tier"] == "default"
            and kwargs["store"] is False
        )
        assert "tools" not in kwargs and "previous_response_id" not in kwargs
        if self.fail:
            raise RuntimeError("sensitive transport detail must not escape")
        return SimpleNamespace(
            model=MODEL,
            status="completed",
            service_tier="default",
            output_text='{"answers":[0,1,2,3,0,1,2,3]}',
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=20,
                output_tokens_details=SimpleNamespace(reasoning_tokens=0),
            ),
        )


def test_live_path_counts_tokens_and_validates_usage_with_no_network(budget):
    fake = FakeClient()
    result, attempt = ContinualTeacher(budget, fake).request(
        "eight numeric problems", "learned-7"
    )
    assert parse_answers(result) == [0, 1, 2, 3] * 2
    assert fake.count_calls == fake.generation_calls == 1 and attempt == 0
    assert budget.snapshot()["attempts"][0]["status"] == "complete"


@pytest.mark.parametrize("count,fail", [(2049, False), (100, True)])
def test_too_many_input_tokens_or_network_failure_stops_without_retry(
    budget, count, fail
):
    fake = FakeClient(count, fail)
    with pytest.raises(BudgetStop) as caught:
        ContinualTeacher(budget, fake).request("eight problems", "learned-7")
    assert "sensitive" not in str(caught.value)
    assert fake.count_calls == 1 and fake.generation_calls == int(fail)
    assert budget.snapshot()["closed"]
    assert len(budget.snapshot()["attempts"]) == 1


def test_labels_ids_and_future_jobs_never_enter_model_observation_or_prompt():
    rows = [case(101, i, 1, "calibration") for i in range(8)]
    changed = [
        {
            **r,
            "label": (r["label"] + 1) % 4,
            "id": "secret-answer",
            "future_jobs": [999],
        }
        for r in rows
    ]
    assert torch.equal(features(rows), features(changed))
    assert prompt_for(rows) == prompt_for(changed)
    obs = ask_observation(features(rows), torch.full((8, 4), 0.25), 4, 0, 0.25, 0, 0)
    assert obs.shape == (61,)
    assert [r["label"] for r in rows] == [0, 1, 2, 3] * 2
    assert all(r["label"] == answer(r["p"], r["v"], r["horizon"]) for r in rows)


@pytest.mark.parametrize(
    "text",
    [
        '{"answers":[0]}',
        '{"answers":[0,1,2,3,0,1,2,4]}',
        '{"answers":[0,1,2,3,0,1,2,true]}',
        "not json",
    ],
)
def test_malformed_teacher_reply_never_becomes_training_data(text):
    with pytest.raises(ValueError):
        parse_answers(text)


def test_actual_tutor_labels_change_online_answer_update():
    torch.manual_seed(101)
    a = AnswerModel()
    b = copy.deepcopy(a)
    rows = [case(101, i, 1, "calibration") for i in range(8)]
    x = features(rows)
    with torch.no_grad():
        logits, values = a(x)
        actions = logits.argmax(-1)
        old_logs = logits.log_softmax(-1)[torch.arange(8), actions]
    for model, labels in [
        (a, [r["label"] for r in rows]),
        (b, [(r["label"] + 1) % 4 for r in rows]),
    ]:
        update_answers(
            model,
            torch.optim.Adam(model.parameters(), lr=0.003),
            x,
            actions,
            old_logs,
            values,
            torch.zeros(8),
            labels,
        )
    assert fingerprint(a.state_dict()) != fingerprint(b.state_dict())


def test_ask_policy_updates_from_return_and_is_not_a_fixed_gate():
    torch.manual_seed(101)
    a = AskModel()
    b = copy.deepcopy(a)
    obs = torch.zeros(61)
    with torch.no_grad():
        logits, value = a(obs)
    positive = [
        {
            "obs": obs,
            "action": i % 2,
            "log_prob": float(logits.log_softmax(-1)[i % 2]),
            "value": float(value),
            "reward": float(i % 2),
        }
        for i in range(16)
    ]
    negative = [{**r, "reward": 1 - r["reward"]} for r in positive]
    before = fingerprint(a.state_dict())
    update_ask(a, torch.optim.Adam(a.parameters(), lr=0.001), positive, 0)
    update_ask(b, torch.optim.Adam(b.parameters(), lr=0.001), negative, 0)
    assert fingerprint(a.state_dict()) != before
    assert fingerprint(a.state_dict()) != fingerprint(b.state_dict())


def test_diagnostic_probes_preserve_answer_ask_and_optimizers():
    torch.manual_seed(7)
    model, ask = AnswerModel(), AskModel()
    opt = torch.optim.Adam(model.parameters())
    ask_opt = torch.optim.Adam(ask.parameters())
    probes = [case(999, i, h, "probe") for h in (1, 2) for i in range(8)]
    before = fingerprint(
        [model.state_dict(), opt.state_dict(), ask.state_dict(), ask_opt.state_dict()]
    )
    result = measure(model, opt, ask, ask_opt, probes, 0, 0)
    assert result["state_hash_before"] == result["state_hash_after"] == before


def test_packet_position_alone_cannot_reveal_class_to_ask_model():
    packets = stream(7, "calibration")
    majority_correct = sum(
        max(Counter(rows[position]["label"] for rows in packets).values())
        for position in range(8)
    )
    accuracy = majority_correct / (len(packets) * 8)
    assert accuracy < 0.35, f"Packet position alone predicts {accuracy:.1%} of answers"
