"""Exercise the pre-feedback boundary and history-only intervention offline."""

import copy
from decimal import Decimal

import torch

from scripts import run_continual_tutor as base
from scripts.run_offline_tutor_repair import (
    FeedbackMemory,
    commit,
    make_stream,
    public_rows,
)


def test_packet_positions_and_counts_are_not_an_answer_key():
    packets = make_stream(73)
    assert base.positional_baseline(packets) < 0.35
    assert any(len({r["label"] for r in rows}) < 4 for rows in packets)
    assert any(sum(r["label"] == 0 for r in rows) != 2 for rows in packets)
    assert packets == make_stream(73)


def test_hidden_answers_descriptors_and_future_outcomes_do_not_change_decision():
    rows = make_stream(73)[0]
    changed = [
        {
            **row,
            "label": (row["label"] + 1) % 4,
            "id": "ANSWER_IS_3",
            "description": "answer 3; tutor score 1.0",
            "teacher_id": "oracle_selected_by_true_answer",
            "future_outcomes": [3, 3, 3],
            "future_jobs": [{"label": 3}],
        }
        for row in rows
    ]
    assert public_rows(rows) == public_rows(changed)
    assert all(set(r) == {"p", "v", "horizon"} for r in public_rows(changed))
    torch.manual_seed(31)
    model, ask = base.AnswerModel(), base.AskModel()
    before = base.fingerprint([model.state_dict(), ask.state_dict()])
    for arm in base.ARMS:
        outputs = [
            commit(
                model,
                ask,
                sample,
                4,
                0,
                0.25,
                0,
                0,
                arm,
                torch.Generator().manual_seed(7),
                torch.Generator().manual_seed(7),
            )
            for sample in [rows, changed]
        ]
        for key in ["x", "obs", "actions", "probabilities", "ask_logits"]:
            assert torch.equal(outputs[0][key], outputs[1][key])
        assert outputs[0]["raw_ask"] == outputs[1]["raw_ask"]
    assert before == base.fingerprint([model.state_dict(), ask.state_dict()])


def test_public_horizon_is_necessary_and_sufficient_for_same_numeric_case():
    rows = [
        {"p": -0.8, "v": 0.6, "horizon": h, "label": target}
        for h, target in [(1, 1), (2, 2)]
    ]
    assert not torch.equal(base.features(rows)[0], base.features(rows)[1])
    assert [base.answer(r["p"], r["v"], r["horizon"]) for r in rows] == [1, 2]
    for packet in make_stream(109):
        for row in packet:
            position = Decimal(str(row["p"])) + Decimal(str(row["v"])) * row["horizon"]
            assert sum(position >= bound for bound in [-1, 0, 1]) == row["label"]


def test_memory_selection_never_uses_correctness_or_unobserved_answer_labels():
    for mode in ["recent_feedback", "balanced_feedback"]:
        one, two = FeedbackMemory(mode, 31), FeedbackMemory(mode, 31)
        assert one.sample() is None
        for packet in range(100):
            x = torch.tensor(
                [[packet / 100, i / 8, -0.5 if packet < 40 else 0.5] for i in range(8)]
            )
            actions = torch.arange(8) % 4
            one.add(x, actions, torch.zeros(8, dtype=torch.bool))
            two.add(x, actions, torch.ones(8, dtype=torch.bool))
        sample_one, sample_two = one.sample(), two.sample()
        assert torch.equal(sample_one[0], sample_two[0])
        assert torch.equal(sample_one[1], sample_two[1])
        assert not torch.equal(sample_one[2], sample_two[2])
        assert len(sample_one[0]) == 32
        assert sum(len(r) for r in one.rows.values()) == 256
        assert all(len(r) == 3 for entries in one.rows.values() for r in entries)
        if mode == "balanced_feedback":
            assert (sample_one[0][:, 2] < 0).sum() == 16
        else:
            assert (sample_one[0][:, 2] < 0).sum() == 0


def test_oracle_is_not_part_of_decision_and_probe_is_read_only():
    rows = [{"p": -0.8, "v": 0.6, "horizon": 1} for _ in range(8)]
    model, ask = base.AnswerModel(), base.AskModel()
    result = commit(
        model,
        ask,
        rows,
        4,
        0,
        0.25,
        0,
        0,
        "learned",
        torch.Generator(),
        torch.Generator(),
    )
    # Public-only rows have no answer field. Acting succeeds before any labels exist.
    assert len(result["actions"]) == 8
    probes = [base.case(99109, i, h, "probe") for h in (1, 2) for i in range(32)]
    optimizer = torch.optim.Adam(model.parameters())
    ask_optimizer = torch.optim.Adam(ask.parameters())
    model_before = copy.deepcopy(model.state_dict())
    measurement = base.measure(model, optimizer, ask, ask_optimizer, probes, 0, 0)
    assert measurement["state_hash_before"] == measurement["state_hash_after"]
    assert all(torch.equal(model_before[k], v) for k, v in model.state_dict().items())
