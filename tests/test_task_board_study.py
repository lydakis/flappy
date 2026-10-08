"""Information boundaries and actual learning chronology for the board study."""

import copy
from collections import Counter

import numpy as np
import pytest
import torch

from scripts import run_continual_tutor as base
from scripts.run_task_board_study import (
    Allocator,
    History,
    Student,
    counterfactual,
    draw_case,
    latent_key,
    learn_packet,
    observation,
    partition,
    public_view,
    valid_schedules,
)


def cases(horizon=1):
    values = [(-1.8, -0.2), (-0.8, 0.2), (0.2, 0.2), (1.2, 0.2)] * 2
    return [
        {"p": p, "v": v, "horizon": horizon, "label": base.answer(p, v, horizon)}
        for p, v in values
    ]


def test_schedule_catalog_is_uniform_sampleable_and_has_paid_returns():
    schedules = valid_schedules()
    assert len(schedules) > 6 and len({tuple(s) for s in schedules}) == len(schedules)
    for schedule in schedules:
        assert Counter(schedule) == {1: 4, 2: 4}
        runs = [h for i, h in enumerate(schedule) if i == 0 or h != schedule[i - 1]]
        assert min(runs.count(1), runs.count(2)) >= 2


def test_current_only_never_observes_future_and_board_stops_at_two_cards():
    schedule = [1, 2, 1, 2, 1, 2, 1, 2]
    changed = [1, 1, 2, 2, 2, 1, 1, 2]
    assert public_view(schedule, 7, False) == public_view(changed, 7, False)
    view = public_view(schedule, 7, True)
    assert view["future"] == [
        {"skill": 2, "pay": 1.0, "available_in": 25},
        {"skill": 1, "pay": 0.25, "available_in": 57},
    ]
    far_future = [*schedule[:3], *reversed(schedule[3:])]
    assert view == public_view(far_future, 7, True)
    assert public_view(schedule, 255, True)["future"] == []
    assert public_view(schedule, 64, False)["past_contracts"] == [1, 2]


def test_private_descriptors_outcomes_and_recommendations_cannot_enter_allocator():
    schedule = valid_schedules()[0]
    view = public_view(schedule, 33, True)
    changed = copy.deepcopy(view)
    changed.update(
        {
            "label": 3,
            "future_outcomes": [3] * 8,
            "candidate_examples": cases(),
            "teacher_recommendation": "B",
        }
    )
    changed["current"].update({"id": "answer3", "label": 3, "hidden_phase": 2})
    for card in changed["future"]:
        card.update({"answers": [3] * 8, "position": 1.5, "estimated_success": 1.0})
    first = observation(view, 33, 4, 0, History())
    second = observation(changed, 33, 4, 0, History())
    assert first.shape == (47,) and torch.equal(first, second)
    allocator = Allocator(True)
    assert torch.equal(allocator(first)[0], allocator(second)[0])


def test_initial_policy_has_no_prescribed_lesson_and_no_tutor_mask_is_exact():
    view = public_view(valid_schedules()[0], 0, True)
    obs = observation(view, 0, 4, 0, History())
    assert torch.allclose(
        Allocator(True)(obs)[0].softmax(-1), torch.tensor([0.475, 0.475, 0.025, 0.025])
    )
    assert torch.equal(
        Allocator(False)(obs)[0].softmax(-1), torch.tensor([0.5, 0.5, 0.0, 0.0])
    )


def test_case_partitions_and_unique_sampling_exclude_cross_source_reuse():
    rng, used = np.random.default_rng(17), set()
    for region in range(5):
        for label in range(4):
            row = draw_case(rng, label, 2, region, used, "not-visible")
            assert partition(latent_key(row)) == region
            assert row["label"] == label == base.answer(row["p"], row["v"], 2)
    assert len(used) == 20


def test_packet_cannot_read_labels_before_commit_and_practice_adds_no_income():
    committed = []

    class GuardedCase(dict):
        def __getitem__(self, key):
            if key == "label":
                assert committed, "Grading leaked into pre-feedback decision"
            return super().__getitem__(key)

    rows = [GuardedCase(r) for r in cases()]
    student = Student(17)
    before = base.fingerprint(student.model.state_dict())
    result = learn_packet(
        student,
        rows,
        buy_labels=True,
        paid=False,
        on_commit=lambda actions: committed.append(actions),
    )
    assert committed == [result["committed_answers"]]
    assert result["paid_income"] == 0 and result["optimizer_steps"] == 4
    assert result["teacher_labels"] == [r["label"] for r in cases()]
    assert result["correct"] == [
        a == r["label"] for a, r in zip(committed[0], cases(), strict=True)
    ]
    assert base.fingerprint(student.model.state_dict()) != before
    assert student.updates == 4 and student.memory.seen == {0: 8}
    assert all(len(record) == 3 for record in student.memory.rows[0])


def test_practice_and_paid_use_identical_correctness_learning_but_distinct_income():
    practice, paid = Student(21), Student(21)
    a = learn_packet(practice, cases(2), buy_labels=False, paid=False)
    b = learn_packet(paid, cases(2), buy_labels=False, paid=True)
    assert a["committed_answers"] == b["committed_answers"]
    assert a["paid_income"] == 0
    assert b["paid_income"] == sum(b["correct"])
    assert base.fingerprint(practice.model.state_dict()) == base.fingerprint(
        paid.model.state_dict()
    )


def test_paid_labels_change_learning_without_retroactively_rescuing_answers():
    no_query, queried = Student(31), Student(31)
    a = learn_packet(no_query, cases(), buy_labels=False, paid=False)
    b = learn_packet(queried, cases(), buy_labels=True, paid=False)
    assert (
        a["committed_answers"] == b["committed_answers"]
        and a["correct"] == b["correct"]
    )
    assert base.fingerprint(no_query.model.state_dict()) != base.fingerprint(
        queried.model.state_dict()
    )


def test_buffer_uses_only_public_horizon_and_previously_experienced_records():
    student = Student(23)
    first = [{**r, "private_task_id": 2, "hidden_phase": "B"} for r in cases()]
    a = learn_packet(student, first, buy_labels=False, paid=False)
    assert a["replay_skill_counts"] == [] and student.memory.seen == {0: 8}
    b = learn_packet(student, cases(2), buy_labels=False, paid=True)
    assert b["replay_skill_counts"] == [32, 0]
    c = learn_packet(student, cases(), buy_labels=False, paid=True)
    assert c["replay_skill_counts"] == [16, 16]


def test_four_action_allocator_is_updated_by_returns_with_mask_consistent():
    torch.manual_seed(83)
    allocator = Allocator(True)
    obs = observation(public_view(valid_schedules()[0], 0, True), 0, 4, 0, History())
    before = base.fingerprint(allocator.policy.state_dict())
    with torch.no_grad():
        logits, value = allocator(obs)
    rollout = [
        {
            "obs": obs,
            "action": i % 4,
            "log_prob": float(logits.log_softmax(-1)[i % 4]),
            "value": float(value),
            "reward": float(i % 4 == 1),
        }
        for i in range(16)
    ]
    base.update_ask(
        allocator, torch.optim.Adam(allocator.parameters(), lr=0.001), rollout, 0.0
    )
    assert base.fingerprint(allocator.policy.state_dict()) != before


@pytest.mark.parametrize("board", [False, True])
def test_counterfactual_holds_observed_past_fixed(board):
    schedule = valid_schedules()[0]
    alternative = counterfactual(schedule, 0)
    assert alternative is not None and alternative[0] == schedule[0]
    a = observation(public_view(schedule, 0, board), 0, 4, 0, History())
    b = observation(public_view(alternative, 0, board), 0, 4, 0, History())
    assert torch.equal(a[:37], b[:37])
    assert torch.equal(a, b) == (not board)
