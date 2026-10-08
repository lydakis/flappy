"""Scientific integrity checks for the offline active-learning mechanism."""

import copy

import numpy as np
import torch

from scripts.run_adaptive_curriculum import (
    BLOCKS,
    HELP_THRESHOLD,
    ORDER,
    Selector,
    Student,
    case,
    dataset,
    evaluate,
    fingerprint,
    oracle,
)


def test_tasks_are_balanced_visible_numeric_rules_and_disjoint_splits():
    identifiers = set()
    for split in ["training", "heldout", "validation_train", "validation_test"]:
        for task in range(6):
            labels = []
            for index in range(12):
                sample = case(task, index, 7, split)
                assert sample["id"] not in identifiers
                identifiers.add(sample["id"])
                features = np.array(sample["features"])
                assert features.shape == (5,)
                assert features[2:].sum() == 1
                assert sample["label"] == oracle(features)
                labels.append(sample["label"])
            assert sum(labels) == 6
    assert case(3, 5, 7) == case(3, 5, 7)


def test_adaptive_choice_responds_to_training_progress_without_changing_arms():
    low = Selector("adaptive", 7)
    high = Selector("adaptive", 7)
    high.observe(4, 0.2)
    _, low_p = low.choose(6)
    _, high_p = high.choose(6)
    assert high_p[4] > low_p[4]
    assert all(p >= 0.1 / 6 for p in high_p)
    assert np.isclose(sum(high_p), 1)
    for arm in ["random", "fixed"]:
        a, b = Selector(arm, 7), Selector(arm, 7)
        b.observe(4, 0.2)
        assert a.choose(6) == b.choose(6)


def test_zero_progress_adaptive_matches_random_common_random_numbers():
    adaptive, random = Selector("adaptive", 7), Selector("random", 7)
    assert [adaptive.choose(n)[0] for n in range(64)] == [
        random.choose(n)[0] for n in range(64)
    ]


def test_fixed_schedule_has_common_warmup_and_equal_task_exposure():
    fixed = Selector("fixed", 7)
    schedule = [fixed.choose(block)[0] for block in range(BLOCKS)]
    assert schedule[:6] == ORDER
    assert [schedule.count(task) for task in range(6)] == [32] * 6


def test_help_remains_available_and_high_confidence_never_receives_label():
    student = Student(7)
    x = np.array(case(0, 0, 7)["features"], dtype=np.float32)
    calls = []

    def teacher(features):
        calls.append(features.copy())
        return oracle(features)

    # Stub only inference for a precise query-boundary check.
    actual = student.probabilities
    student.probabilities = lambda _: np.array(
        [[HELP_THRESHOLD + 0.01, 1 - HELP_THRESHOLD - 0.01]]
    )
    before = fingerprint(student.state())
    result = student.practise(x, teacher)
    assert not result["asked"] and not calls
    assert fingerprint(student.state()) == before
    student.probabilities = actual
    # The policy still asks on an uncertain observation; there is no clock/quota input.
    result = student.practise(x, teacher)
    assert result["asked"] and len(calls) == 1
    assert student.queries == student.updates == len(student.replay_y) == 1


def test_replay_contains_only_requested_labels_and_one_update_per_query():
    student = Student(19)
    asked = []

    def teacher(x):
        label = oracle(x)
        asked.append(label)
        return label

    for n in range(12):
        x = np.array(case(n % 6, n // 6, 19)["features"], dtype=np.float32)
        student.practise(x, teacher)
    assert student.replay_y == asked
    assert student.updates == student.queries == len(asked)


def test_evaluation_is_frozen_and_cannot_change_future_selection_or_training():
    a, b = Student(43), Student(43)
    selector_a, selector_b = Selector("adaptive", 43), Selector("adaptive", 43)
    heldout = dataset("heldout", 999, 4)
    flipped = copy.deepcopy(heldout)
    for sample in flipped:
        sample["label"] = 1 - sample["label"]
    a_before = fingerprint(a.state())
    rng_before = torch.get_rng_state().clone()
    first, second = evaluate(a, heldout), evaluate(b, flipped)
    assert fingerprint(a.state()) == a_before == fingerprint(b.state())
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert np.isclose(first["macro_accuracy"] + second["macro_accuracy"], 1)
    for block in range(8):
        ta, _ = selector_a.choose(block)
        tb, _ = selector_b.choose(block)
        assert ta == tb
        x = np.array(case(ta, block, 43)["features"], dtype=np.float32)
        ra, rb = a.practise(x, oracle), b.practise(x, oracle)
        assert ra == rb
        selector_a.observe(ta, ra["positive_loss_reduction"])
        selector_b.observe(tb, rb["positive_loss_reduction"])
    assert fingerprint(a.state()) == fingerprint(b.state())
    assert selector_a.state() == selector_b.state()
