"""Checks for bounded, outcome-independent harder-menu selection."""

import copy

import numpy as np
import torch

from scripts.run_adaptive_curriculum import Selector, Student, evaluate, fingerprint
from scripts.run_harder_curriculum import (
    CALIBRATION_SEEDS,
    CANDIDATES,
    NAMESPACES,
    first_qualifying,
    make_case,
    oracle,
    qualifies,
    score,
)


def metric(values):
    return {
        "macro_accuracy": float(np.mean(values)),
        "per_task": [{"teacher_free_accuracy": v} for v in values],
    }


def qualifying_rows(candidate):
    return [
        {
            "candidate": candidate,
            "seed": seed,
            "warmup": metric([0.9, 0.6, 0.9, 0.6, 0.9, 0.6]),
            "supervised_final": metric([0.98, 0.9, 0.98, 0.9, 0.98, 0.9]),
        }
        for seed in CALIBRATION_SEEDS
    ]


def test_nonlinear_cases_are_balanced_in_margin_and_split_disjoint():
    identifiers, observations = set(), set()
    for candidate in CANDIDATES:
        for split in NAMESPACES:
            for task in range(6):
                labels = []
                for index in range(8):
                    sample = make_case(candidate["id"], task, index, 7, split)
                    features = np.array(sample["features"])
                    assert features.shape == (5,)
                    assert features[2:].sum() == 1
                    assert np.argmax(features[2:]) == task // 2
                    margin = abs(score(features, candidate["id"]))
                    lo, hi = (0.2, 2) if task % 2 == 0 else (0.02, 0.15)
                    assert lo <= margin <= hi
                    assert sample["label"] == oracle(features, candidate["id"])
                    assert sample["id"] not in identifiers
                    assert tuple(features) not in observations
                    identifiers.add(sample["id"])
                    observations.add(tuple(features))
                    labels.append(sample["label"])
                assert labels == [0, 1] * 4
    assert make_case("wavy", 3, 5, 7) == make_case("wavy", 3, 5, 7)


def test_gate_rejects_ceiling_unlearnability_and_single_weak_task():
    row = qualifying_rows("quadratic")[0]
    assert qualifies(row["warmup"], row["supervised_final"])["passed"]
    assert not qualifies(metric([0.99] * 6), metric([1.0] * 6))["passed"]
    assert not qualifies(row["warmup"], metric([0.7] * 6))["passed"]
    assert not qualifies(row["warmup"], metric([1, 1, 1, 1, 1, 0.79]))["passed"]
    # Macro headroom alone is insufficient: two near-boundary tasks must qualify.
    assert not qualifies(metric([0.5, 0.81, 0.5, 0.81, 0.5, 0.7]), metric([1] * 6))[
        "passed"
    ]


def test_selection_requires_both_seeds_and_uses_fixed_order_not_arm_outcomes():
    quadratic = qualifying_rows("quadratic")
    wavy = qualifying_rows("wavy")
    assert first_qualifying(quadratic[:1]) is None
    assert first_qualifying([quadratic[0], quadratic[0]]) is None
    for row in quadratic:
        row["adaptive_advantage"] = -1
    for row in wavy:
        row["adaptive_advantage"] = 1
    assert first_qualifying(wavy + quadratic) == "quadratic"
    quadratic[1]["supervised_final"] = metric([0.6] * 6)
    assert first_qualifying(wavy + quadratic) == "wavy"


def test_changed_final_probe_cannot_change_training_or_task_selection():
    a, b = Student(43), Student(43)
    sa, sb = Selector("adaptive", 43), Selector("adaptive", 43)
    probe = [
        make_case("quadratic", t, i, 900, "heldout") for t in range(6) for i in range(4)
    ]
    flipped = copy.deepcopy(probe)
    for example in flipped:
        example["label"] = 1 - example["label"]
    before = fingerprint(a.state())
    rng = torch.get_rng_state().clone()
    first, second = evaluate(a, probe), evaluate(b, flipped)
    assert np.isclose(first["macro_accuracy"] + second["macro_accuracy"], 1)
    assert fingerprint(a.state()) == fingerprint(b.state()) == before
    assert torch.equal(rng, torch.get_rng_state())

    def teacher(x):
        return oracle(x, "quadratic")

    for block in range(8):
        ta, _ = sa.choose(block)
        tb, _ = sb.choose(block)
        assert ta == tb
        x = np.array(
            make_case("quadratic", ta, block, 43)["features"], dtype=np.float32
        )
        ra, rb = a.practise(x, teacher), b.practise(x, teacher)
        assert ra == rb
        sa.observe(ta, ra["positive_loss_reduction"])
        sb.observe(tb, rb["positive_loss_reduction"])
    assert fingerprint(a.state()) == fingerprint(b.state())
    assert sa.state() == sb.state()
