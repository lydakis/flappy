"""Data, budget, safe grammar and frozen evaluation controls for transfer."""

import copy
from collections import defaultdict

import pytest
import torch

from scripts.run_representation_transfer import (
    ARMS,
    DOMAINS,
    SPLITS,
    SequenceStudent,
    calibration_passes,
    encode,
    evaluate,
    execute_program,
    fingerprint,
    generate_cases,
    latent_key,
    probe_examples,
    render,
    simulate,
    tokens,
    training_examples,
)


def test_latent_splits_disjoint_and_answer_not_in_rendered_identifiers():
    seen = set()
    for split in SPLITS:
        cases = generate_cases(split, 99, 48)
        for case in cases:
            key = latent_key(case)
            assert key not in seen
            seen.add(key)
            for domain in DOMAINS:
                text = render(case, domain, 0)
                assert case["id"] not in text
                changed = {**case, "id": "secret-answer=1", "label": 1 - case["label"]}
                assert render(changed, domain, 0) == text
        assert len({latent_key(c) for c in cases}) == 48
    assert generate_cases("train", 99, 8) == generate_cases("train", 99, 8)


def test_representation_budgets_and_joint_template_balance():
    cases = generate_cases("train", 7, 96)
    for arm in ARMS:
        rows = training_examples(cases, arm)
        assert len(rows) == 96
        assert len({r["latent_id"] for r in rows}) == 96
        assert [r["label"] for r in rows] == [c["label"] for c in cases]
        groups = defaultdict(list)
        for row, case in zip(rows, cases, strict=True):
            groups[
                (
                    row["domain"],
                    row["template"],
                    case["velocity_milli"] > 0,
                    case["steps"],
                )
            ].append(row["label"])
        assert all(sum(y) * 2 == len(y) for y in groups.values())
        assert {d: sum(r["domain"] == d for r in rows) for d in DOMAINS} == {
            d: 32 if arm == "mixed" else 96 if arm == d else 0 for d in DOMAINS
        }
    for row in probe_examples(cases):
        assert row["template"] in (
            [0, 1] if row["group"] == "seen_template" else [2, 3]
        )


def test_all_code_forms_match_independent_step_simulator():
    for case in generate_cases("calibration_test", 13, 48):
        for template in range(4):
            result = execute_program(render(case, "code", template))
            assert (
                result
                == case["label"]
                == simulate(
                    case["position_milli"] / 1000,
                    case["velocity_milli"] / 1000,
                    case["steps"],
                )
            )


@pytest.mark.parametrize(
    "program",
    [
        "import os; output(1)",
        "position=__import__('os'); velocity=1; steps=1; output(position>0)",
        "position=1; velocity=1; steps=1; output(open('x'))",
        "position=1; velocity=1; steps=1; output(position.__class__)",
        "position=1; velocity=1; steps=1; output(position**velocity>0)",
        "position=1; velocity=1; steps=1; output(position/velocity>0)",
        "position=1; velocity=1; steps=100; output(position>0)",
        "position=1; position=1; steps=1; output(position>0)",
    ],
)
def test_program_interpreter_rejects_outside_fixed_grammar(program):
    with pytest.raises(ValueError):
        execute_program(program)


def test_heldout_templates_introduce_no_new_symbols_within_domain():
    case = {"position_milli": -1000, "velocity_milli": 700, "steps": 2}
    for domain in DOMAINS:
        trained = set(tokens(render(case, domain, 0))) | set(
            tokens(render(case, domain, 1))
        )
        heldout = set(tokens(render(case, domain, 2))) | set(
            tokens(render(case, domain, 3))
        )
        assert heldout <= trained
        encode([render(case, domain, t) for t in range(4)])


def test_frozen_evaluation_does_not_change_optimizer_or_next_update():
    torch.manual_seed(7)
    a = SequenceStudent()
    b = copy.deepcopy(a)
    oa = torch.optim.Adam(a.parameters(), lr=0.003)
    ob = torch.optim.Adam(b.parameters(), lr=0.003)
    probe = probe_examples(generate_cases("test", 77, 16))
    flipped = copy.deepcopy(probe)
    for row in flipped:
        row["label"] = 1 - row["label"]
    first = evaluate(a, oa, probe, 0)
    second = evaluate(b, ob, flipped, 0)
    assert all(
        x["accuracy"] + y["accuracy"] == 1
        for x, y in zip(first["metrics"], second["metrics"], strict=True)
    )
    assert fingerprint([a.state_dict(), oa.state_dict()]) == fingerprint(
        [b.state_dict(), ob.state_dict()]
    )
    rows = training_examples(generate_cases("train", 13, 32), "mixed")
    inputs = encode([r["input"] for r in rows])
    labels = torch.tensor([r["label"] for r in rows])
    for model, optimizer in [(a, oa), (b, ob)]:
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model(*inputs), labels).backward()
        optimizer.step()
    assert fingerprint([a.state_dict(), oa.state_dict()]) == fingerprint(
        [b.state_dict(), ob.state_dict()]
    )


def test_calibration_requires_all_three_specialists_and_both_source_gates():
    rows = [
        {
            "arm": domain,
            "evaluations": [
                {
                    "metrics": [
                        {"domain": domain, "group": "seen_template", "accuracy": 0.9},
                        {
                            "domain": domain,
                            "group": "heldout_template",
                            "accuracy": 0.8,
                        },
                    ]
                }
            ],
        }
        for domain in DOMAINS
    ]
    assert calibration_passes(rows)
    assert not calibration_passes(rows[:2])
    rows[0]["evaluations"][0]["metrics"][1]["accuracy"] = 0.74
    assert not calibration_passes(rows)
