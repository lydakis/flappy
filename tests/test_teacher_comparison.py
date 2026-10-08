import copy

import numpy as np
import pytest
import torch

from llm.budgeted_teacher import BudgetStop
from scripts.run_teacher_comparison import (
    build,
    correct_action,
    evaluate,
    features,
    make_case,
    parse_labels,
)


def test_task_and_features_only_contain_visible_fields():
    case = {
        "minimum_quality": 5,
        "items": [
            {"in_stock": 0, "quality": 9, "price": 1},
            {"in_stock": 1, "quality": 4, "price": 3},
            {"in_stock": 1, "quality": 5, "price": 7},
            {"in_stock": 1, "quality": 8, "price": 5},
        ],
    }
    assert correct_action(case) == 3
    np.testing.assert_allclose(
        features(case), [0.5, 0, 0.9, 0.1, 1, 0.4, 0.3, 1, 0.5, 0.7, 1, 0.8, 0.5]
    )
    for seed in range(100):
        for variant in ["train", "new_prices", "new_threshold"]:
            generated = make_case(seed, variant)
            assert 0 <= correct_action(generated) < 4
            assert features(generated).shape == (13,)


def test_held_out_evaluation_does_not_update_or_consume_rng():
    learner = build(7)
    random_state = torch.get_rng_state().clone()
    state = copy.deepcopy(learner.policy.state_dict())
    result = evaluate(learner, {"iid": [make_case(900000 + n) for n in range(20)]})
    assert result["learning_hash_before"] == result["learning_hash_after"]
    assert torch.equal(random_state, torch.get_rng_state())
    assert learner.policy_updates == learner.rnd_updates == 0
    for key, value in state.items():
        assert torch.equal(value, learner.policy.state_dict()[key])


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"actions":[4],"strategy":"rule"}',
        '{"actions":[true],"strategy":"rule"}',
        '{"actions":[0],"strategy":"rule","code":"x"}',
    ],
)
def test_teacher_output_parser_rejects_bad_labels(text):
    with pytest.raises(BudgetStop):
        parse_labels(text, expected=1)
