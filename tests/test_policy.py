import numpy as np

from rl.policy import MaskedCategoricalPolicy, PolicyConfig


def test_masked_policy_respects_mask():
    policy = MaskedCategoricalPolicy(PolicyConfig(state_dim=4, subgoal_dim=4, action_dim=5))
    state = np.zeros(4, dtype=np.float32)
    subgoal = np.zeros(4, dtype=np.float32)
    mask = np.array([1, 0, 0, 0, 0], dtype=np.float32)
    action = policy.sample(state, subgoal, mask=mask, deterministic=False)
    assert action == 0
