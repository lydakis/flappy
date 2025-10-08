import numpy as np

from rl.rnd_ppo_agent import LearnerConfig, PPORNDLearner


def test_sample_action_respects_mask():
    learner = PPORNDLearner(
        learner_config=LearnerConfig(
            feature_dim=4,
            subgoal_dim=2,
            max_actions=4,
            rollout_size=8,
            minibatch_size=4,
        )
    )
    state = np.zeros(4, dtype=np.float32)
    subgoal = np.zeros(2, dtype=np.float32)
    mask = np.array([1, 0, 0, 0], dtype=np.float32)
    sample = learner.sample_action_with_context(state, subgoal, mask, action_count=2)
    assert sample.action == 0
