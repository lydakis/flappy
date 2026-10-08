import numpy as np
import torch

from rl.rnd_ppo_agent import LearnerConfig, PPORNDLearner
from scripts.run_local_learning import (
    CueEnv,
    OfflineCoach,
    TinyTransformer,
    WindowAgent,
)


def test_final_cue_observation_has_no_hidden_answer():
    env = CueEnv(delay=3)
    final = []
    for seed in [100, 101]:
        env.next_seed = seed
        env.reset()
        for _ in range(3):
            obs, _, _, _, _ = env.step(env.actions()[0])
        final.append(obs["dom_text"])
    assert final[0] == final[1]


def test_window_is_idempotent_and_clears_between_episodes():
    env = CueEnv(delay=3)
    learner = PPORNDLearner(
        learner_config=LearnerConfig(feature_dim=256, subgoal_dim=8, max_actions=2)
    )
    agent = WindowAgent(env, OfflineCoach(), learner=learner, window=4)
    agent._reset_episode_state()
    obs, _ = env.reset()
    first = agent._state_vector(obs)
    assert np.array_equal(first, agent._state_vector(obs))
    assert len(agent.history) == 1
    agent._reset_episode_state()
    assert agent.history == []


def test_transformer_has_no_dropout_and_obeys_mask():
    model = TinyTransformer(64, 4, 2, 8)
    state = torch.zeros(1, 256)
    goal = torch.zeros(1, 8)
    mask = torch.tensor([[1.0, 0.0]])
    first = model(state, goal, mask)
    assert torch.equal(first, model(state, goal, mask))
    assert torch.isneginf(first[0, 1])
