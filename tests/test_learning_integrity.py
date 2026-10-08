"""Regression checks for the real HybridAgent -> PPO/RND call path."""

import copy

import numpy as np
import pytest
import torch

from agents.hybrid import HybridAgent
from envs.browsergym_client import make_planner_action
from eval.harness import EvalConfig, evaluate_agent
from llm.coach import CoachDirective
from rl.rnd_ppo_agent import LearnerConfig, PPORNDLearner


class OfflineCoach:
    def advise(self, **kwargs):
        return CoachDirective(subgoal="choose")


class EpisodeEnv:
    def __init__(self, terminal=True):
        self.terminal = terminal

    def reset(self, **kwargs):
        return {"dom_text": "cue left"}, {}

    def encode_observation(self, obs):
        return obs

    def step(self, action):
        return (
            {"dom_text": "done"},
            1.0,
            self.terminal,
            False,
            {"success": True, "episode_reward": 1.0},
        )


def make_agent(terminal=True):
    torch.manual_seed(7)
    learner = PPORNDLearner(
        learner_config=LearnerConfig(
            feature_dim=8,
            subgoal_dim=4,
            hidden_dim=8,
            max_actions=2,
            rollout_size=2,
            minibatch_size=2,
            policy_epochs=1,
        )
    )
    agent = HybridAgent(
        EpisodeEnv(terminal), OfflineCoach(), learner=learner, max_steps=1
    )
    actions = [
        make_planner_action("click", selector="#left"),
        make_planner_action("click", selector="#right"),
    ]
    agent._action_catalog = lambda obs: (actions, agent._inventory_strings(actions))
    return agent, learner


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_same(left, right)
    else:
        assert a == b


def snapshot(learner):
    return copy.deepcopy(
        {
            "policy": learner.policy.state_dict(),
            "value": learner.value_net.state_dict(),
            "rnd": learner.rnd_module.state_dict(),
            "optimizer": learner.optimizer.state_dict(),
            "rnd_optimizer": learner.rnd_optimizer.state_dict(),
            "steps": learner.total_steps,
            "rollout": [vars(t) for t in learner.rollout],
            "count": learner._count,
            "mean": learner._running_mean,
            "var": learner._running_var,
            "policy_updates": learner.policy_updates,
            "optimizer_steps": learner.optimizer_steps,
            "rnd_updates": learner.rnd_updates,
            "last_update": learner.last_update,
        }
    )


def test_frozen_evaluation_preserves_all_learning_state():
    agent, learner = make_agent()
    before = snapshot(learner)
    evaluate_agent(agent, lambda: None, {"id": "fixture"}, EvalConfig(True, 4, False))
    assert_same(before, snapshot(learner))


def test_learner_failure_is_not_silent_random_success(monkeypatch):
    agent, learner = make_agent()

    def fail(*args, **kwargs):
        raise RuntimeError("injected sampling failure")

    monkeypatch.setattr(learner, "sample_action_with_context", fail)
    with pytest.raises(RuntimeError, match="injected sampling failure"):
        agent.run_episode("fixture")
    assert agent.learner is learner


def test_sampler_never_enables_actions_outside_catalog():
    _, learner = make_agent()
    with torch.no_grad():
        learner.policy.action_head.weight.zero_()
        learner.policy.action_head.bias.copy_(torch.tensor([0.0, 100.0]))
    sample = learner.sample_action_with_context(
        np.zeros(8, np.float32), np.zeros(4, np.float32), None, 1, deterministic=True
    )
    assert sample.action == 0
    assert sample.mask[1] == 0


def test_zero_mask_is_hard_constraint_even_at_extreme_logits():
    _, learner = make_agent()
    with torch.no_grad():
        learner.policy.action_head.weight.zero_()
        learner.policy.action_head.bias.copy_(torch.tensor([0.0, 100.0]))
    sample = learner.sample_action_with_context(
        np.zeros(8, np.float32),
        np.zeros(4, np.float32),
        np.array([1.0, 0.0], np.float32),
        2,
        deterministic=True,
    )
    assert sample.action == 0


def test_agent_step_cap_ends_return_sequence():
    agent, learner = make_agent(terminal=False)
    agent.run_episode("fixture")
    assert learner.rollout[-1].done is True


def test_running_variance_is_variance_not_sum_of_squares():
    _, learner = make_agent()
    for value in [1.0, 2.0, 3.0, 4.0] * 100:
        learner._update_running_stats(value)
    assert learner._running_var == pytest.approx(1.25, abs=0.01)


def test_updates_only_after_rollout_and_frozen_intrinsic_cannot_train():
    agent, learner = make_agent()
    before = copy.deepcopy(learner.policy.state_dict())
    agent.run_episode("fixture")
    assert learner.policy_updates == 0
    assert_same(before, learner.policy.state_dict())
    agent.run_episode("fixture")
    assert learner.policy_updates == 1
    assert learner.optimizer_steps == 1
    assert learner.rnd_updates == 2
    assert any(
        not torch.equal(before[k], v) for k, v in learner.policy.state_dict().items()
    )
    agent.set_training(False)
    frozen = snapshot(learner)
    learner.compute_intrinsic(np.ones(8, np.float32))
    assert_same(frozen, snapshot(learner))


def test_frozen_mode_restored_after_failure(monkeypatch):
    agent, learner = make_agent()

    def fail(*args, **kwargs):
        assert learner.training is False
        raise RuntimeError("environment failure")

    monkeypatch.setattr(agent, "run_episode", fail)
    with pytest.raises(RuntimeError, match="environment failure"):
        evaluate_agent(
            agent, lambda: None, {"id": "fixture"}, EvalConfig(True, 1, False)
        )
    assert agent.training is True
    assert learner.training is True
    assert agent.reflexion_read_only is False


def test_checkpoint_preserves_rnd_stats_and_update_counters(tmp_path):
    agent, learner = make_agent()
    agent.run_episode("fixture")
    agent.run_episode("fixture")
    path = tmp_path / "learner.pt"
    learner.save(str(path))
    _, restored = make_agent()
    restored.load(str(path))
    assert_same(snapshot(learner), snapshot(restored))
