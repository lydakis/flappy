import numpy as np

from agents.hybrid import HybridAgent
from envs.browsergym_client import make_planner_action
from flappy.macro_registry import MacroRegistry
from flappy.interfaces import MaskDelta
from llm.coach import CoachDirective


class DummyEnv:
    pass


class DummyCoach:
    pass


def build_agent() -> HybridAgent:
    env = DummyEnv()
    coach = DummyCoach()
    agent = HybridAgent(env=env, coach=coach, learner=None, memory=None)
    return agent


def make_raw_obs() -> dict:
    strings = [
        "INPUT",
        "checkbox",
        "ch0",
        "value",
        "on",
        "Alpha",
        "BR",
        "",
        "INPUT",
        "checkbox",
        "ch1",
        "value",
        "on",
        "Beta",
        "BUTTON",
        "subbtn",
        "class",
        "secondary-action",
        "Submit",
    ]
    return {
        "goal": "Select Alpha, Beta and click Submit.",
        "dom_object": {"strings": strings},
    }


def test_submit_guardrail_blocks_until_targets_met():
    agent = build_agent()
    raw_obs = make_raw_obs()
    agent._extract_checkbox_targets(raw_obs)
    actions = [
        make_planner_action("click", selector="#ch0"),
        make_planner_action("click", selector="#subbtn"),
        make_planner_action("click", selector="#ch1"),
    ]
    inventory = agent._inventory_strings(actions)
    agent._identify_submit_selectors(actions)

    state_vec = np.zeros(agent.state_encoder.dim, dtype=np.float32)
    decision = agent._resolve_masks(
        state_vec,
        len(actions),
        inventory,
        actions,
    )
    assert decision.final[1] == 0.0
    assert decision.guardrail_applied is True
    assert agent._guardrail_submit_locked is True

    agent._register_action(actions[0])
    agent._register_action(actions[2])

    decision_after = agent._resolve_masks(
        state_vec,
        len(actions),
        inventory,
        actions,
    )
    assert decision_after.final[1] == 1.0
    assert agent._targets_completed() is True


def test_plan_registration_populates_macro_registry():
    registry = MacroRegistry()
    agent = build_agent()
    agent.macro_registry = registry
    agent.current_inventory = ["0: click #submit"]
    agent._handle_plan_sketch("Click(#submit)")
    stats = registry.stats()
    assert stats
    assert agent._current_macro_name in stats


def test_checkbox_toggle_detection_resets_consumed():
    agent = build_agent()
    raw_obs = make_raw_obs()
    agent._extract_checkbox_targets(raw_obs)
    action = make_planner_action("click", selector="#ch0")

    agent._register_action(action)
    assert "#ch0" in agent._checked_selectors
    assert "#ch0" in agent._consumed_selectors

    agent._register_action(action)
    assert "#ch0" not in agent._checked_selectors
    assert "#ch0" not in agent._consumed_selectors


def test_consumed_selectors_masked_from_policy():
    agent = build_agent()
    raw_obs = make_raw_obs()
    agent._extract_checkbox_targets(raw_obs)
    actions = [
        make_planner_action("click", selector="#ch0"),
        make_planner_action("click", selector="#ch1"),
    ]
    inventory = agent._inventory_strings(actions)
    state_vec = np.zeros(agent.state_encoder.dim, dtype=np.float32)

    decision_initial = agent._resolve_masks(state_vec, len(actions), inventory, actions)
    assert np.allclose(decision_initial.final, 1.0)

    agent._register_action(actions[0])

    decision_after = agent._resolve_masks(state_vec, len(actions), inventory, actions)
    assert np.isclose(decision_after.final[0], 0.0)
    assert np.isclose(decision_after.final[1], 1.0)
    assert decision_after.guardrail_applied is True


def test_non_target_checkboxes_masked_out():
    agent = build_agent()
    agent._checkbox_targets = {"#ch0"}
    actions = [
        make_planner_action("click", selector="#ch0"),
        make_planner_action("click", selector="#ch1"),
        make_planner_action("click", selector="#subbtn"),
    ]
    inventory = agent._inventory_strings(actions)
    agent._identify_submit_selectors(actions)
    state_vec = np.zeros(agent.state_encoder.dim, dtype=np.float32)

    decision = agent._resolve_masks(state_vec, len(actions), inventory, actions)
    assert np.isclose(decision.final[0], 1.0)
    assert np.isclose(decision.final[1], 0.0)
    assert np.isclose(decision.final[2], 0.0)
    assert decision.guardrail_applied is True


def test_submit_unlocked_after_targets_completed():
    agent = build_agent()
    raw_obs = make_raw_obs()
    agent._extract_checkbox_targets(raw_obs)
    actions = [
        make_planner_action("click", selector="#ch0"),
        make_planner_action("click", selector="#ch1"),
        make_planner_action("click", selector="#subbtn"),
    ]
    inventory = agent._inventory_strings(actions)
    agent._identify_submit_selectors(actions)
    agent.current_inventory = inventory

    mask_delta = MaskDelta(allow=["#ch0", "#ch1"], block=["#subbtn"])
    agent.current_directive = CoachDirective(
        subgoal="Select targets",
        mask_delta=mask_delta,
    )

    state_vec = np.zeros(agent.state_encoder.dim, dtype=np.float32)
    locked_decision = agent._resolve_masks(state_vec, len(actions), inventory, actions)
    assert np.isclose(locked_decision.final[2], 0.0)
    assert agent._guardrail_submit_locked is True

    agent._register_action(actions[0])
    agent._register_action(actions[1])
    assert agent._targets_completed() is True
    assert "#subbtn" not in [pattern.lower() for pattern in agent.current_directive.mask_delta.block]

    unlocked_decision = agent._resolve_masks(state_vec, len(actions), inventory, actions)
    assert np.isclose(unlocked_decision.final[0], 0.0)
    assert np.isclose(unlocked_decision.final[1], 0.0)
    assert np.isclose(unlocked_decision.final[2], 1.0)
    assert agent._guardrail_submit_locked is False
    allow_patterns = set(agent.current_directive.mask_delta.allow)
    assert any(pattern in allow_patterns for pattern in ("#subbtn", "click #subbtn"))
