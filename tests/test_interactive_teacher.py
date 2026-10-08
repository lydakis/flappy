"""Offline budget and original live-coach path checks; no credentials or API IO."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flappy.interfaces import MaskDelta
from llm.budgeted_teacher import (
    MODEL,
    PER_ATTEMPT,
    BudgetedTeacher,
    BudgetStop,
    SharedBudget,
)
from llm.coach import Coach, CoachDirective
from llm.linked_budget import LinkedBudget
from llm.prompts import COACH_SYSTEM_PROMPT
from scripts.run_interactive_teacher import (
    HORIZON,
    RULE,
    RecordingClient,
    build,
    episode,
    weight_delta,
)


@pytest.fixture
def linked(tmp_path):
    previous = SharedBudget(tmp_path / "previous.json")
    previous.initialize()
    for _ in range(4):
        attempt = previous.reserve("demonstrations")
        previous.finish(
            attempt, {"input_tokens": 100, "output_tokens": 50, "reasoning_tokens": 20}
        )
    previous.close()
    new = LinkedBudget(tmp_path / "new.json", previous.path)
    new.initialize()
    return new


def test_linked_ledger_retains_prior_reservations_and_caps_ten(linked):
    prior_bytes = linked.previous.read_bytes()
    for n in range(10):
        assert linked.reserve("advice") == n
        linked.finish(
            n, {"input_tokens": 100, "output_tokens": 50, "reasoning_tokens": 20}
        )
    with pytest.raises(BudgetStop, match="Cumulative"):
        linked.reserve("reflection")
    assert Decimal(linked.snapshot()["prior"]["reserved_usd"]) == 4 * PER_ATTEMPT
    assert linked.previous.read_bytes() == prior_bytes
    with pytest.raises(FileExistsError):
        linked.initialize()


def test_changed_prior_ledger_blocks_paid_attempts(linked):
    state = json.loads(linked.previous.read_text())
    state["unexpected_change"] = True
    linked.previous.write_text(json.dumps(state))
    with pytest.raises(BudgetStop, match="changed"):
        linked.reserve("advice")


def test_missing_live_ledger_does_not_reset(linked):
    linked.path.unlink()
    with pytest.raises(BudgetStop):
        linked.reserve("advice")


def test_two_live_clients_cannot_reserve_concurrently(linked):
    def reserve(_):
        try:
            return LinkedBudget(linked.path, linked.previous).reserve("advice")
        except BudgetStop:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(reserve, range(4)))
    assert results.count(0) == 1
    assert results.count(None) == 3


class Transport:
    def __init__(self, linked, fail=False):
        self.responses = self
        self.linked = linked
        self.calls = []
        self.fail = fail

    def create(self, **kwargs):
        assert self.linked.snapshot()["attempts"][-1]["status"] == "reserved"
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("sensitive fixture exception must be suppressed")
        return SimpleNamespace(
            model=MODEL,
            status="completed",
            output_text="SUBGOAL: click left\nMASK_ALLOW: #left\nPLAN: Click(#left)",
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=50,
                output_tokens_details=SimpleNamespace(reasoning_tokens=20),
            ),
        )


class FixtureEnv:
    def __init__(self):
        self.t = 0
        self.rows = []

    def observation(self):
        return {
            "dom_text": f"Current cue: {'left' if self.t < 8 else 'right'}",
            "goal": RULE,
            "extra_element_properties": {
                side: {"tag": "button", "selector": "#" + side}
                for side in ["left", "right"]
            },
        }

    def reset(self, **kwargs):
        self.t = 0
        self.rows = []
        return self.observation(), {}

    def encode_observation(self, obs):
        return obs

    def step(self, action):
        correct = action.selector == ("#left" if self.t < 8 else "#right")
        self.rows.append({"correct": correct})
        self.t += 1
        return (
            self.observation(),
            float(correct),
            self.t == HORIZON,
            False,
            {"success": correct},
        )


def test_original_coach_prompts_masks_and_periodic_triggers_share_gate(
    linked, tmp_path
):
    transport = Transport(linked)
    recording = RecordingClient(BudgetedTeacher(linked, transport), tmp_path)
    coach = Coach(recording)
    agent = build(FixtureEnv(), 7, coach, "fixture")
    result = episode(agent)
    assert [e["after_actions"] for e in result["help_events"]] == [0, 11, 21]
    assert [e["reasons"] for e in result["help_events"]] == [
        ["episode_start"],
        ["periodic"],
        ["periodic"],
    ]
    assert len(transport.calls) == 3
    assert all(
        call["input"][0]["content"] == COACH_SYSTEM_PROMPT for call in transport.calls
    )
    assert "Blackboard (driver signals):" in transport.calls[1]["input"][1]["content"]
    assert len(result["help_events"][0]["inventory"]) == 7
    assert all(e["allowed_actions"] == [0] for e in result["actions"])
    assert coach.reflect("fixture", []) == ""
    assert len(transport.calls) == 3  # reflection denied before another reservation
    assert all(a["purpose"] == "advice" for a in linked.snapshot()["attempts"])


def test_live_transport_failure_has_no_retry_or_reflection_bypass(linked, tmp_path):
    transport = Transport(linked, fail=True)
    coach = Coach(RecordingClient(BudgetedTeacher(linked, transport), tmp_path))
    with pytest.raises(BudgetStop):
        coach.advise(
            task_id="fixture", dom_summary="", inventory=[], recent_actions=[], notes=""
        )
    assert coach.reflect("fixture", []) == ""
    with pytest.raises(BudgetStop):
        coach.advise(
            task_id="fixture", dom_summary="", inventory=[], recent_actions=[], notes=""
        )
    assert len(transport.calls) == 1
    assert linked.snapshot()["closed"]


def test_heuristic_entropy_and_stuck_requests_are_not_policy_actions():
    class Fixed:
        def advise(self, **kwargs):
            return CoachDirective(subgoal="choose")

    agent = build(FixtureEnv(), 7, Fixed(), "fixture")
    agent.entropy_window.extend([2.1] * 5)
    assert agent._should_request_guidance(1, {})
    assert agent.next_reasons == ["entropy"]
    agent.entropy_window.clear()
    assert agent._should_request_guidance(1, {"stuck": True})
    assert agent.next_reasons == ["environment_stuck"]
    actions, _ = agent._action_catalog(agent.env.observation())
    assert "ask" not in [a.name for a in actions]


def test_singleton_coach_mask_allows_ppo_updates_without_actor_learning():
    class Fixed:
        def advise(self, **kwargs):
            return CoachDirective(
                subgoal="click left", mask_delta=MaskDelta(allow=["#left"])
            )

    agent = build(FixtureEnv(), 7, Fixed(), "fixture")
    before = copy.deepcopy(agent.learner.policy.state_dict())
    critic_before = copy.deepcopy(agent.learner.value_net.state_dict())
    outcome = episode(agent)
    assert agent.learner.policy_updates == 3
    assert agent.learner.optimizer_steps == 24
    assert all(e["entropy"] == 0 and e["log_prob"] == 0 for e in outcome["actions"])
    assert weight_delta(before, agent.learner.policy) == 0
    assert weight_delta(critic_before, agent.learner.value_net) > 0
    for name, tensor in before.items():
        assert torch.equal(tensor, agent.learner.policy.state_dict()[name])
    assert np.isfinite(agent.learner.last_update["value_loss"])
