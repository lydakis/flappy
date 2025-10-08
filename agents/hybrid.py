"""Hybrid agent: RL driver with LLM coach guidance."""

from __future__ import annotations

import logging
import random
import uuid
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from envs.browsergym_client import BrowserGymEnvWrapper, PlannerAction, make_planner_action
from envs.selectors import extract_interactive_selectors
from llm.coach import Coach, CoachDirective
from llm.memory import JsonlMemoryStore, MemoryEntry, retrieve_top_k
from rl.context import SubgoalEncoder
from rl.features import DomTextHasher
from rl.rnd_ppo_agent import PPORNDLearner, SampleOutput

logger = logging.getLogger(__name__)


class HybridAgent:
    """RL-only actuator with LLM-issued advisory context."""

    def __init__(
        self,
        env: BrowserGymEnvWrapper,
        coach: Coach,
        learner: Optional[PPORNDLearner] = None,
        memory: Optional[JsonlMemoryStore] = None,
        *,
        planner_interval: int = 10,
        max_steps: int = 200,
        reflexion_enabled: bool = True,
        reflexion_read_only: bool = False,
        stuck_entropy_threshold: float = 2.0,
        stuck_window: int = 5,
    ) -> None:
        self.env = env
        self.coach = coach
        self.learner = learner
        self.memory = memory
        self.planner_interval = planner_interval
        self.max_steps = max_steps
        self.reflexion_enabled = reflexion_enabled
        self.reflexion_read_only = reflexion_read_only
        self.subgoal_encoder = SubgoalEncoder(dim=learner.config.subgoal_dim if learner else 256)
        feature_dim = learner.config.feature_dim if learner else 2048
        self.state_encoder = DomTextHasher(dim=feature_dim)
        self.recent_actions: Deque[str] = deque(maxlen=20)
        self.current_subgoal = ""
        self.current_subgoal_vec = np.zeros(self.subgoal_encoder.dim, dtype=np.float32)
        self.current_directive: Optional[CoachDirective] = None
        self.current_mask: Optional[np.ndarray] = None
        self.current_inventory: List[str] = []
        self.interventions = 0
        self.entropy_window: Deque[float] = deque(maxlen=stuck_window)
        self.stuck_entropy_threshold = stuck_entropy_threshold
        self.max_actions = learner.max_actions if learner else 20

    def run_episode(self, task_id: str) -> Dict[str, float]:
        obs, info = self.env.reset(return_info=True)
        observation = self.env.encode_observation(obs)
        reflections = self._retrieve_reflections(task_id)
        episode_id = str(uuid.uuid4())
        episode_trace: List[str] = []
        intrinsic_total = 0.0

        action_candidates, inventory_strings = self._action_catalog(observation["dom_text"])
        if not action_candidates:
            action_candidates = self._default_action_catalog()
            inventory_strings = self._inventory_strings(action_candidates)
        self.current_inventory = inventory_strings
        self._request_guidance(
            task_id=task_id,
            dom_summary=observation["dom_text"],
            inventory=inventory_strings,
            reflections=reflections,
        )

        for step in range(self.max_steps):
            action_candidates, inventory_strings = self._action_catalog(observation["dom_text"])
            if not action_candidates:
                action_candidates = self._default_action_catalog()
                inventory_strings = self._inventory_strings(action_candidates)
            self.current_inventory = inventory_strings
            self._refresh_mask(inventory_strings)

            state_vec = self._state_vector(observation)
            subgoal_vec = self.current_subgoal_vec.copy()
            mask_vec = self.current_mask.copy() if self.current_mask is not None else None
            action_idx, sample = self._select_action(
                state_vec,
                subgoal_vec,
                mask_vec,
                len(action_candidates),
            )
            planner_action = action_candidates[action_idx]
            action_desc = self._describe_action(action_idx, planner_action)
            self.recent_actions.append(action_desc)
            episode_trace.append(action_desc)

            obs, reward, terminated, truncated, info = self.env.step(planner_action)
            observation = self.env.encode_observation(obs)
            entropy = sample.entropy if sample is not None else info.get("policy_entropy", 0.0)
            self._update_entropy(entropy)

            if self._should_request_guidance(step, info):
                self._request_guidance(
                    task_id=task_id,
                    dom_summary=observation["dom_text"],
                    inventory=inventory_strings,
                    reflections=reflections,
                )

            next_state_vec = self._state_vector(observation)
            next_subgoal_vec = self.current_subgoal_vec.copy()

            if self.learner is not None and sample is not None:
                intrinsic_reward = self.learner.compute_intrinsic(next_state_vec)
                intrinsic_total += intrinsic_reward
                self.learner.observe_transition(
                    state=state_vec,
                    subgoal=subgoal_vec,
                    sample=sample,
                    reward=reward,
                    intrinsic=intrinsic_reward,
                    done=bool(terminated or truncated),
                    next_state=next_state_vec,
                    next_subgoal=next_subgoal_vec,
                )

            if terminated or truncated:
                break

        success = bool(info.get("success", False))
        reward_total = float(info.get("episode_reward", 0.0))

        if self.reflexion_enabled and self.memory and not self.reflexion_read_only:
            reflection_text = self.coach.reflect(task_id, episode_trace)
            if reflection_text:
                entry = MemoryEntry(
                    task_id=task_id,
                    episode_id=episode_id,
                    success=success,
                    notes=reflection_text,
                    selectors_used=[],
                    failure_modes=[],
                    subgoal=self.current_subgoal,
                    mask_allow=self._mask_items(True),
                    mask_block=self._mask_items(False),
                )
                self.memory.append(entry)

        return {
            "success": float(success),
            "reward": reward_total,
            "steps": step + 1,
            "coach_interventions": float(self.interventions),
            "intrinsic_reward": intrinsic_total,
            "trace": list(episode_trace),
        }

    def _select_action(
        self,
        state_vec: np.ndarray,
        subgoal_vec: np.ndarray,
        mask: Optional[np.ndarray],
        action_count: int,
    ) -> tuple[int, Optional[SampleOutput]]:
        """Choose an action index, respecting mask constraints."""
        valid_indices = self._valid_indices(mask, action_count)
        if not valid_indices:
            valid_indices = list(range(action_count))
        if self.learner is None:
            return random.choice(valid_indices), None
        try:
            sample = self.learner.sample_action_with_context(
                state_vec, subgoal_vec, mask, action_count
            )
            action_idx = int(sample.action)
        except Exception as exc:  # pragma: no cover - learner optional
            logger.warning("Learner sample failed, falling back to random: %s", exc)
            self.learner = None
            return random.choice(valid_indices), None
        if action_idx not in valid_indices:
            action_idx = random.choice(valid_indices)
        return action_idx, sample

    def _should_request_guidance(self, step: int, info: Dict[str, float]) -> bool:
        if step > 0 and step % self.planner_interval == 0:
            return True
        if self.entropy_window and np.mean(self.entropy_window) > self.stuck_entropy_threshold:
            return True
        if info.get("stuck", False):
            return True
        return False

    def _request_guidance(
        self,
        *,
        task_id: str,
        dom_summary: str,
        inventory: Sequence[str],
        reflections: str,
    ) -> None:
        directive = self.coach.advise(
            task_id=task_id,
            dom_summary=dom_summary,
            recent_actions=self.recent_actions,
            inventory=inventory,
            notes=reflections,
        )
        self.current_directive = directive
        self.current_subgoal = directive.subgoal
        self.current_subgoal_vec = self.subgoal_encoder.encode(directive.subgoal)
        self.current_inventory = list(inventory)
        self.current_mask = self._build_mask(directive, inventory)
        self.interventions += 1

    def _build_mask(self, directive: CoachDirective, inventory: Sequence[str]) -> np.ndarray:
        mask = np.ones(len(inventory), dtype=np.float32)
        if directive.mask_allow:
            allowed = self._match_items(directive.mask_allow, inventory)
            if allowed:
                mask[:] = 0.0
                mask[allowed] = 1.0
        if directive.mask_block:
            blocked = self._match_items(directive.mask_block, inventory)
            mask[blocked] = 0.0
        return mask

    def _valid_indices(self, mask: Optional[np.ndarray], action_count: int) -> List[int]:
        if mask is None:
            return list(range(action_count))
        return [idx for idx in range(action_count) if idx < len(mask) and mask[idx] > 0.0]

    def _update_entropy(self, entropy: float) -> None:
        self.entropy_window.append(float(entropy))

    def _mask_items(self, allowed: bool) -> List[str]:
        if self.current_directive is None:
            return []
        if allowed:
            return list(self.current_directive.mask_allow)
        return list(self.current_directive.mask_block)

    def _default_action_catalog(self) -> List[PlannerAction]:
        """Return a small fallback action set; to be replaced by DOM-derived candidates."""
        return self._default_navigation_actions()

    def _inventory_strings(self, actions: Iterable[PlannerAction]) -> List[str]:
        inventory = []
        for idx, action in enumerate(actions):
            detail = action.name
            if action.name in {"click", "type"} and action.selector:
                detail = f"{action.name} {action.selector}"
            elif action.name == "press" and action.key:
                detail = f"{action.name} {action.key}"
            elif action.name == "wait" and action.wait_ms is not None:
                detail = f"{action.name} {action.wait_ms}ms"
            inventory.append(f"{idx}: {detail}")
        return inventory

    def _match_items(self, patterns: Iterable[str], inventory: Sequence[str]) -> List[int]:
        matches: List[int] = []
        for pattern in patterns:
            pattern_lower = pattern.lower()
            if pattern_lower.isdigit():
                matches.append(int(pattern_lower))
                continue
            for idx, item in enumerate(inventory):
                if pattern_lower in item.lower():
                    matches.append(idx)
        return sorted(set(idx for idx in matches if 0 <= idx < len(inventory)))

    def _describe_action(self, index: int, action: PlannerAction) -> str:
        if action.name == "wait":
            return f"{index}: wait({action.wait_ms}ms)"
        if action.name == "press":
            return f"{index}: press({action.key})"
        if action.name == "scroll":
            return f"{index}: scroll({action.direction})"
        if action.name == "click":
            return f"{index}: click({action.selector})"
        if action.name == "type":
            return f"{index}: type({action.selector})"
        return f"{index}: {action.name}"

    def _retrieve_reflections(self, task_id: str) -> str:
        if not self.memory or not self.reflexion_enabled:
            return ""
        entries = self.memory.load()
        top = retrieve_top_k(entries, task_id, query=task_id, k=3)
        return "\n".join(entry.notes for entry in top)

    def _refresh_mask(self, inventory: Sequence[str]) -> None:
        if self.current_directive is None:
            self.current_mask = None
            return
        self.current_mask = self._build_mask(self.current_directive, inventory)

    def _state_vector(self, observation: Dict[str, str]) -> np.ndarray:
        return self.state_encoder.encode({"dom_text": observation.get("dom_text", "")})

    def _action_catalog(self, dom_text: str) -> Tuple[List[PlannerAction], List[str]]:
        selector_budget = max(1, self.max_actions // 2)
        selectors = extract_interactive_selectors(dom_text, max_candidates=selector_budget)
        actions: List[PlannerAction] = []
        for selector in selectors:
            actions.append(make_planner_action("click", selector=selector))
            actions.append(make_planner_action("type", selector=selector, text="test"))
        actions.extend(self._default_navigation_actions())
        if len(actions) > self.max_actions:
            actions = actions[: self.max_actions]
        inventory = self._inventory_strings(actions)
        return actions, inventory

    def _default_navigation_actions(self) -> List[PlannerAction]:
        return [
            make_planner_action("wait", wait_ms=200),
            make_planner_action("scroll", direction="down"),
            make_planner_action("scroll", direction="up"),
            make_planner_action("press", key="Tab"),
            make_planner_action("press", key="Enter"),
        ]
