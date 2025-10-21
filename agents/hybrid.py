"""Hybrid agent: RL driver with LLM coach guidance."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import random
import uuid
from pathlib import Path
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from envs.browsergym_client import BrowserGymEnvWrapper, PlannerAction, make_planner_action
from envs.selectors import extract_interactive_selectors
from llm.coach import Coach, CoachDirective
from llm.ideas import Idea, IdeaStore
from llm.memory import JsonlMemoryStore, MemoryEntry, retrieve_top_k
from rl.context import SubgoalEncoder
from rl.features import DomTextHasher
from rl.rnd_ppo_agent import PPORNDLearner, SampleOutput
from flappy.blackboard import AffordanceHint, BlackboardState, DriverEvent
from flappy.dsl import DSLNode, DSLVerb, count_nodes, is_kernel_only, make_leaf
from flappy.extract import DocumentExtractor
from flappy.interfaces import GoalContext, MaskDecision, combine_masks
from flappy.macro_registry import Macro, MacroRegistry
from flappy.memory import Note, NoteStore
from flappy.rag import Answer, SimpleRAG
from flappy.synth import PlanSynthesiser, Sketch
from flappy.verify import PlanVerifier, VerificationResult

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
        note_store: Optional[NoteStore] = None,
        rag: Optional[SimpleRAG] = None,
        extractor: Optional[DocumentExtractor] = None,
        macro_registry: Optional[MacroRegistry] = None,
        synthesiser: Optional[PlanSynthesiser] = None,
        verifier: Optional[PlanVerifier] = None,
        idea_store: Optional[IdeaStore] = None,
        ddl_inject: bool = False,
        ddl_top_k: int = 3,
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
        self.current_inventory: List[str] = []
        self.interventions = 0
        self.entropy_window: Deque[float] = deque(maxlen=stuck_window)
        self.stuck_entropy_threshold = stuck_entropy_threshold
        self.max_actions = learner.max_actions if learner else 20
        self._debug_obs_dumped = False
        self.blackboard = BlackboardState()
        self.current_goal = GoalContext()
        self.mask_decision = MaskDecision()
        self._checkbox_targets: set[str] = set()
        self._checked_selectors: set[str] = set()
        self._submit_selectors: set[str] = set()
        self._last_goal_text = ""
        self._label_to_selector: Dict[str, str] = {}
        self._guardrail_submit_locked = False
        self._mask_source_counts: Dict[str, int] = {}
        self._mask_iou_sum = 0.0
        self._mask_iou_count = 0
        self._coach_mask_steps = 0
        self._policy_mask_steps = 0
        self._submit_guardrail_steps = 0
        self._last_mask_source = "none"
        self.note_store = note_store
        self.rag = rag or (SimpleRAG() if note_store else None)
        self.extractor = extractor or (DocumentExtractor() if note_store else None)
        self.idea_store = idea_store
        self.ddl_inject = ddl_inject
        self.ddl_top_k = ddl_top_k
        self.macro_registry = macro_registry or MacroRegistry()
        self._notes_episode_count = 0
        self.current_plan: Optional[DSLNode] = None
        self.synthesiser = synthesiser or PlanSynthesiser()
        self.verifier = verifier or PlanVerifier()
        self._plan_node_count = 0
        self._plan_kernel_only = True
        self._plan_verified: Optional[bool] = None
        self._current_macro_name: Optional[str] = None
        self._macro_usage_pending = False
        self._consumed_selectors: set[str] = set()

    def run_episode(self, task_id: str) -> Dict[str, float]:
        obs, info = self.env.reset(return_info=True)
        observation = self.env.encode_observation(obs)
        reflections = self._retrieve_reflections(task_id)
        episode_id = str(uuid.uuid4())
        episode_trace: List[str] = []
        intrinsic_total = 0.0
        self.interventions = 0
        self.blackboard.clear()
        self.mask_decision = MaskDecision()
        self._checkbox_targets.clear()
        self._checked_selectors.clear()
        self._submit_selectors.clear()
        self._last_goal_text = ""
        self._label_to_selector.clear()
        self._guardrail_submit_locked = False
        self._consumed_selectors.clear()
        self._mask_source_counts = {}
        self._mask_iou_sum = 0.0
        self._mask_iou_count = 0
        self._coach_mask_steps = 0
        self._policy_mask_steps = 0
        self._submit_guardrail_steps = 0
        self._last_mask_source = "none"
        self._notes_episode_count = 0
        self.current_plan = None
        self._plan_node_count = 0
        self._plan_kernel_only = True
        self._plan_verified = None
        self._current_macro_name = None
        self._macro_usage_pending = False

        action_candidates, inventory_strings = self._action_catalog(obs)
        if not action_candidates:
            action_candidates = self._default_action_catalog()
            inventory_strings = self._inventory_strings(action_candidates)
        self.current_inventory = inventory_strings
        self._request_guidance(
            task_id=task_id,
            dom_summary=observation["dom_text"],
            inventory=inventory_strings,
            reflections=reflections,
            raw_obs=obs,
            observation=observation,
        )

        for step in range(self.max_steps):
            action_candidates, inventory_strings = self._action_catalog(obs)
            if not action_candidates:
                action_candidates = self._default_action_catalog()
                inventory_strings = self._inventory_strings(action_candidates)
            self.current_inventory = inventory_strings
            self._update_task_context(obs, action_candidates)

            state_vec = self._state_vector(observation)
            subgoal_vec = self.current_subgoal_vec.copy()
            mask_decision = self._resolve_masks(
                state_vec,
                len(action_candidates),
                inventory_strings,
                action_candidates,
            )
            self._last_mask_source = mask_decision.source
            self._mask_source_counts[mask_decision.source] = (
                self._mask_source_counts.get(mask_decision.source, 0) + 1
            )
            action_idx, sample = self._select_action(
                state_vec,
                subgoal_vec,
                mask_decision,
                len(action_candidates),
            )
            planner_action = action_candidates[action_idx]
            action_desc = self._describe_action(action_idx, planner_action)
            self.recent_actions.append(action_desc)
            episode_trace.append(action_desc)

            self._register_action(planner_action)
            obs, reward, terminated, truncated, info = self.env.step(planner_action)
            observation = self.env.encode_observation(obs)
            entropy = sample.entropy if sample is not None else info.get("policy_entropy", 0.0)
            self._update_entropy(entropy)
            self._update_driver_signals(
                entropy=entropy,
                mask_decision=mask_decision,
                action_candidates=action_candidates,
            )

            if self._should_request_guidance(step, info):
                self._request_guidance(
                    task_id=task_id,
                    dom_summary=observation["dom_text"],
                    inventory=inventory_strings,
                    reflections=reflections,
                    raw_obs=obs,
                    observation=observation,
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

        if self._current_macro_name and self._macro_usage_pending:
            self.macro_registry.record_success(self._current_macro_name, success)
            self._macro_usage_pending = False

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
            "targets_total": len(self._checkbox_targets),
            "targets_checked": len(self._checked_selectors),
            "targets_completed": float(self._targets_completed()),
            "submit_guardrail_steps": float(self._submit_guardrail_steps),
            "mask_iou": self._mask_iou_sum / self._mask_iou_count
            if self._mask_iou_count
            else None,
            "mask_source_last": self._last_mask_source,
            "mask_source_counts": dict(self._mask_source_counts),
            "notes_written": self._notes_episode_count,
            "plan_nodes": self._plan_node_count,
            "plan_kernel_only": self._plan_kernel_only,
            "plan_verified": self._plan_verified,
            "macro_last": self._current_macro_name,
            "macro_registry_size": len(self.macro_registry.list_for()),
            "macro_stats": self.macro_registry.stats(),
        }

    def _select_action(
        self,
        state_vec: np.ndarray,
        subgoal_vec: np.ndarray,
        mask_decision: MaskDecision,
        action_count: int,
    ) -> tuple[int, Optional[SampleOutput]]:
        """Choose an action index, respecting mask constraints."""
        final_mask = mask_decision.final if mask_decision.final.size else None
        valid_indices = self._valid_indices(final_mask, action_count)
        if not valid_indices:
            valid_indices = list(range(action_count))
        if self.learner is None:
            return random.choice(valid_indices), None
        try:
            sample = self.learner.sample_action_with_context(
                state_vec,
                subgoal_vec,
                final_mask,
                action_count,
                policy_mask=mask_decision.policy,
                coach_mask=mask_decision.coach,
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
        raw_obs: Dict[str, Any],
        observation: Dict[str, Any],
    ) -> None:
        directive = self.coach.advise(
            task_id=task_id,
            dom_summary=dom_summary,
            recent_actions=self.recent_actions,
            inventory=inventory,
            notes=reflections,
            blackboard=self.blackboard,
            target_map=self._format_target_map(),
        )
        self.current_directive = directive
        self.current_subgoal = directive.subgoal
        embedding = self.subgoal_encoder.encode(directive.subgoal)
        self.current_subgoal_vec = embedding
        self.current_goal = GoalContext(text=directive.subgoal, embedding=embedding)
        self.current_inventory = list(inventory)
        self.mask_decision = MaskDecision()
        self.blackboard.coach_to_driver.goal = directive.goal
        self.blackboard.coach_to_driver.plan_sketch = directive.plan_sketch
        self.blackboard.coach_to_driver.mask_delta = directive.mask_delta
        self.blackboard.coach_to_driver.notes_request = directive.notes_request
        self._handle_plan_sketch(directive.plan_sketch)
        self._handle_notes_request(directive, raw_obs, observation)
        self.interventions += 1

    def _mask_from_directive(self, inventory: Sequence[str]) -> Optional[np.ndarray]:
        if self.current_directive is None or not inventory:
            return None
        mask = np.ones(len(inventory), dtype=np.float32)
        allow_patterns = self.current_directive.mask_allow
        block_patterns = self.current_directive.mask_block
        if allow_patterns:
            allowed = self._match_items(allow_patterns, inventory)
            if allowed:
                mask[:] = 0.0
                mask[allowed] = 1.0
        if block_patterns:
            blocked = self._match_items(block_patterns, inventory)
            if blocked:
                mask[blocked] = 0.0
        return mask

    def _handle_plan_sketch(self, plan_text: Optional[str]) -> None:
        if not plan_text:
            self.current_plan = None
            self._plan_node_count = 0
            self._plan_kernel_only = True
            self._plan_verified = None
            return
        tokens = [token.strip() for token in plan_text.split(";") if token.strip()]
        nodes: List[DSLNode] = []
        for token in tokens:
            verb_name = token
            args: List[str] = []
            if "(" in token and token.endswith(")"):
                prefix, arg_str = token.split("(", 1)
                verb_name = prefix.strip()
                args = [arg_str[:-1]] if arg_str.endswith(")") else [arg_str]
            try:
                verb = DSLVerb(verb_name)
            except ValueError:
                verb = DSLVerb.WRITE_NOTE
                args = [token]
            nodes.append(DSLNode(verb=verb, args=args))
        if not nodes:
            plan = make_leaf(DSLVerb.WRITE_NOTE, plan_text)
        elif len(nodes) == 1:
            plan = nodes[0]
        else:
            plan = DSLNode(verb=DSLVerb.WRITE_NOTE, args=["PLAN"], children=nodes)
        self.current_plan = plan
        self._analyse_current_plan(plan)

    def _analyse_current_plan(self, plan: DSLNode) -> None:
        self._plan_node_count = count_nodes(plan)
        self._plan_kernel_only = is_kernel_only(plan)
        self._plan_verified = None
        if self.synthesiser is not None:
            sketch = Sketch(root=plan, holes=[])
            try:
                selectors = self._current_selectors()
                context = {"selectors": selectors}
                candidates = list(
                    self.synthesiser.enumerate(sketch, max_candidates=1, context=context)
                )
                if candidates:
                    self.current_plan = candidates[0].root
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Plan synthesiser failed: %s", exc)
        plan_for_verification = self.current_plan or plan
        if self.verifier is not None and plan_for_verification is not None:
            try:
                context = {"selectors": self._current_selectors()}
                result: VerificationResult = self.verifier.verify(
                    plan_for_verification,
                    trace=[],
                    context=context,
                )
                self._plan_verified = bool(result.ok)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Plan verifier failed: %s", exc)
                self._plan_verified = None
        if self.current_plan is None:
            self.current_plan = plan_for_verification
        if self.current_plan is not None and self._plan_verified:
            self._register_macro_from_plan(self.current_plan)
        else:
            self._current_macro_name = None
        self._macro_usage_pending = False

    def _handle_notes_request(
        self,
        directive: CoachDirective,
        raw_obs: Dict[str, Any],
        observation: Dict[str, Any],
    ) -> None:
        if not directive.notes_request or self.note_store is None:
            return
        url = raw_obs.get("url") or observation.get("url") or "about:blank"
        goal = raw_obs.get("goal", "")
        dom_text = observation.get("dom_text", "")
        snippet = directive.notes_request.strip()
        if dom_text:
            snippet = f"{snippet} || {dom_text[:200]}"
        title = goal[:128] if goal else directive.subgoal[:128]
        note = Note(
            id=str(uuid.uuid4()),
            url=url,
            title=title or "Planner Note",
            snippet=snippet,
            note_type="text",
            confidence=0.1,
        )
        self.note_store.append(note)
        self._notes_episode_count += 1

    def _register_macro_from_plan(self, plan: DSLNode) -> None:
        serialized = json.dumps(plan.to_dict(), sort_keys=True)
        macro_hash = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:8]
        macro_name = f"plan_{macro_hash}"
        existing_names = {macro.name for macro in self.macro_registry.list_for()}
        if macro_name not in existing_names:
            macro_body = [copy.deepcopy(plan)]
            macro = Macro(name=macro_name, body=macro_body, tier="ephemeral")
            self.macro_registry.propose(macro)
        self._current_macro_name = macro_name
        self._macro_usage_pending = True

    def _current_selectors(self) -> List[str]:
        selectors: List[str] = []
        for item in self.current_inventory:
            parts = item.split(":", 1)
            if len(parts) != 2:
                continue
            detail = parts[1].strip()
            if detail.startswith("click "):
                selectors.append(detail.split(" ", 1)[1])
            elif detail.startswith("type "):
                selectors.append(detail.split(" ", 1)[1])
        return selectors

    def _format_target_map(self) -> str:
        if not self._label_to_selector:
            if not self._checkbox_targets:
                return "(none)"
            return ", ".join(sorted(self._checkbox_targets))
        items = [f"{label}: {selector}" for label, selector in sorted(self._label_to_selector.items())]
        return "\n".join(items)

    def _resolve_masks(
        self,
        state_vec: np.ndarray,
        action_count: int,
        inventory: Sequence[str],
        action_candidates: Sequence[PlannerAction],
    ) -> MaskDecision:
        coach_mask = self._mask_from_directive(inventory)
        policy_mask = self._policy_mask(state_vec, action_count)
        if coach_mask is not None:
            self._coach_mask_steps += 1
        if policy_mask is not None:
            self._policy_mask_steps += 1
        decision = combine_masks(policy_mask, coach_mask, action_count)
        if decision.coach is not None and decision.policy is not None:
            coach_binary = decision.coach > 0.5
            policy_binary = decision.policy > 0.5
            union = np.logical_or(coach_binary, policy_binary).sum()
            if union == 0:
                iou = 1.0
            else:
                iou = np.logical_and(coach_binary, policy_binary).sum() / float(union)
            self._mask_iou_sum += float(iou)
            self._mask_iou_count += 1
        decision = self._apply_guardrails(decision, action_candidates)
        self.mask_decision = decision
        return decision

    def _policy_mask(self, state_vec: np.ndarray, action_count: int) -> Optional[np.ndarray]:
        if self.learner is None or not hasattr(self.learner, "predict_action_mask"):
            return None
        try:
            return self.learner.predict_action_mask(
                state_vec=state_vec,
                subgoal_vec=self.current_subgoal_vec,
                action_count=action_count,
            )
        except Exception as exc:  # pragma: no cover - learner optional
            logger.debug("Policy mask prediction failed: %s", exc)
            return None

    def _apply_guardrails(
        self,
        decision: MaskDecision,
        action_candidates: Sequence[PlannerAction],
    ) -> MaskDecision:
        action_count = len(action_candidates)
        if decision.final.size != action_count:
            final = np.ones(action_count, dtype=np.float32)
            limit = min(action_count, decision.final.size)
            if limit > 0:
                final[:limit] = decision.final[:limit]
            decision.final = final

        guardrail_applied = False
        submit_locked = False
        pending_targets: set[str] = set()
        if self._checkbox_targets:
            pending_targets = self._checkbox_targets - self._checked_selectors
            for idx, action in enumerate(action_candidates):
                if idx >= decision.final.size or decision.final[idx] <= 0.0:
                    continue
                if action.name != "click" or not action.selector:
                    continue
                selector = action.selector
                if selector.startswith("#ch") and selector not in self._checkbox_targets and selector not in self._submit_selectors:
                    decision.final[idx] = 0.0
                    guardrail_applied = True
            if pending_targets:
                for idx, action in enumerate(action_candidates):
                    if idx >= decision.final.size or decision.final[idx] <= 0.0:
                        continue
                    if action.name != "click" or not action.selector:
                        decision.final[idx] = 0.0
                        guardrail_applied = True
                        continue
                    if action.selector not in pending_targets:
                        decision.final[idx] = 0.0
                        guardrail_applied = True
            elif self._submit_selectors:
                for idx, action in enumerate(action_candidates):
                    if idx >= decision.final.size:
                        continue
                    if action.name == "click" and action.selector in self._submit_selectors:
                        if decision.final[idx] <= 0.0:
                            decision.final[idx] = 1.0
                            guardrail_applied = True
                        continue
                    if decision.final[idx] > 0.0:
                        decision.final[idx] = 0.0
                        guardrail_applied = True
        if self._consumed_selectors:
            for idx, action in enumerate(action_candidates):
                if action.name != "click" or not action.selector:
                    continue
                if action.selector not in self._consumed_selectors:
                    continue
                if idx < decision.final.size and decision.final[idx] > 0.0:
                    decision.final[idx] = 0.0
                    guardrail_applied = True
        if self._submit_selectors and not self._targets_completed():
            for idx, action in enumerate(action_candidates):
                if action.name == "click" and action.selector in self._submit_selectors:
                    if idx < decision.final.size:
                        decision.final[idx] = 0.0
                        guardrail_applied = True
                        submit_locked = True
        if guardrail_applied:
            existing = decision.source or "none"
            decision.source = (
                f"{existing}+guardrail" if existing not in {"none", ""} else "guardrail"
            )
            decision.guardrail_applied = True
            self._submit_guardrail_steps += 1
        self._guardrail_submit_locked = submit_locked
        return decision

    def _targets_completed(self) -> bool:
        if not self._checkbox_targets:
            return True
        return self._checkbox_targets.issubset(self._checked_selectors)

    def _update_task_context(
        self,
        raw_obs: Dict[str, Any],
        action_candidates: Sequence[PlannerAction],
    ) -> None:
        self._extract_checkbox_targets(raw_obs)
        self._identify_submit_selectors(action_candidates)

    def _extract_checkbox_targets(self, raw_obs: Dict[str, Any]) -> None:
        goal_text = raw_obs.get("goal") or ""
        if not goal_text:
            return
        if goal_text == self._last_goal_text and self._checkbox_targets:
            return
        labels = self._parse_goal_labels(goal_text)
        if not labels:
            return
        mapping = self._map_checkbox_labels(raw_obs)
        selectors = {mapping[label] for label in labels if label in mapping}
        if selectors:
            self._checkbox_targets = selectors
            self._checked_selectors = {sel for sel in self._checked_selectors if sel in selectors}
        self._label_to_selector = {label: mapping[label] for label in labels if label in mapping}
        self._last_goal_text = goal_text

    def _parse_goal_labels(self, goal_text: str) -> List[str]:
        goal_lower = goal_text.lower()
        select_idx = goal_lower.find("select ")
        if select_idx == -1:
            return []
        segment = goal_text[select_idx + len("select ") :]
        for terminator in [
            " and click submit",
            " and press submit",
            " and click the submit button",
        ]:
            term_idx = segment.lower().find(terminator)
            if term_idx != -1:
                segment = segment[:term_idx]
                break
        segment = segment.strip()
        if not segment:
            return []
        segment = segment.replace(" and ", ",")
        parts = [part.strip(" .") for part in segment.split(",")]
        return [part for part in parts if part]

    def _map_checkbox_labels(self, raw_obs: Dict[str, Any]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        dom_obj = raw_obs.get("dom_object")
        if not isinstance(dom_obj, dict):
            return mapping
        strings = dom_obj.get("strings")
        if not isinstance(strings, list):
            return mapping
        skip_tokens = {
            "br",
            "label",
            "button",
            "submit",
            "value",
            "on",
            "class",
            "checkbox",
            "input",
            "secondary-action",
            "and",
        }
        for idx, token in enumerate(strings):
            if not isinstance(token, str):
                continue
            if token == "INPUT" and idx + 2 < len(strings):
                input_type = strings[idx + 1]
                element_id = strings[idx + 2]
                if input_type == "checkbox" and element_id:
                    label = self._lookup_label(strings, idx + 3, skip_tokens)
                    if label:
                        mapping[label] = f"#{element_id}"
            elif token.startswith("ch") and token[2:].isdigit():
                selector = f"#{token}"
                label = self._lookup_label(strings, idx + 1, skip_tokens)
                if label:
                    mapping[label] = selector
        return mapping

    def _lookup_label(
        self,
        strings: Sequence[Any],
        start_index: int,
        skip_tokens: set[str],
        window: int = 6,
    ) -> Optional[str]:
        for offset in range(window):
            idx = start_index + offset
            if idx >= len(strings):
                break
            candidate = strings[idx]
            if not isinstance(candidate, str):
                continue
            candidate = candidate.strip()
            if not candidate:
                continue
            if candidate.lower() in skip_tokens:
                continue
            if candidate.isdigit():
                continue
            if any(ch.isalnum() for ch in candidate):
                return candidate
        return None

    def _identify_submit_selectors(self, action_candidates: Sequence[PlannerAction]) -> None:
        for action in action_candidates:
            if action.name != "click" or not action.selector:
                continue
            selector_lower = action.selector.lower()
            if (
                "submit" in selector_lower
                or "subbtn" in selector_lower
                or selector_lower.endswith("submit")
            ):
                self._submit_selectors.add(action.selector)

    def _register_action(self, action: PlannerAction) -> None:
        if action.name != "click" or not action.selector:
            return
        selector = action.selector
        if selector in self._checkbox_targets:
            toggled = self._detected_checkbox_toggle(selector)
            if toggled:
                self._checked_selectors.discard(selector)
                self._consumed_selectors.discard(selector)
            elif selector not in self._consumed_selectors:
                self._checked_selectors.add(selector)
                self._consumed_selectors.add(selector)
        if self._targets_completed():
            self._release_submit_block()

    def _detected_checkbox_toggle(self, selector: str) -> bool:
        if selector not in self._checkbox_targets:
            return False
        if selector not in self._consumed_selectors:
            return False
        return selector in self._checked_selectors

    def _release_submit_block(self) -> None:
        if not self._submit_selectors:
            return
        if self.current_directive is not None:
            delta = self.current_directive.mask_delta
            lower_selectors = [selector.lower() for selector in self._submit_selectors]
            if delta.block:
                delta.block = [
                    pattern
                    for pattern in delta.block
                    if not any(sel in pattern.lower() for sel in lower_selectors)
                ]
            for selector in self._submit_selectors:
                for candidate in (selector, f"click {selector}"):
                    if candidate not in delta.allow:
                        delta.allow.append(candidate)
        self._guardrail_submit_locked = False

    def answer_question(self, question: str) -> Optional[Answer]:
        if self.note_store is None or self.rag is None:
            return None
        notes = self.note_store.search(question, limit=5)
        return self.rag.answer(question, notes)

    def _update_driver_signals(
        self,
        *,
        entropy: float,
        mask_decision: MaskDecision,
        action_candidates: Sequence[PlannerAction],
    ) -> None:
        driver_msg = self.blackboard.driver_to_coach
        driver_msg.surprisal = float(entropy)
        driver_msg.suggested_query = None
        affordances: List[AffordanceHint] = []
        mask_values = mask_decision.final
        if mask_values.size and action_candidates:
            top_indices = np.argsort(mask_values)[::-1][: min(3, len(action_candidates))]
            for idx in top_indices:
                if idx >= len(action_candidates):
                    continue
                affordances.append(
                    AffordanceHint(
                        idx=int(idx),
                        text=self._describe_action(idx, action_candidates[idx]),
                        score=float(mask_values[idx]),
                    )
                )
        driver_msg.affordances = affordances
        events: List[DriverEvent] = []
        events.append(
            DriverEvent(
                label="TARGET_PROGRESS",
                payload={
                    "required": len(self._checkbox_targets),
                    "checked": len(self._checked_selectors),
                    "complete": self._targets_completed(),
                    "submit_locked": bool(self._guardrail_submit_locked),
                },
            )
        )
        if self._checkbox_targets and not self._targets_completed():
            remaining = sorted(self._checkbox_targets - self._checked_selectors)
            events.append(
                DriverEvent(
                    label="SUBMIT_LOCKED",
                    payload={"remaining": remaining},
                )
            )
        if self._notes_episode_count:
            events.append(
                DriverEvent(
                    label="NOTES_WRITTEN",
                    payload={"count": self._notes_episode_count},
                )
            )
        if self.current_plan is not None:
            events.append(
                DriverEvent(
                    label="PLAN_ANALYSIS",
                    payload={
                        "nodes": self._plan_node_count,
                        "kernel_only": self._plan_kernel_only,
                        "verified": self._plan_verified,
                    },
                )
            )
        if self._current_macro_name:
            events.append(
                DriverEvent(
                    label="MACRO_CANDIDATE",
                    payload={
                        "name": self._current_macro_name,
                        "pending": self._macro_usage_pending,
                    },
                )
            )
        driver_msg.events = events

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
            return self._join_with_ideas(task_id, [])
        entries = self.memory.load()
        top = retrieve_top_k(entries, task_id, query=task_id, k=3)
        reflections = [entry.notes for entry in top if entry.notes]
        return self._join_with_ideas(task_id, reflections)

    def _join_with_ideas(self, task_id: str, reflections: List[str]) -> str:
        snippets = list(reflections)
        ideas = self._retrieve_ddl_ideas(task_id)
        if ideas:
            snippets.extend(ideas)
        return "\n".join(snippet for snippet in snippets if snippet)

    def _retrieve_ddl_ideas(self, task_id: str) -> List[str]:
        if not self.ddl_inject or self.idea_store is None:
            return []
        ideas = self.idea_store.load()
        formatted: List[str] = []
        for idea in reversed(ideas):
            if idea.task_scope not in {task_id, None}:
                continue
            formatted.append(self._format_idea(idea))
            if len(formatted) >= self.ddl_top_k:
                break
        return formatted

    def _format_idea(self, idea: Idea) -> str:
        scores = idea.scores
        novelty = scores.get("novelty", 0.0)
        coherence = scores.get("coherence", 0.0)
        usefulness = scores.get("usefulness", 0.0)
        scope = idea.task_scope or "global"
        justification = idea.justification.strip()
        hypothesis = idea.hypothesis.strip()
        summary = f"IDEA[{scope}] {hypothesis}"
        summary += f" (novelty={novelty:.1f}, coherence={coherence:.1f}, usefulness={usefulness:.1f})"
        if justification:
            summary += f" :: {justification}"
        return summary

    def _state_vector(self, observation: Dict[str, str]) -> np.ndarray:
        return self.state_encoder.encode({"dom_text": observation.get("dom_text", "")})

    def _action_catalog(self, raw_obs: Dict[str, Any]) -> Tuple[List[PlannerAction], List[str]]:
        actions: List[PlannerAction] = []
        action_keys: set[Tuple] = set()

        def add_action(action: PlannerAction) -> None:
            key = (
                action.name,
                action.selector,
                action.text,
                action.key,
                action.direction,
                action.wait_ms,
            )
            if key in action_keys or len(actions) >= self.max_actions:
                return
            action_keys.add(key)
            actions.append(action)

        if not self._debug_obs_dumped:
            try:
                ensure_path = Path("logs")
                ensure_path.mkdir(parents=True, exist_ok=True)
                with (ensure_path / "last_raw_obs.json").open("w", encoding="utf-8") as f:
                    json.dump(raw_obs, f, default=str)
            except Exception as exc:  # pragma: no cover - debug helper
                logger.debug("Failed to dump raw observation: %s", exc)
            self._debug_obs_dumped = True

        extra = raw_obs.get("extra_element_properties")
        if isinstance(extra, dict):
            for element in extra.values():
                if not isinstance(element, dict):
                    continue
                selector = (
                    element.get("selector")
                    or element.get("css_selector")
                    or element.get("unique_selector")
                )
                if not selector:
                    continue
                tag = (element.get("tag") or element.get("nodeName") or "").lower()
                input_type = (element.get("type") or element.get("inputType") or "").lower()
                role = (element.get("role") or "").lower()
                clickable = tag in {"button", "a", "option", "label"} or input_type in {
                    "button",
                    "submit",
                    "checkbox",
                    "radio",
                } or role in {"button", "link", "checkbox", "option"}
                text_input = tag in {"textarea"} or (
                    tag == "input"
                    and input_type
                    in {"text", "email", "search", "number", "password", "url", "tel"}
                )
                if clickable:
                    add_action(make_planner_action("click", selector=selector))
                if text_input:
                    add_action(
                        make_planner_action(
                            "type",
                            selector=selector,
                            text=element.get("value", ""),
                        )
                    )
                if len(actions) >= self.max_actions:
                    break

        dom_obj = raw_obs.get("dom_object")
        if len(actions) < self.max_actions and isinstance(dom_obj, dict):
            strings = dom_obj.get("strings", [])
            for i, token in enumerate(strings):
                if len(actions) >= self.max_actions:
                    break
                if token == "INPUT" and i + 2 < len(strings):
                    input_type = strings[i + 1]
                    element_id = strings[i + 2]
                    if input_type == "checkbox" and element_id:
                        add_action(make_planner_action("click", selector=f"#{element_id}"))
                if token == "BUTTON" and i + 1 < len(strings):
                    element_id = strings[i + 1]
                    if element_id:
                        add_action(make_planner_action("click", selector=f"#{element_id}"))
                if isinstance(token, str) and token.startswith("ch") and token[2:].isdigit():
                    add_action(make_planner_action("click", selector=f"#{token}"))

        if len(actions) < self.max_actions:
            if dom_obj:
                serialized = json.dumps(dom_obj)
            else:
                dom_text = raw_obs.get("dom_text") or raw_obs.get("text") or ""
                if isinstance(dom_text, bytes):
                    try:
                        dom_text = dom_text.decode("utf-8")
                    except UnicodeDecodeError:
                        dom_text = dom_text.decode("utf-8", errors="ignore")
                elif not isinstance(dom_text, str):
                    dom_text = str(dom_text)
                serialized = dom_text
            for selector in extract_interactive_selectors(
                serialized, max_candidates=self.max_actions
            ):
                add_action(make_planner_action("click", selector=selector))
                if len(actions) >= self.max_actions:
                    break

        for nav_action in self._default_navigation_actions():
            add_action(nav_action)

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
