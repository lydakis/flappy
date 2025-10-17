"""LLM coach interface providing goals, plans, and mask deltas."""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from typing import Iterable, List, Optional

from llm import prompts
from llm.memory import JsonlMemoryStore
from llm.openai_client import OpenAIPlannerClient
from flappy.blackboard import BlackboardState
from flappy.interfaces import MaskDelta

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CoachDirective:
    subgoal: str
    mask_delta: MaskDelta = dataclasses.field(default_factory=MaskDelta)
    plan_sketch: Optional[str] = None
    notes_request: Optional[str] = None

    @property
    def goal(self) -> str:
        return self.subgoal

    @property
    def mask_allow(self) -> List[str]:
        return self.mask_delta.allow

    @property
    def mask_block(self) -> List[str]:
        return self.mask_delta.block


class Coach:
    """High-level coordinator that queries the LLM for guidance."""

    def __init__(
        self,
        client: OpenAIPlannerClient,
        *,
        memory: Optional[JsonlMemoryStore] = None,
    ) -> None:
        self.client = client
        self.memory = memory

    def advise(
        self,
        *,
        task_id: str,
        dom_summary: str,
        recent_actions: Iterable[str],
        inventory: Iterable[str],
        notes: str,
        blackboard: Optional[BlackboardState] = None,
        target_map: str = "",
    ) -> CoachDirective:
        """Request a new subgoal and optional mask constraints."""
        blackboard_json = ""
        if blackboard is not None:
            try:
                blackboard_json = blackboard.to_json(indent=2)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to serialise blackboard: %s", exc)
                blackboard_json = json.dumps({}, indent=2)
        developer_prompt = prompts.COACH_DEVELOPER_PROMPT.format(
            task_id=task_id,
            dom=dom_summary,
            recent_actions="\n".join(recent_actions),
            inventory="\n".join(inventory),
            notes=notes,
            blackboard=blackboard_json,
            target_map=target_map or "(none)",
        )
        messages = [
            {"role": "system", "content": prompts.COACH_SYSTEM_PROMPT},
            {"role": "developer", "content": developer_prompt},
        ]
        response_text = self.client.invoke_text(messages)
        directive = self._parse_response(response_text)
        logger.debug("Coach directive: %s", directive)
        return directive

    def reflect(self, task_id: str, episode_trace: Iterable[str]) -> str:
        """Generate a reflection string for episodic memory."""
        prompt = prompts.REFLECTION_PROMPT.format(task_id=task_id)
        messages = [
            {"role": "system", "content": prompts.COACH_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{prompt}\nTrace:\n" + "\n".join(episode_trace),
            },
        ]
        try:
            return self.client.invoke_text(messages).strip()
        except Exception as exc:  # pragma: no cover - external failures
            logger.warning("Coach reflection failed: %s", exc)
            return ""

    @staticmethod
    def _parse_response(text: str) -> CoachDirective:
        subgoal = ""
        mask_allow: List[str] = []
        mask_block: List[str] = []
        plan: Optional[str] = None
        notes_req: Optional[str] = None
        for line in text.splitlines():
            clean = line.strip()
            if not clean:
                continue
            if clean.upper().startswith("SUBGOAL:"):
                subgoal = clean.split(":", 1)[1].strip()
            elif clean.upper().startswith("MASK_ALLOW:"):
                mask_allow = _split_items(clean.split(":", 1)[1])
            elif clean.upper().startswith("MASK_BLOCK:"):
                mask_block = _split_items(clean.split(":", 1)[1])
            elif clean.upper().startswith("PLAN:"):
                plan = clean.split(":", 1)[1].strip()
            elif clean.upper().startswith("PLAN_SKETCH:"):
                plan = clean.split(":", 1)[1].strip()
            elif clean.upper().startswith("NOTES_REQUEST:"):
                notes_req = clean.split(":", 1)[1].strip()
        if not subgoal:
            raise ValueError("Coach response missing SUBGOAL line.")
        return CoachDirective(
            subgoal=subgoal,
            mask_delta=MaskDelta(allow=mask_allow, block=mask_block),
            plan_sketch=plan,
            notes_request=notes_req,
        )


def _split_items(payload: str) -> List[str]:
    items = [item.strip() for item in re.split(r"[;,]", payload) if item.strip()]
    return items
