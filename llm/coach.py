"""LLM coach interface providing subgoals and action masks."""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Iterable, List, Optional

from llm import prompts
from llm.memory import JsonlMemoryStore
from llm.openai_client import OpenAIPlannerClient

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CoachDirective:
    subgoal: str
    mask_allow: List[str] = dataclasses.field(default_factory=list)
    mask_block: List[str] = dataclasses.field(default_factory=list)


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
    ) -> CoachDirective:
        """Request a new subgoal and optional mask constraints."""
        developer_prompt = prompts.COACH_DEVELOPER_PROMPT.format(
            task_id=task_id,
            dom=dom_summary,
            recent_actions="\n".join(recent_actions),
            inventory="\n".join(inventory),
            notes=notes,
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
        if not subgoal:
            raise ValueError("Coach response missing SUBGOAL line.")
        return CoachDirective(subgoal=subgoal, mask_allow=mask_allow, mask_block=mask_block)


def _split_items(payload: str) -> List[str]:
    items = [item.strip() for item in re.split(r"[;,]", payload) if item.strip()]
    return items
