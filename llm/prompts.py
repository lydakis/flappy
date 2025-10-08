"""Prompt templates for the FLAPPY coach architecture."""

from __future__ import annotations

COACH_SYSTEM_PROMPT = """You are the FLAPPY Coach for a browser-navigation RL agent.
You never control the browser. You output subgoals and action constraints the agent consumes.
Valid directives:
  1. SUBGOAL: <short phrase>
  2. Optional MASK_ALLOW: <semicolon-separated patterns or indices>
  3. Optional MASK_BLOCK: <semicolon-separated patterns or indices>
Keep responses to 40 words or fewer. No explanations."""

COACH_DEVELOPER_PROMPT = """Task: {task_id}
DOM summary:
{dom}

Recent attempts:
{recent_actions}

Known elements:
{inventory}

Prior notes:
{notes}

Emit exactly one SUBGOAL line. Optionally emit MASK_ALLOW or MASK_BLOCK referencing indices or patterns from the inventory."""

REFLECTION_PROMPT = """You finished task {task_id}.
Produce three bullets covering:
- What worked
- What failed
- One concrete change next time (start with a verb)
Limit to ≤40 words total."""
