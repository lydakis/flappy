# Day‑Dreaming Loop (DDL) for FLAPPY

## Overview

- Purpose: add an offline/idle background process that recombines stored experiences to generate, score, and persist “ideas” (hypotheses, plans, analogies). Accepted ideas feed back into the system’s memory and guidance to improve exploration over time.
- Fit: leverages existing modules (LLM coach, episodic memory, RAG, verifier, macro registry) and adds a small set of focused components to operationalize a generator → critic → write‑back loop without changing core agent behavior by default.

## Goals

- Continuous background search for non‑obvious connections across reflections and notes.
- Lightweight generator → critic pipeline with measurable “daydreaming tax” (cost/latency) and ROI.
- Persist accepted ideas separately from episodic reflections; surface them when relevant.
- Minimal intrusion: run as a standalone script; agent injection is opt‑in and configurable.

## Non‑Goals

- Training new LLMs or full world‑model learning (Dreamer) in this phase.
- Replacing PPO+RND; DDL complements exploration rather than alters the control loop.
- Building a production vector database; start with extensible stubs.

## Architecture

New components (proposed locations):

- `llm/ideas.py` — Idea dataclass + `IdeaStore` (JSONL).
- `llm/ddl.py` — Orchestrator and utilities:
  - `DaydreamingLoop`: scheduler + budgets.
  - `PairSampler`: draws pairs (or k‑tuples) from `memory.jsonl` and `notes.jsonl`.
  - `Synthesizer`: calls LLM to propose a hypothesis from a pair.
  - `Critic`: calls LLM to score novelty/coherence/usefulness and decide accept/reject.
- `scripts/run_daydream.py` — CLI entrypoint for background runs.

Reused components:

- LLM client: `llm/openai_client.py`
- Episodic memory: `llm/memory.py`
- Typed notes: `flappy/memory.py`
- Optional checks: `flappy/verify.py` (extend later for heuristic sanity checks)
- Optional surfacing during episodes: `agents/hybrid.py` retrieval path

## Data Model

Idea (JSONL; stored in `ideas.jsonl` by default):

```json
{
  "id": "uuid",
  "source_ids": ["memory:episode_id", "note:id"],
  "task_scope": "browsergym/miniwob.click-checkboxes" | null,
  "hypothesis": "concise idea text",
  "scores": {"novelty": 0-10, "coherence": 0-10, "usefulness": 0-10},
  "accepted": true,
  "tags": ["analogy", "plan", "question"],
  "prompt_versions": {"synth": "v1", "critic": "v1"},
  "timestamp": "ISO8601",
  "hash": "sha1(normalized hypothesis)",
  "cost": {"input_tokens": n, "output_tokens": n, "latency_sec": f}
}
```

## Prompts

- Synthesizer (system):
  - “You are a creative synthesizer. Given two concepts/snippets, propose one concise, non‑obvious hypothesis/analogy/plan. Avoid obvious statements. 1–3 sentences.”
- Critic (system; JSON output):
  - “You are a discerning critic. Score the hypothesis for novelty, coherence, usefulness on 1–10. Return JSON: {"novelty":int, "coherence":int, "usefulness":int, "justification":string}. Keep justification ≤30 words.”

## Algorithm

1. Sampling
   - Draw uniformly or with reservoir/anti‑spaced repetition across:
     - Reflections from `llm/memory.JsonlMemoryStore` (task‑scoped and global).
     - Notes from `flappy.memory.NoteStore` (note_type in {text, idea}).
   - Start with pairs; allow k>2 later.
2. Generate
   - Build a compact pair payload (titles/snippets only). Call Synthesizer.
3. Critique
   - Call Critic and parse scores; optionally call `PlanVerifier` for selector sanity if the output looks like a plan.
4. Accept / Write‑Back
   - Accept if scores exceed thresholds (default: novelty≥7, coherence≥6, usefulness≥6).
   - Append `Idea` to `ideas.jsonl` (and optionally mirror into `NoteStore` with `note_type="idea"`).
5. Dedup
   - Compute `hash = sha1(normalize(hypothesis))`; skip if seen.
6. Budgeting
   - Stop when hitting token/time budget or max accepted ideas per run.

## Scheduling & Budgets

CLI example:

```bash
python scripts/run_daydream.py \
  --memory memory.jsonl \
  --notes notes.jsonl \
  --ideas ideas.jsonl \
  --pairs 500 \
  --accept-thresholds 7,6,6 \
  --time-budget-sec 900 \
  --token-budget 200000 \
  --sleep-sec 0
```

Run ad‑hoc, via cron, or CI nightly. Default to no mutation of agent behavior.

## Surfacing Ideas (Opt‑In)

- Retrieval injection (no invasive changes required):
  - Extend `agents/hybrid.py:_retrieve_reflections` to concatenate top‑K accepted ideas matching `task_id` (or global ideas) via a simple search in `IdeaStore`.
  - Gate by config: `--ddl-inject` or `configs/default.yaml: ddl.inject: true`.
- Alternative surface points:
  - As coach “Prior notes” (existing `notes` field passed to `llm/coach.Coach.advise`).
  - As `NOTES_REQUEST` materialization to `flappy.memory.NoteStore` with `note_type="idea"`.

## Metrics & Telemetry

- Capture per‑run:
  - LLM usage: `OpenAIPlannerClient.asdict()` (requests, tokens, latency, estimated cost).
  - Sampling stats: pairs tried, acceptance rate, dedup rate.
  - Idea quality: score distributions; justification snippets.
- File sinks:
  - CSV: `logs/ddl/metrics.csv`
  - JSONL: `ideas.jsonl`
  - TensorBoard (optional): `logs/tb/ddl/`
- ROI experiments:
  - Offline A/B eval with `scripts/run_eval.py` comparing Baseline vs. DDL‑injected reflections (same seeds, N tasks). Report deltas on `success_rate`, `steps_to_success`, `avg_coach_interventions`.

## Configuration

`configs/default.yaml` (new section):

```yaml
ddl:
  enable: false
  inject: false
  pairs: 500
  k: 2
  thresholds: { novelty: 7, coherence: 6, usefulness: 6 }
  time_budget_sec: 900
  token_budget: 200000
  max_accept: 50
```

CLI flags override config.

## Failure Modes & Guardrails

- Low hit‑rate / high cost: enforce budgets; backoff on low acceptance; log cost/idea.
- Trivial/obvious outputs: stricter critic threshold; simple keyword filters.
- Memory spam: separate `ideas.jsonl`; only surface top‑scored, deduped ideas.
- Hallucinated selectors/plans: optionally pass through `PlanVerifier` and discard failures.
- Topic drift: when task‑conditioned, bias sampling to same `task_id`; keep a small percentage for cross‑task mixing.

## APIs & Interfaces

- `llm/ideas.py`
  - `@dataclass Idea`
  - `class IdeaStore(path: Path)` → `append(idea: Idea)`, `load() -> List[Idea]`, `search(query: str, limit: int) -> List[Idea]`
- `llm/ddl.py`
  - `class DaydreamingLoop(...).run_once() -> RunStats`
  - `class PairSampler(...).sample(n:int) -> Iterable[Tuple[SourceItem, SourceItem]]`
  - `class Synthesizer(client).propose(a: SourceItem, b: SourceItem) -> str`
  - `class Critic(client).score(text:str) -> Scores`
- `scripts/run_daydream.py`
  - Parses CLI, enforces budgets, writes metrics/ideas.

## Integration Plan

- Phase 0 (Offline Only)
  - Implement DDL components + CLI. No injection into agent execution. Validate acceptance rate, cost/idea on MiniWoB reflections/notes.
- Phase 1 (Read‑Only Injection)
  - Enable `ddl.inject` to append top‑K ideas to coach “Prior notes” in `agents/hybrid.py:_retrieve_reflections`. Gate via flag; default off.
- Phase 2 (Tight Looping)
  - Explore writing accepted, verified plans to `MacroRegistry` (`flappy/macro_registry.py`) as tier “experimental” and track success.

## Testing

- Unit tests (stubs for LLM):
  - `tests/llm/test_ddl_sampler.py` — sampling coverage, dedup.
  - `tests/llm/test_ddl_critic_parse.py` — strict JSON parsing, thresholds.
  - `tests/llm/test_ideas_store.py` — append/load/search, hash dedup.
  - `tests/agents/test_hybrid_inject_ideas.py` — verify injection path is optional and append‑only.
- Smoke run: `python scripts/run_daydream.py --pairs 20 --time-budget-sec 60` should produce `ideas.jsonl` and `logs/ddl/metrics.csv`.

## Security & Privacy

- Never send URLs, selectors, or long DOM dumps to the LLM; summarize locally and redact PII.
- Cap prompt size; include only short snippets/titles from sources.
- Respect `.env` for keys; no credentials in artifacts.

## Risks & Mitigations

- Cost risk: start with tight budgets; export per‑run cost metrics.
- Quality risk: conservative thresholds; keep “inject” off by default; evaluate A/B before rollout.
- Data drift: tag prompt versions in ideas; re‑score periodically if prompts change.

## Open Questions

- How aggressive should cross‑task mixing be for MiniWoB vs. real web tasks?
- Should accepted ideas always be mirrored into `NoteStore` for unified RAG?
- When to promote high‑scoring plans to `MacroRegistry` automatically vs. human‑in‑the‑loop?

## Example CLI

- Background run (offline):

```bash
python scripts/run_daydream.py \
  --memory memory.jsonl \
  --notes notes.jsonl \
  --ideas ideas.jsonl \
  --pairs 500 \
  --time-budget-sec 900 \
  --token-budget 150000
```

- Eval with injection (read‑only):

```bash
python scripts/run_eval.py --agent hybrid --tasks eval/tasks.yaml --frozen --no-headless
```

with `ddl.inject: true` set in `configs/default.yaml`.

