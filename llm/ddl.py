"""Day-dreaming loop orchestration utilities."""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from llm.ideas import IDEA_SCORE_KEYS, Idea, IdeaStore
from llm.memory import JsonlMemoryStore, MemoryEntry
from llm.openai_client import OpenAIPlannerClient, LLMStats
from flappy.memory import Note, NoteStore


SYNTH_PROMPT_VERSION = "v1"
CRITIC_PROMPT_VERSION = "v1"


@dataclass
class SourceItem:
    """Lightweight wrapper over memory or note snippets used for sampling."""

    uid: str
    title: str
    snippet: str
    origin: str
    task_scope: Optional[str] = None

    def short_snippet(self, limit: int = 280) -> str:
        snippet = self.snippet.strip()
        if len(snippet) <= limit:
            return snippet
        return snippet[: limit - 3].rstrip() + "..."


@dataclass
class DaydreamingConfig:
    pairs: int = 100
    k: int = 2
    thresholds: Dict[str, float] = field(
        default_factory=lambda: {"novelty": 7.0, "coherence": 6.0, "usefulness": 6.0}
    )
    max_accept: Optional[int] = 50
    time_budget_sec: Optional[float] = None
    token_budget: Optional[int] = None
    sleep_sec: float = 0.0
    dedup: bool = True


@dataclass
class RunStats:
    pairs_evaluated: int = 0
    ideas_generated: int = 0
    ideas_accepted: int = 0
    duplicates_skipped: int = 0
    threshold_rejects: int = 0
    parse_failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    latency_sec: float = 0.0


class PairSampler:
    """Random sampler over source items supporting multi-way combinations."""

    def __init__(self, items: Sequence[SourceItem], *, rng: Optional[random.Random] = None) -> None:
        self.items = list(items)
        self.rng = rng or random.Random()

    def sample(self, *, count: int, k: int) -> Iterable[Tuple[SourceItem, ...]]:
        if k < 2:
            raise ValueError("k must be ≥2 for daydreaming combinations")
        if len(self.items) < k:
            return []
        seen: set[Tuple[str, ...]] = set()
        results: List[Tuple[SourceItem, ...]] = []
        attempts = 0
        max_attempts = count * 4
        while len(results) < count and attempts < max_attempts:
            attempts += 1
            selection = tuple(self.rng.sample(self.items, k))
            key = tuple(sorted(item.uid for item in selection))
            if key in seen:
                continue
            seen.add(key)
            results.append(selection)
        return results


class Synthesizer:
    def __init__(self, client: OpenAIPlannerClient, *, prompt_version: str = SYNTH_PROMPT_VERSION) -> None:
        self.client = client
        self.prompt_version = prompt_version

    def propose(self, items: Sequence[SourceItem]) -> str:
        concepts = []
        for idx, item in enumerate(items, start=1):
            concept = f"Concept {idx}: {item.title or '(untitled)'}\nSummary: {item.short_snippet()}"
            concepts.append(concept)
        user_prompt = "\n\n".join(concepts)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a creative synthesizer. Given concepts, propose one concise, non-obvious "
                    "hypothesis, analogy, research question, or plan. Avoid obvious statements."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Generate exactly one idea (1-3 sentences) connecting the concepts.\n" + user_prompt
                ),
            },
        ]
        response = self.client.invoke_text(messages, metadata={"prompt_version": self.prompt_version})
        return response.strip()


class Critic:
    def __init__(self, client: OpenAIPlannerClient, *, prompt_version: str = CRITIC_PROMPT_VERSION) -> None:
        self.client = client
        self.prompt_version = prompt_version

    def score(self, hypothesis: str) -> Dict[str, float]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a discerning critic. Score the hypothesis for novelty, coherence, and usefulness "
                    "on a 1-10 scale. Return JSON: {\"novelty\":int, \"coherence\":int, \"usefulness\":int, "
                    "\"justification\":string}. Keep justification ≤30 words."
                ),
            },
            {
                "role": "user",
                "content": f"Hypothesis: {hypothesis}",
            },
        ]
        text = self.client.invoke_text(messages, metadata={"prompt_version": self.prompt_version})
        payload = self._parse_json(text)
        scores: Dict[str, float] = {}
        for key in IDEA_SCORE_KEYS:
            if key in payload:
                try:
                    scores[key] = float(payload[key])
                except (TypeError, ValueError):
                    scores[key] = 0.0
            else:
                scores[key] = 0.0
        scores["justification"] = payload.get("justification", "")  # type: ignore[assignment]
        return scores

    @staticmethod
    def _parse_json(text: str) -> Dict[str, object]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"{.*}", text, re.DOTALL)
            if not match:
                raise ValueError("Critic response missing JSON payload")
            return json.loads(match.group(0))


class DaydreamingLoop:
    def __init__(
        self,
        *,
        client: OpenAIPlannerClient,
        idea_store: IdeaStore,
        memory_store: Optional[JsonlMemoryStore] = None,
        note_store: Optional[NoteStore] = None,
        config: Optional[DaydreamingConfig] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.client = client
        self.idea_store = idea_store
        self.memory_store = memory_store
        self.note_store = note_store
        self.config = config or DaydreamingConfig()
        self.rng = rng or random.Random()
        self.synthesizer = Synthesizer(client)
        self.critic = Critic(client)

    def run_once(self) -> RunStats:
        stats = RunStats()
        config = self.config
        start_time = time.perf_counter()
        initial_usage = _snapshot_stats(self.client.stats)

        items = self._collect_items()
        sampler = PairSampler(items, rng=self.rng)
        pairs = sampler.sample(count=config.pairs, k=config.k)

        existing_hashes = set(self.idea_store.hashes()) if config.dedup else set()
        accepted = 0
        token_budget = config.token_budget

        for pair in pairs:
            if config.time_budget_sec is not None:
                if time.perf_counter() - start_time >= config.time_budget_sec:
                    break
            if config.max_accept is not None and accepted >= config.max_accept:
                break
            stats.pairs_evaluated += 1

            synth_before = _snapshot_stats(self.client.stats)
            try:
                idea_text = self.synthesizer.propose(pair)
            except Exception:
                stats.parse_failures += 1
                continue
            synth_after = _snapshot_stats(self.client.stats)

            critic_before = synth_after
            try:
                score_payload = self.critic.score(idea_text)
            except Exception:
                stats.parse_failures += 1
                continue
            critic_after = _snapshot_stats(self.client.stats)

            delta_requests = critic_after["requests"] - synth_before["requests"]
            delta_input = critic_after["input_tokens"] - synth_before["input_tokens"]
            delta_output = critic_after["output_tokens"] - synth_before["output_tokens"]
            delta_latency = critic_after["latency_sec"] - synth_before["latency_sec"]

            stats.requests += int(delta_requests)
            stats.input_tokens += int(delta_input)
            stats.output_tokens += int(delta_output)
            stats.latency_sec += delta_latency

            stats.ideas_generated += 1

            scores = {k: float(score_payload.get(k, 0.0)) for k in IDEA_SCORE_KEYS}
            if not _meets_thresholds(scores, config.thresholds):
                stats.threshold_rejects += 1
                if token_budget is not None and stats.input_tokens >= token_budget:
                    break
                continue

            idea_hash = _hash_text(idea_text)
            if config.dedup and (idea_hash in existing_hashes):
                stats.duplicates_skipped += 1
                continue

            accepted += 1
            existing_hashes.add(idea_hash)

            source_ids = [item.uid for item in pair]
            task_scope = _derive_task_scope(pair)
            tags = sorted({_derive_tag(item) for item in pair})

            idea = Idea(
                id=_generate_uuid(),
                hypothesis=idea_text,
                scores=scores,
                accepted=True,
                source_ids=source_ids,
                task_scope=task_scope,
                tags=tags,
                prompt_versions={"synth": SYNTH_PROMPT_VERSION, "critic": CRITIC_PROMPT_VERSION},
                hash=idea_hash,
                cost={
                    "requests": float(delta_requests),
                    "input_tokens": float(delta_input),
                    "output_tokens": float(delta_output),
                    "latency_sec": float(delta_latency),
                },
                justification=str(score_payload.get("justification", "")),
            )
            self.idea_store.append(idea)
            stats.ideas_accepted += 1

            if token_budget is not None and stats.input_tokens >= token_budget:
                break
            if config.sleep_sec > 0:
                time.sleep(config.sleep_sec)

        final_usage = _snapshot_stats(self.client.stats)
        stats.requests = int(max(stats.requests, final_usage["requests"] - initial_usage["requests"]))
        stats.input_tokens = int(max(stats.input_tokens, final_usage["input_tokens"] - initial_usage["input_tokens"]))
        stats.output_tokens = int(max(stats.output_tokens, final_usage["output_tokens"] - initial_usage["output_tokens"]))
        stats.latency_sec = max(stats.latency_sec, final_usage["latency_sec"] - initial_usage["latency_sec"])
        return stats

    def _collect_items(self) -> List[SourceItem]:
        items: List[SourceItem] = []
        if self.memory_store is not None:
            for entry in self.memory_store.load():
                items.append(_from_memory(entry))
        if self.note_store is not None:
            for note in self.note_store.load():
                items.append(_from_note(note))
        return items


def _snapshot_stats(stats: LLMStats) -> Dict[str, float]:
    return {
        "requests": float(stats.total_requests),
        "input_tokens": float(stats.total_input_tokens),
        "output_tokens": float(stats.total_output_tokens),
        "latency_sec": float(stats.total_latency_sec),
    }


def _hash_text(text: str) -> str:
    normalized = " ".join(text.strip().split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _generate_uuid() -> str:
    import uuid

    return str(uuid.uuid4())


def _derive_task_scope(items: Sequence[SourceItem]) -> Optional[str]:
    scopes = {item.task_scope for item in items if item.task_scope}
    if len(scopes) == 1:
        return scopes.pop()
    return None


def _derive_tag(item: SourceItem) -> str:
    if item.origin == "memory" and item.task_scope:
        return f"memory:{item.task_scope}"
    return item.origin


def _meets_thresholds(scores: Dict[str, float], thresholds: Dict[str, float]) -> bool:
    for key, threshold in thresholds.items():
        if scores.get(key, 0.0) < threshold:
            return False
    return True


def _from_memory(entry: MemoryEntry) -> SourceItem:
    snippet = entry.notes or entry.subgoal or ""
    title = entry.subgoal or entry.task_id
    return SourceItem(
        uid=f"memory:{entry.episode_id}",
        title=title,
        snippet=snippet,
        origin="memory",
        task_scope=entry.task_id,
    )


def _from_note(note: Note) -> SourceItem:
    return SourceItem(
        uid=f"note:{note.id}",
        title=note.title,
        snippet=note.snippet,
        origin=note.note_type or "note",
        task_scope=None,
    )


__all__ = [
    "DaydreamingLoop",
    "DaydreamingConfig",
    "RunStats",
    "SourceItem",
    "PairSampler",
    "Synthesizer",
    "Critic",
]
