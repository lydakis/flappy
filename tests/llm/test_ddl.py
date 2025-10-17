import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.ddl import DaydreamingConfig, DaydreamingLoop
from llm.ideas import IdeaStore
from llm.memory import JsonlMemoryStore, MemoryEntry
from llm.openai_client import LLMStats


class StubClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.stats = LLMStats()

    def invoke_text(self, messages, metadata=None):  # noqa: D401
        del messages, metadata
        if not self.outputs:
            raise RuntimeError("StubClient exhausted outputs")
        text = self.outputs.pop(0)
        self.stats.total_requests += 1
        self.stats.total_input_tokens += 5
        self.stats.total_output_tokens += len(text.split())
        self.stats.total_latency_sec += 0.01
        return text


def build_memory(tmp_path: Path) -> JsonlMemoryStore:
    store = JsonlMemoryStore(tmp_path / "memory.jsonl")
    store.append(
        MemoryEntry(
            task_id="browsergym/miniwob.click-checkboxes",
            episode_id="ep1",
            success=True,
            notes="Reflection on clicking the right checkboxes",
            selectors_used=[],
            failure_modes=[],
            subgoal="Click highlighted options",
        )
    )
    store.append(
        MemoryEntry(
            task_id="browsergym/miniwob.click-checkboxes",
            episode_id="ep2",
            success=False,
            notes="Need to avoid double submits",
            selectors_used=[],
            failure_modes=["double-submit"],
            subgoal="Submit once",
        )
    )
    return store


def test_daydreaming_loop_accepts_and_dedups(tmp_path: Path) -> None:
    memory_store = build_memory(tmp_path)
    idea_store = IdeaStore(tmp_path / "ideas.jsonl")

    synth_text = "Hypothesis: reuse checkbox reflections to pre-screen actions"
    critic_json = json.dumps(
        {
            "novelty": 8,
            "coherence": 7,
            "usefulness": 7,
            "justification": "Connects prior notes to mask policy",
        }
    )
    client = StubClient([synth_text, critic_json])
    config = DaydreamingConfig(
        pairs=1,
        k=2,
        thresholds={"novelty": 7.0, "coherence": 6.0, "usefulness": 6.0},
        max_accept=1,
    )
    loop = DaydreamingLoop(
        client=client,
        idea_store=idea_store,
        memory_store=memory_store,
        config=config,
        rng=random.Random(0),
    )
    stats = loop.run_once()
    assert stats.ideas_accepted == 1
    ideas = idea_store.load()
    assert len(ideas) == 1
    assert "Hypothesis" in ideas[0].hypothesis

    # Second run should detect duplicate hash and skip
    client_again = StubClient([synth_text, critic_json])
    loop_again = DaydreamingLoop(
        client=client_again,
        idea_store=idea_store,
        memory_store=memory_store,
        config=config,
        rng=random.Random(0),
    )
    stats_again = loop_again.run_once()
    assert stats_again.ideas_accepted == 0
    assert stats_again.duplicates_skipped >= 1
