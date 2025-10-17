from pathlib import Path

from agents.hybrid import HybridAgent
from llm.ideas import Idea, IdeaStore
from llm.memory import JsonlMemoryStore, MemoryEntry


class DummyEnv:
    pass


class DummyCoach:
    pass


def build_memory(path: Path) -> JsonlMemoryStore:
    store = JsonlMemoryStore(path)
    store.append(
        MemoryEntry(
            task_id="browsergym/miniwob.click-checkboxes",
            episode_id="ep0",
            success=True,
            notes="Checkbox reflections helped",
            selectors_used=[],
            failure_modes=[],
            subgoal="Select highlighted options",
        )
    )
    return store


def build_idea_store(path: Path) -> IdeaStore:
    store = IdeaStore(path)
    store.append(
        Idea(
            id="idea-1",
            hypothesis="Use mask decay when novelty drops",
            scores={"novelty": 7.5, "coherence": 6.5, "usefulness": 6.2},
            accepted=True,
            source_ids=["memory:ep0"],
            task_scope="browsergym/miniwob.click-checkboxes",
            tags=["memory:browsergym/miniwob.click-checkboxes"],
            hash="hash-idea-1",
            justification="Encourages exploration",
        )
    )
    return store


def test_reflections_include_ddl_when_enabled(tmp_path: Path) -> None:
    memory_store = build_memory(tmp_path / "memory.jsonl")
    idea_store = build_idea_store(tmp_path / "ideas.jsonl")
    agent = HybridAgent(
        env=DummyEnv(),
        coach=DummyCoach(),
        learner=None,
        memory=memory_store,
        idea_store=idea_store,
        ddl_inject=True,
        ddl_top_k=2,
    )
    reflections = agent._retrieve_reflections("browsergym/miniwob.click-checkboxes")
    assert "Checkbox reflections helped" in reflections
    assert "IDEA[" in reflections


def test_reflections_without_ddl(tmp_path: Path) -> None:
    memory_store = build_memory(tmp_path / "memory.jsonl")
    idea_store = build_idea_store(tmp_path / "ideas.jsonl")
    agent = HybridAgent(
        env=DummyEnv(),
        coach=DummyCoach(),
        learner=None,
        memory=memory_store,
        idea_store=idea_store,
        ddl_inject=False,
    )
    reflections = agent._retrieve_reflections("browsergym/miniwob.click-checkboxes")
    assert "Checkbox reflections helped" in reflections
    assert "IDEA[" not in reflections
