import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.ideas import Idea, IdeaStore


def test_idea_store_append_load_search(tmp_path: Path) -> None:
    store = IdeaStore(tmp_path / "ideas.jsonl")
    idea = Idea(
        id="idea-1",
        hypothesis="Connect checkbox reflections with mask planning",
        scores={"novelty": 8.0, "coherence": 7.5, "usefulness": 7.0},
        accepted=True,
        source_ids=["memory:ep1"],
        task_scope="browsergym/miniwob.click-checkboxes",
        tags=["memory:browsergym/miniwob.click-checkboxes"],
        hash="hash-1",
    )
    store.append(idea)

    ideas = store.load()
    assert len(ideas) == 1
    loaded = ideas[0]
    assert loaded.hypothesis == idea.hypothesis
    assert store.contains_hash("hash-1") is True

    matches = store.search("checkbox", limit=5, task_scope="browsergym/miniwob.click-checkboxes")
    assert matches
    assert matches[0].hypothesis == idea.hypothesis

    non_matches = store.search("forms", limit=5, task_scope="browsergym/miniwob.enter-text")
    assert non_matches == []
