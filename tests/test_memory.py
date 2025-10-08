from pathlib import Path

from llm.memory import JsonlMemoryStore, MemoryEntry, retrieve_top_k


def test_memory_roundtrip(tmp_path: Path):
    store = JsonlMemoryStore(tmp_path / "memory.jsonl")
    entry = MemoryEntry(
        task_id="miniwob/click-checkboxes",
        episode_id="ep1",
        success=True,
        notes="Click the checkbox faster next time",
        selectors_used=["input[type=checkbox]"],
        failure_modes=[],
        subgoal="check all boxes",
    )
    store.append(entry)
    loaded = store.load()
    assert loaded[0].notes == entry.notes
    assert loaded[0].subgoal == "check all boxes"


def test_retrieve_top_k_returns_best_match(tmp_path: Path):
    store = JsonlMemoryStore(tmp_path / "memory.jsonl")
    entries = [
        MemoryEntry(
            task_id="task",
            episode_id=str(idx),
            success=False,
            notes=text,
            selectors_used=[],
            failure_modes=[],
        )
        for idx, text in enumerate(
            [
                "Always click the login button first",
                "Type the email before password",
                "Scroll down to find submit button",
            ]
        )
    ]
    for entry in entries:
        store.append(entry)
    retrieved = retrieve_top_k(store.load(), "task", query="login", k=1)
    assert retrieved[0].notes.startswith("Always click")
