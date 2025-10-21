from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest


class DummyRLAgent:
    """Minimal PureRLAgent replacement for testing."""

    def __init__(self, env_fn):  # noqa: D401 - match real signature
        self.env_fn = env_fn
        self.model = SimpleNamespace(load=lambda path: None)


@pytest.mark.usefixtures("monkeypatch")
def test_baseline_rl_avoids_llm(monkeypatch):
    import scripts.run_eval as run_eval

    monkeypatch.setattr(
        run_eval, "parse_args", lambda: argparse.Namespace(
            agent="baseline_rl",
            tasks="ignored.yaml",
            episodes=1,
            frozen=False,
            checkpoint=None,
            headless=True,
            memory="memory.jsonl",
            env=None,
            note_store="notes.jsonl",
            idea_store="ideas.jsonl",
            ddl_inject=False,
            ddl_top_k=3,
            qa_question=None,
        )
    )

    monkeypatch.setattr(run_eval, "load_task_list", lambda path: {"env/test": {}})
    monkeypatch.setattr(run_eval, "evaluate_agent", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_eval, "PureRLAgent", DummyRLAgent)
    monkeypatch.setattr(run_eval, "load_memory", lambda path: None)
    monkeypatch.setattr(run_eval, "NoteStore", lambda path: None)
    monkeypatch.setattr(run_eval, "SimpleRAG", lambda: None)
    monkeypatch.setattr(run_eval, "DocumentExtractor", lambda: None)
    monkeypatch.setattr(run_eval, "IdeaStore", lambda path: None)

    def _boom() -> None:
        raise AssertionError("OpenAIPlannerClient should not be instantiated")

    monkeypatch.setattr(run_eval, "OpenAIPlannerClient", _boom)

    # Should not raise despite OpenAIPlannerClient blowing up when touched.
    run_eval.main()
