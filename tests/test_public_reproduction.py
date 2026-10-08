"""Public-checkout setup and retired API boundaries; no experiment execution."""

import json
from pathlib import Path

import pytest

from llm.budgeted_teacher import BudgetedTeacher, BudgetStop
from scripts import run_continual_tutor as continual
from scripts import run_interactive_teacher as interactive
from scripts import run_offline_tutor_repair as repair
from scripts import run_task_board_study as board
from scripts import run_teacher_comparison as teacher


def test_historical_paid_entrypoints_stop_before_client_or_input_access(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("retired entrypoint accessed data or credentials")

    monkeypatch.setattr(BudgetedTeacher, "from_key_file", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    with pytest.raises(BudgetStop, match="retired"):
        teacher.collect(Path("absent"))
    with pytest.raises(BudgetStop, match="retired"):
        interactive.run(False)
    with pytest.raises(RuntimeError, match="retired"):
        continual.run(False)


def source_fixture(tmp_path, names):
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")


def test_board_preparation_is_self_contained_and_freezes_inputs(tmp_path, monkeypatch):
    source_fixture(
        tmp_path,
        [
            "scripts/run_task_board_study.py",
            "tests/test_task_board_study.py",
            "scripts/run_offline_tutor_repair.py",
            "scripts/run_continual_tutor.py",
            "scripts/run_adaptive_curriculum.py",
            "docs/task-board-protocol.md",
            "configs/task-board-study.json",
        ],
    )
    (tmp_path / "configs/task-board-study.json").write_text('{"fixture": true}')
    output = tmp_path / "logs/task-board-study"
    monkeypatch.setattr(board, "ROOT", tmp_path)
    monkeypatch.setattr(board, "OUT", output)
    monkeypatch.setattr(board, "SEEDS", [307])
    monkeypatch.setattr(board, "make_data", lambda *args: {"phase": "fixture"})
    monkeypatch.setattr(board, "draw_case", lambda *args: {"fixture": True})
    monkeypatch.setattr(board, "audit_data", lambda: {"passed": True})
    monkeypatch.setattr(board.base, "resources", lambda: {})
    board.prepare()
    board.verify()
    assert json.loads((output / "preserved-evidence.json").read_text()) == {}
    assert json.loads((output / "plan.json").read_text())["fixture"]
    assert not (output / "runs").exists()
    with pytest.raises(FileExistsError):
        board.prepare()
    (output / "probes.json").write_text("[]")
    with pytest.raises(AssertionError):
        board.verify()


def test_replay_preparation_needs_no_account_or_conversation_files(
    tmp_path, monkeypatch
):
    source_fixture(
        tmp_path,
        [
            "scripts/run_offline_tutor_repair.py",
            "scripts/run_continual_tutor.py",
            "tests/test_offline_tutor_repair.py",
            "configs/offline-tutor-repair.json",
        ],
    )
    (tmp_path / "configs/offline-tutor-repair.json").write_text('{"fixture": true}')
    output = tmp_path / "logs/offline-tutor-repair"
    monkeypatch.setattr(repair, "ROOT", tmp_path)
    monkeypatch.setattr(repair, "OUTPUT", output)
    monkeypatch.setattr(repair, "make_stream", lambda seed: [])
    monkeypatch.setattr(repair.base, "case", lambda *args: {"fixture": True})
    monkeypatch.setattr(repair, "audit", lambda *args: {"passed": True})
    repair.prepare()
    repair.verify()
    assert json.loads((output / "integrity.json").read_text())["passed"]
    assert not (output / "runs").exists()
    with pytest.raises(FileExistsError):
        repair.prepare()
