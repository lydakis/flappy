#!/usr/bin/env python3
"""Approved, bounded offline board study. No credential or API client path."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import socket
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn

from scripts import run_continual_tutor as base
from scripts.run_offline_tutor_repair import (
    FeedbackMemory,
    no_network,
    public_rows,
    update_with_feedback,
)

OUT = ROOT / "logs/task-board-study"
CHECKPOINTS = ROOT / "checkpoints/task-board-study"
SEEDS = [307, 401, 503]
ARMS = ["current_no_tutor", "board_no_tutor", "current_tutor", "board_tutor"]


def save_compact(path: Path, value) -> None:
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n")


def valid_schedules() -> list[list[int]]:
    result = []
    for indices in itertools.combinations(range(8), 4):
        schedule = [1 if i in indices else 2 for i in range(8)]
        starts = Counter(
            h for i, h in enumerate(schedule) if i == 0 or h != schedule[i - 1]
        )
        if min(starts[1], starts[2]) >= 2:
            result.append(schedule)
    return result


def pay(skill: int) -> float:
    return 0.25 if skill == 1 else 1.0


def public_view(schedule: list[int], cycle: int, board: bool) -> dict:
    contract = cycle // 32
    future = []
    if board:
        for offset in (1, 2):
            index = contract + offset
            if index < 8:
                h = schedule[index]
                future.append(
                    {"skill": h, "pay": pay(h), "available_in": index * 32 - cycle}
                )
    skill = schedule[contract]
    return {
        "current": {"skill": skill, "pay": pay(skill), "remaining": 32 - cycle % 32},
        "past_contracts": schedule[:contract],
        "future": future,
    }


class History:
    def __init__(self):
        self.count = [0, 0]
        self.accuracy = [0.25, 0.25]
        self.last_seen = [-256, -256]

    def observe(self, skill: int, correct: list[bool], cycle: int) -> None:
        index = skill - 1
        self.count[index] += len(correct)
        self.accuracy[index] = 0.9 * self.accuracy[index] + 0.1 * sum(correct) / len(
            correct
        )
        self.last_seen[index] = cycle


def observation(
    view: dict, cycle: int, credits: float, calls: int, history: History
) -> torch.Tensor:
    current = view["current"]
    values = [
        float(current["skill"] == 1),
        float(current["skill"] == 2),
        current["pay"],
        current["remaining"] / 32,
        cycle / 256,
        min(credits, 64) / 64,
        (12 - calls) / 12,
    ]
    for i in range(8):
        h = view["past_contracts"][i] if i < len(view["past_contracts"]) else 0
        values.extend([float(h == 1), float(h == 2), float(h != 0)])
    for i in range(2):
        values.extend(
            [
                history.count[i] / 4096,
                history.accuracy[i],
                min(cycle - history.last_seen[i], 256) / 256,
            ]
        )
    for i in range(2):
        if i < len(view["future"]):
            card = view["future"][i]
            values.extend(
                [
                    float(card["skill"] == 1),
                    float(card["skill"] == 2),
                    card["pay"],
                    card["available_in"] / 64,
                    1.0,
                ]
            )
        else:
            values.extend([0.0] * 5)
    assert len(values) == 47
    return torch.tensor(values, dtype=torch.float32)


class Allocator(nn.Module):
    def __init__(self, tutor_enabled: bool):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(47, 32), nn.Tanh())
        self.policy = nn.Linear(32, 4)
        self.value = nn.Linear(32, 1)
        self.tutor_enabled = tutor_enabled
        nn.init.zeros_(self.policy.weight)
        with torch.no_grad():
            self.policy.bias.copy_(torch.tensor([0.475, 0.475, 0.025, 0.025]).log())

    def forward(self, x):
        hidden = self.trunk(x)
        logits = self.policy(hidden)
        if not self.tutor_enabled:
            logits = logits + logits.new_tensor([0.0, 0.0, -1e9, -1e9])
        return logits, self.value(hidden).squeeze(-1)


class Student:
    def __init__(self, seed: int):
        torch.manual_seed(seed)
        self.model = base.AnswerModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003)
        self.memory = FeedbackMemory("balanced_feedback", seed)
        self.rng = torch.Generator().manual_seed(seed + 40000)
        self.updates = 0


def learn_packet(
    student: Student, rows: list[dict], *, buy_labels: bool, paid: bool, on_commit=None
) -> dict:
    x = base.features(public_rows(rows))
    with torch.no_grad():
        logits, values = student.model(x)
        probabilities = logits.softmax(-1)
        actions = torch.multinomial(probabilities, 1, generator=student.rng).squeeze(-1)
        decision = {
            "x": x,
            "actions": actions,
            "old_log": logits.log_softmax(-1)[torch.arange(8), actions],
            "old_values": values,
        }
    if on_commit is not None:
        on_commit(actions.tolist())
    # All outputs are fixed before either ordinary grading or purchased labels.
    targets = torch.tensor([r["label"] for r in rows])
    correct = actions == targets
    labels = (
        [base.answer(r["p"], r["v"], r["horizon"]) for r in public_rows(rows)]
        if buy_labels
        else None
    )
    if labels is not None:
        assert labels == targets.tolist()
    replay = student.memory.sample()
    before = dict(student.memory.seen)
    learning_signal = correct.float() * pay(rows[0]["horizon"])
    update = update_with_feedback(
        student.model, student.optimizer, decision, learning_signal, labels, replay
    )
    student.memory.add(x, actions, correct)
    student.updates += 4
    replay_counts = (
        []
        if replay is None
        else [int((replay[0][:, 2] < 0).sum()), int((replay[0][:, 2] > 0).sum())]
    )
    return {
        "examples": rows,
        "committed_answers": actions.tolist(),
        "pre_feedback_greedy": probabilities.argmax(-1).tolist(),
        "correct": correct.tolist(),
        "paid_income": float(correct.sum()) * pay(rows[0]["horizon"]) if paid else 0.0,
        "teacher_labels": labels,
        "teacher": "offline_perfect_oracle" if buy_labels else None,
        "memory_seen_before": before,
        "replay_skill_counts": replay_counts,
        **update,
    }


def latent_key(row: dict) -> tuple[int, int, int]:
    return (round(row["p"] * 1000), round(row["v"] * 1000), row["horizon"])


def partition(key: tuple[int, int, int]) -> int:
    return hashlib.sha256(":".join(map(str, key)).encode()).digest()[0] % 5


def draw_case(
    numeric_rng, target: int, horizon: int, region: int, used: set, identity: str
) -> dict:
    for _ in range(20000):
        p, v = map(int, numeric_rng.integers(-2000, 2001, size=2))
        final = (p + v * horizon) / 1000
        label = int(final >= -1) + int(final >= 0) + int(final >= 1)
        if (
            label != target
            or min(abs(final - boundary) for boundary in (-1, 0, 1)) < 0.15
        ):
            continue
        key = (p, v, horizon)
        if key in used or partition(key) != region:
            continue
        used.add(key)
        return {
            "id": identity,
            "p": p / 1000,
            "v": v / 1000,
            "horizon": horizon,
            "label": label,
        }
    raise RuntimeError("Case generation bound")


def make_data(seed: int, episode: int, schedule: list[int], used: set) -> dict:
    phase = "train" if episode < 4 else "assessment"
    numeric = {
        role: np.random.default_rng(
            np.random.SeedSequence([913457, seed, episode, role])
        )
        for role in range(3)
    }
    targets = {
        role: np.random.default_rng(
            np.random.SeedSequence([512093, seed, episode, role])
        )
        for role in range(3)
    }
    result = {
        "seed": seed,
        "episode": episode,
        "phase": phase,
        "schedule": schedule,
        "paid": [],
        "practice": [],
    }
    for cycle in range(256):
        packets = []
        for role in range(3):
            h = schedule[cycle // 32] if role == 0 else role
            region = (0 if role == 0 else 1) + (0 if phase == "train" else 2)
            rows = [
                draw_case(
                    numeric[role],
                    int(targets[role].integers(4)),
                    h,
                    region,
                    used,
                    f"{seed}:{episode}:{cycle}:{role}:{i}",
                )
                for i in range(8)
            ]
            packets.append(rows)
        result["paid"].append(packets[0])
        result["practice"].append(packets[1:])
    return result


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    plan = json.loads((ROOT / "configs/task-board-study.json").read_text())
    protocol = ROOT / "docs/task-board-protocol.md"
    plan["proposal_sha256"] = hashlib.sha256(protocol.read_bytes()).hexdigest()
    base.save(OUT / "plan.json", plan)
    base.save(OUT / "preserved-evidence.json", {})
    (OUT / "approved-protocol.md").write_bytes(protocol.read_bytes())
    if (OUT / "source-data-hashes.json").exists():
        raise RuntimeError("Board study already prepared")
    started = time.monotonic()
    used = set()
    schedules = valid_schedules()
    schedule_records = []
    for seed in SEEDS:
        indices = np.random.default_rng(seed + 490000).choice(
            len(schedules), size=6, replace=False
        )
        for episode, index in enumerate(indices):
            schedule = schedules[int(index)]
            data = make_data(seed, episode, schedule, used)
            save_compact(OUT / f"data-{seed}-{episode}.json", data)
            schedule_records.append(
                {
                    "seed": seed,
                    "episode": episode,
                    "phase": data["phase"],
                    "schedule": schedule,
                }
            )
            if time.monotonic() - started > 300:
                raise RuntimeError("Data preparation time limit")
    rng = np.random.default_rng(202610101)
    probes = [
        draw_case(rng, i % 4, h, 4, used, f"probe:{h}:{i}")
        for h in (1, 2)
        for i in range(256)
    ]
    save_compact(OUT / "probes.json", probes)
    base.save(OUT / "schedules.json", schedule_records)
    audit = audit_data()
    audit.update(
        {
            "preparation_seconds": time.monotonic() - started,
            "resources": base.resources(),
        }
    )
    base.save(OUT / "integrity.json", audit)
    paths = [
        "scripts/run_task_board_study.py",
        "tests/test_task_board_study.py",
        "scripts/run_offline_tutor_repair.py",
        "scripts/run_continual_tutor.py",
        "scripts/run_adaptive_curriculum.py",
        "docs/task-board-protocol.md",
        *[str(p.relative_to(ROOT)) for p in OUT.glob("*.json")],
        "logs/task-board-study/approved-protocol.md",
    ]
    base.save(
        OUT / "source-data-hashes.json",
        {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in paths
        },
    )
    print(json.dumps(audit, indent=2), flush=True)


def audit_data() -> dict:
    used, slot_rows, fitted, heldout = set(), [], defaultdict(Counter), []
    for seed in SEEDS:
        schedules = []
        for episode in range(6):
            data = json.loads((OUT / f"data-{seed}-{episode}.json").read_text())
            schedules.append(tuple(data["schedule"]))
            assert data["schedule"] in valid_schedules()
            counts = np.zeros((8, 4), dtype=int)
            for cycle in range(256):
                for role, rows in enumerate(
                    [data["paid"][cycle], *data["practice"][cycle]]
                ):
                    for slot, row in enumerate(rows):
                        expected_skill = (
                            data["schedule"][cycle // 32] if role == 0 else role
                        )
                        assert row["horizon"] == expected_skill
                        key = latent_key(row)
                        region = (0 if role == 0 else 1) + (0 if episode < 4 else 2)
                        assert key not in used and partition(key) == region
                        assert (
                            base.answer(row["p"], row["v"], row["horizon"])
                            == row["label"]
                        )
                        used.add(key)
                        if role == 0:
                            counts[slot, row["label"]] += 1
                            contract = cycle // 32
                            feature = (
                                row["horizon"],
                                contract,
                                slot,
                                tuple(data["schedule"][contract + 1 : contract + 3]),
                            )
                            if episode < 4:
                                fitted[feature][row["label"]] += 1
                            else:
                                heldout.append((feature, row["label"]))
            score = float(counts.max(-1).sum() / 2048)
            assert score <= 0.35
            slot_rows.append(
                {"seed": seed, "episode": episode, "slot_majority_accuracy": score}
            )
        assert len(set(schedules)) == 6
    for row in json.loads((OUT / "probes.json").read_text()):
        key = latent_key(row)
        assert key not in used and partition(key) == 4
        assert base.answer(row["p"], row["v"], row["horizon"]) == row["label"]
        used.add(key)
    score = sum(
        (fitted[key].most_common(1)[0][0] if key in fitted else 0) == label
        for key, label in heldout
    ) / len(heldout)
    assert score <= 0.35
    return {
        "passed": True,
        "globally_unique_cases_including_unselected_candidates_and_probes": len(used),
        "all_five_partitions_disjoint": True,
        "slot_checks": slot_rows,
        "held_out_board_metadata_only_label_accuracy": score,
        "held_out_paid_cases": len(heldout),
        "real_calls": 0,
    }


def verify() -> None:
    for name, expected in json.loads(
        (OUT / "source-data-hashes.json").read_text()
    ).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    for name, expected in json.loads(
        (OUT / "preserved-evidence.json").read_text()
    ).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name


def output_bytes() -> int:
    return sum(
        p.stat().st_size
        for directory in (OUT, CHECKPOINTS)
        for p in directory.rglob("*")
        if p.is_file()
    )


def check_resources(started: float, preparation_seconds: float) -> dict:
    state = base.resources()
    size = output_bytes()
    if (
        time.monotonic() - started + preparation_seconds > 600
        or state["peak_rss_mib"] > 1800
        or state["free_disk_gib"] < 8
        or size > 100 * 1024 * 1024
    ):
        raise RuntimeError("Board study resource limit reached")
    return {**state, "output_bytes": size}


def counterfactual(schedule, cycle):
    contract = cycle // 32
    for alternative in valid_schedules():
        if (
            alternative[: contract + 1] == schedule[: contract + 1]
            and alternative[contract + 1 : contract + 3]
            != schedule[contract + 1 : contract + 3]
        ):
            return alternative
    return None


def fit_stream(
    arm, seed, episode, allocator, allocator_optimizer, probes, started, prep_seconds
):
    data_path = OUT / f"data-{seed}-{episode}.json"
    data = json.loads(data_path.read_text())
    directory = OUT / "runs" / f"{arm}-{seed}-{episode}"
    directory.mkdir(parents=True, exist_ok=False)
    training, board = episode < 4, arm.startswith("board_")
    student_seed = seed * 100 + episode
    student = Student(student_seed)
    history = History()
    chooser = torch.Generator().manual_seed(student_seed + 700000)
    initial_student = base.fingerprint(student.model.state_dict())
    initial_allocator = base.fingerprint(
        [allocator.state_dict(), allocator_optimizer.state_dict()]
    )
    initial_actor = base.fingerprint(allocator.policy.state_dict())
    credits, calls, income, allocator_updates = 4.0, 0, 0.0, 0
    rollout, evaluations, practice_counts, query_counts = [], [], [0, 0], [0, 0]
    evaluations.append(
        base.measure(
            student.model,
            student.optimizer,
            allocator,
            allocator_optimizer,
            probes,
            0,
            0,
        )
    )
    with (directory / "cycles.jsonl").open("x") as handle:
        for cycle in range(256):
            if cycle % 32 == 0:
                check_resources(started, prep_seconds)
            view = public_view(data["schedule"], cycle, board)
            obs = observation(view, cycle, credits, calls, history)
            with torch.no_grad():
                logits, value = allocator(obs)
                probabilities = logits.softmax(-1)
                action = int(torch.multinomial(probabilities, 1, generator=chooser))
                old_log = float(logits.log_softmax(-1)[action])
            skill, requested = action % 2 + 1, action >= 2
            eligible = allocator.tutor_enabled and credits >= 2 and calls < 12
            granted = requested and eligible
            alternative, cf = counterfactual(data["schedule"], cycle), None
            if not training and alternative is not None:
                altered_view = public_view(alternative, cycle, board)
                altered_obs = observation(altered_view, cycle, credits, calls, history)
                with torch.no_grad():
                    altered_probs = allocator(altered_obs)[0].softmax(-1)
                if not board:
                    assert torch.equal(obs, altered_obs) and torch.equal(
                        probabilities, altered_probs
                    )
                cf = {
                    "alternative_future": altered_view["future"],
                    "probabilities": altered_probs.tolist(),
                    "l1_change": float((altered_probs - probabilities).abs().sum()),
                }
            before = credits
            if granted:
                credits -= 2
                calls += 1
                query_counts[skill - 1] += 1
            practice_counts[skill - 1] += 1
            # The selected candidates become visible only after allocation.
            practice = learn_packet(
                student,
                data["practice"][cycle][skill - 1],
                buy_labels=granted,
                paid=False,
            )
            history.observe(skill, practice["correct"], cycle)
            paid = learn_packet(
                student, data["paid"][cycle], buy_labels=False, paid=True
            )
            history.observe(view["current"]["skill"], paid["correct"], cycle)
            income += paid["paid_income"]
            credits += paid["paid_income"]
            reward = (paid["paid_income"] - 2 * granted) / 8
            if training:
                rollout.append(
                    {
                        "obs": obs.detach(),
                        "action": action,
                        "log_prob": old_log,
                        "value": float(value),
                        "reward": reward,
                    }
                )
            step_updates = 0
            if training and (cycle + 1) % 16 == 0:
                bootstrap = 0.0
                if cycle + 1 < 256:
                    next_view = public_view(data["schedule"], cycle + 1, board)
                    with torch.no_grad():
                        bootstrap = float(
                            allocator(
                                observation(
                                    next_view, cycle + 1, credits, calls, history
                                )
                            )[1]
                        )
                base.update_ask(allocator, allocator_optimizer, rollout, bootstrap)
                allocator_updates += 4
                step_updates = 4
                rollout.clear()
            assert credits >= 0 and calls <= 12 and credits == 4 + income - calls * 2
            record = {
                "cycle": cycle,
                "view": view,
                "allocator_observation": obs.tolist(),
                "action_probabilities": probabilities.tolist(),
                "chosen_action": action,
                "practice_skill": skill,
                "query_requested": requested,
                "eligible": eligible,
                "granted": granted,
                "denied": requested and not eligible,
                "credits_before": before,
                "credits_after": credits,
                "calls": calls,
                "income": paid["paid_income"],
                "allocator_reward": reward,
                "allocator_optimizer_steps": step_updates,
                "practice": practice,
                "paid": paid,
                "future_counterfactual": cf,
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            if (cycle + 1) % 32 == 0:
                handle.flush()
                evaluations.append(
                    base.measure(
                        student.model,
                        student.optimizer,
                        allocator,
                        allocator_optimizer,
                        probes,
                        cycle + 1,
                        calls,
                    )
                )
    final_allocator = base.fingerprint(
        [allocator.state_dict(), allocator_optimizer.state_dict()]
    )
    if not training:
        assert initial_allocator == final_allocator
    assert student.updates == 2048 and allocator_updates == (64 if training else 0)
    result = {
        "arm": arm,
        "seed": seed,
        "episode": episode,
        "phase": data["phase"],
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "schedule": data["schedule"],
        "answer_initial_hash": initial_student,
        "answer_final_hash": base.fingerprint(student.model.state_dict()),
        "allocator_initial_hash": initial_allocator,
        "allocator_final_hash": final_allocator,
        "actor_initial_hash": initial_actor,
        "actor_final_hash": base.fingerprint(allocator.policy.state_dict()),
        "answer_updates": student.updates,
        "allocator_updates": allocator_updates,
        "practice_counts": practice_counts,
        "query_counts": query_counts,
        "calls": calls,
        "income": income,
        "net_income": income - calls * 2,
        "credits": credits,
        "evaluations": evaluations,
        "real_calls": 0,
    }
    base.save(directory / "summary.json", result)
    torch.save(
        {
            "answer_model": student.model.state_dict(),
            "answer_optimizer": student.optimizer.state_dict(),
            "allocator": allocator.state_dict(),
            "allocator_optimizer": allocator_optimizer.state_dict(),
            "memory": student.memory.rows,
            "memory_seen": dict(student.memory.seen),
            "history": vars(history),
            "student_rng": student.rng.get_state(),
            "allocator_rng": chooser.get_state(),
        },
        CHECKPOINTS / f"{arm}-{seed}-{episode}.pt",
    )
    print(
        json.dumps(
            {
                "completed": f"{arm}-{seed}-{episode}",
                "phase": data["phase"],
                "calls": calls,
                "resources": check_resources(started, prep_seconds),
            }
        ),
        flush=True,
    )
    return result


def run() -> None:
    verify()
    integrity = json.loads((OUT / "integrity.json").read_text())
    assert integrity["passed"]
    if (OUT / "runs").exists():
        raise RuntimeError(
            "Study execution already started; do not overwrite or resume silently"
        )
    socket.create_connection = no_network
    socket.socket.connect = no_network
    torch.set_num_threads(1)
    CHECKPOINTS.mkdir(parents=True, exist_ok=False)
    probes = json.loads((OUT / "probes.json").read_text())
    started = time.monotonic()
    result = {
        "complete": False,
        "real_calls": 0,
        "network_disabled_in_process": True,
        "runs": [],
    }
    try:
        for seed in SEEDS:
            for arm in ARMS:
                torch.manual_seed(seed + 900000)
                allocator = Allocator(
                    arm.endswith("_tutor") and not arm.endswith("no_tutor")
                )
                optimizer = torch.optim.Adam(allocator.parameters(), lr=0.001)
                for episode in range(6):
                    result["runs"].append(
                        fit_stream(
                            arm,
                            seed,
                            episode,
                            allocator,
                            optimizer,
                            probes,
                            started,
                            integrity["preparation_seconds"],
                        )
                    )
                    base.save(OUT / "partial.json", result)
        result["complete"] = True
    finally:
        verify()
        result.update(
            {
                "wall_seconds": time.monotonic() - started,
                "preparation_seconds": integrity["preparation_seconds"],
                "resources": check_resources(started, integrity["preparation_seconds"]),
                "prior_evidence_unchanged": True,
            }
        )
        base.save(OUT / "summary.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        run()
