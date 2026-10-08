#!/usr/bin/env python3
"""Fixed-budget numeric/DSL/templated-language transfer; no network or APIs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.run_adaptive_curriculum import fingerprint, resources, save

OUTPUT = ROOT / "logs/representation-transfer"
DOMAINS = ["simulation", "code", "language"]
ARMS = ["mixed", *DOMAINS]
SEEDS = [7, 19, 43]
CALIBRATION_SEED = 101
BUDGET = 12288
BATCH = 32
CHECKPOINTS = [0, 4096, 8192, BUDGET]
PROBE_CASES = 384
MAX_SECONDS = 600
SPLITS = {"calibration_train": 0, "calibration_test": 1, "train": 2, "test": 3}
TEMPLATES = {
    "simulation": [
        "simulation {{ position : {p} , velocity : {v} , steps : {t} }} query : position > 0 after steps",
        "simulation {{ velocity : {v} , position : {p} , steps : {t} }} query : position > 0 after steps",
        "simulation {{ steps : {t} , position : {p} , velocity : {v} }} query : position > 0 after steps",
        "simulation {{ steps : {t} , velocity : {v} , position : {p} }} query : position > 0 after steps",
    ],
    "code": [
        "position = {p}; velocity = {v}; steps = {t}; output(position + velocity * steps > 0)",
        "velocity = {v}; position = {p}; steps = {t}; output(0 < position + steps * velocity)",
        "steps = {t}; position = {p}; velocity = {v}; output(velocity * steps + position > 0)",
        "steps = {t}; velocity = {v}; position = {p}; output(0 < steps * velocity + position)",
    ],
    "language": [
        "with position {p} and velocity {v} per step , after {t} steps is position greater than zero ?",
        "with velocity {v} per step and position {p} , after {t} steps is position greater than zero ?",
        "after {t} steps , with position {p} and velocity {v} per step is position greater than zero ?",
        "after {t} steps , with velocity {v} per step and position {p} is position greater than zero ?",
    ],
}
TOKEN = re.compile(r"-?\d+(?:\.\d+)?|[a-z_]+|[^\s]")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?\Z")


def simulate(position: float, velocity: float, steps: int) -> int:
    """One-dimensional constant-velocity simulator, with bounded integer steps."""
    if steps not in (1, 2):
        raise ValueError("Only one or two simulation steps are supported")
    for _ in range(steps):
        position += velocity
    return int(position > 0)


def execute_program(program: str) -> int:
    """Interpret a tiny validated AST; never execute Python, eval, imports or calls."""
    if len(program) > 300:
        raise ValueError("Program too long")
    try:
        body = ast.parse(program).body
    except SyntaxError as exc:
        raise ValueError("Invalid grammar") from exc
    if len(body) != 4:
        raise ValueError("Exactly three assignments and one output are required")
    bindings = {}

    def literal(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = float(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if not isinstance(node.operand, ast.Constant) or type(
                node.operand.value
            ) not in (int, float):
                raise ValueError("Invalid numeric literal")
            value = -float(node.operand.value)
        else:
            raise ValueError("Assignments require numeric literals")
        if not np.isfinite(value) or abs(value) > 2:
            raise ValueError("Literal out of bounds")
        return value

    for node in body[:3]:
        if (
            not isinstance(node, ast.Assign)
            or len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
        ):
            raise ValueError("Only simple assignments are supported")
        name = node.targets[0].id
        if name not in {"position", "velocity", "steps"} or name in bindings:
            raise ValueError("Unknown or duplicate variable")
        bindings[name] = literal(node.value)
    if set(bindings) != {"position", "velocity", "steps"} or bindings["steps"] not in (
        1,
        2,
    ):
        raise ValueError("Invalid simulation fields")
    final = body[3]
    if not isinstance(final, ast.Expr) or not isinstance(final.value, ast.Call):
        # Reject parsed grammar here, rather than caller argument types.
        raise ValueError("Invalid output grammar")  # noqa: TRY004
    call = final.value
    if (
        not isinstance(call.func, ast.Name)
        or call.func.id != "output"
        or len(call.args) != 1
        or call.keywords
    ):
        raise ValueError("Only a single output marker is supported")

    def arithmetic(node: ast.AST, depth: int = 0) -> float:
        if depth > 4:
            raise ValueError("Expression too deep")
        if isinstance(node, ast.Name) and node.id in bindings:
            return bindings[node.id]
        if (
            isinstance(node, ast.Constant)
            and type(node.value) is int
            and node.value == 0
        ):
            return 0.0
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
            a, b = arithmetic(node.left, depth + 1), arithmetic(node.right, depth + 1)
            return a + b if isinstance(node.op, ast.Add) else a * b
        raise ValueError("Unsupported expression")

    comparison = call.args[0]
    if (
        not isinstance(comparison, ast.Compare)
        or len(comparison.ops) != 1
        or len(comparison.comparators) != 1
    ):
        raise ValueError("Exactly one comparison is supported")
    a, b = arithmetic(comparison.left), arithmetic(comparison.comparators[0])
    if isinstance(comparison.ops[0], ast.Gt):
        return int(a > b)
    if isinstance(comparison.ops[0], ast.Lt):
        return int(a < b)
    raise ValueError("Only less-than and greater-than comparisons are supported")


def latent_key(case: dict) -> str:
    return f"{case['position_milli']}:{case['velocity_milli']}:{case['steps']}"


def generate_cases(split: str, seed: int, count: int) -> list[dict]:
    rng = np.random.default_rng(np.random.SeedSequence([91021, SPLITS[split], seed]))
    cases, used = [], set()
    while len(cases) < count:
        p, v = map(int, rng.integers(-2000, 2001, size=2))
        steps = 1 + (len(cases) // 4) % 2
        margin = p + v * steps
        case = {"position_milli": p, "velocity_milli": v, "steps": steps}
        key = latent_key(case)
        if not (250 <= abs(margin) <= 2000) or int(margin > 0) != len(cases) % 2:
            continue
        # Require opposed position/velocity signs, balanced independently of label.
        if p * v >= 0 or int(v > 0) != (len(cases) // 2) % 2:
            continue
        if key in used or hashlib.sha256(key.encode()).digest()[0] % 4 != SPLITS[split]:
            continue
        used.add(key)
        case.update(
            {
                "id": f"{split}:{seed}:{len(cases)}",
                "label": simulate(p / 1000, v / 1000, steps),
            }
        )
        assert case["label"] == int(margin > 0)
        cases.append(case)
    return cases


def render(case: dict, domain: str, template: int) -> str:
    return TEMPLATES[domain][template].format(
        p=f"{case['position_milli'] / 1000:.3f}",
        v=f"{case['velocity_milli'] / 1000:.3f}",
        t=case["steps"],
    )


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def vocabulary() -> dict[str, int]:
    words = set()
    dummy = {"position_milli": -1234, "velocity_milli": 567, "steps": 2}
    # Declared input alphabet, no learned embeddings or labels from test examples.
    for domain in DOMAINS:
        for template in range(4):
            words.update(
                "<num>" if NUMBER.fullmatch(t) else t
                for t in tokens(render(dummy, domain, template))
            )
    return {"<pad>": 0, **{word: i + 1 for i, word in enumerate(sorted(words))}}


VOCAB = vocabulary()
MAX_LENGTH = max(
    len(
        tokens(
            render({"position_milli": -1234, "velocity_milli": 567, "steps": 2}, d, t)
        )
    )
    for d in DOMAINS
    for t in range(4)
)


def encode(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = torch.zeros((len(texts), MAX_LENGTH), dtype=torch.long)
    values = torch.zeros((len(texts), MAX_LENGTH, 1))
    mask = torch.zeros((len(texts), MAX_LENGTH), dtype=torch.bool)
    for row, text in enumerate(texts):
        pieces = tokens(text)
        if len(pieces) > MAX_LENGTH:
            raise ValueError("Input exceeds declared sequence length")
        for col, piece in enumerate(pieces):
            numeric = NUMBER.fullmatch(piece) is not None
            ids[row, col] = VOCAB["<num>" if numeric else piece]
            values[row, col, 0] = float(piece) / 2 if numeric else 0
            mask[row, col] = True
    return ids, values, mask


class SequenceStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(VOCAB), 16, padding_idx=0)
        self.encoder = nn.GRU(17, 48, batch_first=True)
        self.answer = nn.Sequential(nn.Linear(48, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(
        self, ids: torch.Tensor, values: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        sequence, _ = self.encoder(torch.cat([self.embedding(ids), values], dim=-1))
        pooled = (sequence * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return self.answer(pooled)


def training_examples(cases: list[dict], arm: str) -> list[dict]:
    examples = []
    for index, case in enumerate(cases):
        domain = DOMAINS[index % 3] if arm == "mixed" else arm
        # Both labels occur within each domain/template; neither is answer-coded.
        template = (index // 24) % 2
        rendered = render(case, domain, template)
        if domain == "code":
            assert execute_program(rendered) == case["label"]
        examples.append(
            {
                "latent_id": case["id"],
                "domain": domain,
                "template": template,
                "input": rendered,
                "label": case["label"],
            }
        )
    return examples


def probe_examples(cases: list[dict]) -> list[dict]:
    return [
        {
            "latent_id": case["id"],
            "domain": domain,
            "group": group,
            "template": base + (index // 8) % 2,
            "input": render(case, domain, base + (index // 8) % 2),
            "label": case["label"],
        }
        for domain in DOMAINS
        for group, base in [("seen_template", 0), ("heldout_template", 2)]
        for index, case in enumerate(cases)
    ]


@torch.no_grad()
def evaluate(
    model: SequenceStudent,
    optimizer: torch.optim.Optimizer,
    examples: list[dict],
    count: int,
) -> dict:
    before = fingerprint([model.state_dict(), optimizer.state_dict()])
    rng = torch.get_rng_state().clone()
    encoded = encode([e["input"] for e in examples])
    probabilities = []
    for start in range(0, len(examples), 256):
        probabilities.append(
            model(*(x[start : start + 256] for x in encoded)).softmax(-1)
        )
    probabilities = torch.cat(probabilities).numpy()
    labels = np.array([e["label"] for e in examples])
    results = []
    for domain in DOMAINS:
        for group in ["seen_template", "heldout_template"]:
            indices = np.array(
                [
                    i
                    for i, e in enumerate(examples)
                    if e["domain"] == domain and e["group"] == group
                ]
            )
            selected = probabilities[indices]
            correct = selected.argmax(-1) == labels[indices]
            results.append(
                {
                    "domain": domain,
                    "group": group,
                    "cases": len(indices),
                    "accuracy": float(correct.mean()),
                    "mean_correct_label_probability": float(
                        selected[np.arange(len(indices)), labels[indices]].mean()
                    ),
                    "confident_error_fraction": float(
                        (~correct & (selected.max(-1) >= 0.9)).mean()
                    ),
                }
            )
    after = fingerprint([model.state_dict(), optimizer.state_dict()])
    assert before == after and torch.equal(rng, torch.get_rng_state())
    return {
        "examples_seen": count,
        "labels_used": count,
        "updates": count // BATCH,
        "metrics": results,
        "state_hash_before": before,
        "state_hash_after": after,
    }


def metric(evaluation: dict, domain: str, group: str) -> float:
    return next(
        m["accuracy"]
        for m in evaluation["metrics"]
        if m["domain"] == domain and m["group"] == group
    )


def calibration_passes(results: list[dict]) -> bool:
    return (
        len(results) == 3
        and {r["arm"] for r in results} == set(DOMAINS)
        and all(
            metric(r["evaluations"][-1], r["arm"], "seen_template") >= 0.85
            and metric(r["evaluations"][-1], r["arm"], "heldout_template") >= 0.75
            for r in results
        )
    )


def prior_hashes() -> dict[str, str]:
    # The public reproduction is independent of private historical artifacts.
    return {}


def prepare() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    model = SequenceStudent()
    plan = {
        "concept": "constant-velocity 1D motion: is position positive after one or two steps? p,v in [-2,2] on 0.001 grid, |p+v*steps| in [0.25,2]; opposing position/velocity signs; answer, velocity sign and steps balanced independently",
        "domains": DOMAINS,
        "arms": ARMS,
        "seeds": SEEDS,
        "templates": TEMPLATES,
        "training_template_ids": [0, 1],
        "heldout_template_ids": [2, 3],
        "representation_limits": "numeric simulation records, not pixels; templated English, not natural-language competence; predicting outputs of fixed programs, not code generation; no pretrained encoder",
        "architecture": "shared word/operator embedding16 plus one numeric-value channel(value/2); one GRU48 with masked mean pooling; MLP48->64ReLU->2; no field parser or simulator results supplied to neural input",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "vocabulary": VOCAB,
        "max_sequence_length": MAX_LENGTH,
        "interface_control": "position/velocity/steps identifiers shared across formats; every held-out-template symbol appears in that modality's training templates; known alphabet only, random learned embeddings; zero target training still has syntax/interface confounds",
        "training": {
            "examples_per_arm_seed": BUDGET,
            "teacher_labels_per_arm_seed": BUDGET,
            "batch_size": BATCH,
            "updates": BUDGET // BATCH,
            "optimizer": "Adam lr0.003; cross-entropy; clip1; one pass; no replay, early stopping or label gating",
            "checkpoints": CHECKPOINTS,
            "mixed_domain_exposure": {d: BUDGET // 3 for d in DOMAINS},
            "single_domain_exposure": "12288 source, zero examples in each target",
            "paired_controls": "same seed and complete latent sequence across all four arms; each latent case appears exactly once to each learner in one representation; same capacity/updates/labels; no aligned duplicate views in mixed training",
        },
        "splits": "SHA256 of numeric latent tuple partitions into four disjoint universes: calibration_train/test and main train/test; no identifier or partition feature enters model; latent tuples unique within every training/probe set",
        "evaluation": {
            "latent_cases_per_probe": PROBE_CASES,
            "final_probe_seed": 20261008,
            "calibration_probe_seed": 99101,
            "domain_template_buckets": 6,
            "balanced": True,
            "frozen": True,
            "same_latent_cases_paired_across_eval_buckets_only": True,
        },
        "calibration": {
            "seed": CALIBRATION_SEED,
            "runs": "one independently initialized full-budget specialist for each domain; no mixed run or comparison used",
            "maximum_examples_and_labels": 3 * BUDGET,
            "gate": "every specialist own-domain accuracy>=85% seen templates and>=75% held-out templates on independent calibration probe",
            "failure": "stop entire comparison; report interface/learnability blocker; no tuning or extra candidates",
        },
        "primary_success": "on held-out templates, mixed minus corresponding full-budget specialist, averaged over the three domains, >=2pp in >=2/3 seeds, with no mixed domain worse by >5pp in any seed; each comparison matches total training/labels; specialist diagonal average is a summary of three separate controls, not a jointly trained system",
        "secondary": "full source-target accuracy matrix with exact target exposure; zero-target transfer>=75% held-out accuracy is descriptive and interpreted only alongside source competence; mixed-final vs specialist-4096 matches target exposure but has unequal total budgets and cannot establish a matched-budget advantage; report all checkpoints, not selected favorable ones",
        "scope_limits": "all templates implement the same fixed motion function; unseen templates change ordering and code expression composition, not physics; possible shortcuts and familiar shared field names limit conceptual claims; no learned curriculum/help/continual memory claim",
        "stops": "fixed budgets, failed calibration, nonfinite update, evaluation mutation, source/prior evidence hash mismatch, phase elapsed>600s, RSS>1800MiB or disk<8GiB; CPU1 nice15",
        "prior_hashes": prior_hashes(),
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "new_paid_calls": 0,
    }
    save(OUTPUT / "plan.json", plan)
    paths = [
        ROOT / "scripts/run_representation_transfer.py",
        ROOT / "scripts/run_adaptive_curriculum.py",
        ROOT / "rl/policy.py",
        OUTPUT / "plan.json",
    ]
    for split, seed, count in [
        ("calibration_train", 101, BUDGET),
        ("calibration_test", 99101, PROBE_CASES),
        ("test", 20261008, PROBE_CASES),
        *[("train", seed, BUDGET) for seed in SEEDS],
    ]:
        path = OUTPUT / f"latent-{split}-{seed}.json"
        save(path, generate_cases(split, seed, count))
        paths.append(path)
    save(
        OUTPUT / "source-data-hashes.json",
        {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
    )


def verify_inputs() -> None:
    for name, digest in json.loads(
        (OUTPUT / "source-data-hashes.json").read_text()
    ).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert (
        prior_hashes() == json.loads((OUTPUT / "plan.json").read_text())["prior_hashes"]
    )


def fit(
    arm: str,
    seed: int,
    cases: list[dict],
    probe: list[dict],
    directory: Path,
    started: float,
) -> dict:
    directory.mkdir(exist_ok=False)
    torch.manual_seed(seed)
    model = SequenceStudent()
    initial_hash = fingerprint(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    examples = training_examples(cases, arm)
    inputs = encode([e["input"] for e in examples])
    labels = torch.tensor([e["label"] for e in examples])
    evaluations = [evaluate(model, optimizer, probe, 0)]
    exposures = {d: 0 for d in DOMAINS}
    with (directory / "batches.jsonl").open("x") as stream:
        for start in range(0, len(cases), BATCH):
            if time.monotonic() - started > MAX_SECONDS:
                raise TimeoutError("Representation study phase time limit")
            optimizer.zero_grad()
            logits = model(*(x[start : start + BATCH] for x in inputs))
            loss = F.cross_entropy(logits, labels[start : start + BATCH])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            for e in examples[start : start + BATCH]:
                exposures[e["domain"]] += 1
            count = start + BATCH
            stream.write(
                json.dumps(
                    {
                        "examples_seen": count,
                        "teacher_labels": count,
                        "updates": count // BATCH,
                        "loss": float(loss.detach()),
                        "gradient_norm_before_clip": float(norm),
                        "domain_exposure": dict(exposures),
                        "latent_ids": [
                            e["latent_id"] for e in examples[start : start + BATCH]
                        ],
                        "labels": [e["label"] for e in examples[start : start + BATCH]],
                    }
                )
                + "\n"
            )
            if count % 1024 == 0:
                resources()
                stream.flush()
            if count in CHECKPOINTS:
                evaluations.append(evaluate(model, optimizer, probe, count))
    result = {
        "arm": arm,
        "seed": seed,
        "initial_weight_hash": initial_hash,
        "final_weight_hash": fingerprint(model.state_dict()),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "teacher_labels": len(cases),
        "updates": len(cases) // BATCH,
        "domain_exposure": exposures,
        "evaluations": evaluations,
        "resources": resources(),
        "phase_elapsed_seconds": time.monotonic() - started,
    }
    save(directory / "summary.json", result)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "arm": arm,
            "seed": seed,
            "domain_exposure": exposures,
        },
        directory / "checkpoint.pt",
    )
    print(
        json.dumps(
            {
                "arm": arm,
                "seed": seed,
                "own_or_macro_final_seen": float(
                    np.mean(
                        [
                            metric(evaluations[-1], d, "seen_template")
                            for d in (DOMAINS if arm == "mixed" else [arm])
                        ]
                    )
                ),
                "own_or_macro_final_heldout": float(
                    np.mean(
                        [
                            metric(evaluations[-1], d, "heldout_template")
                            for d in (DOMAINS if arm == "mixed" else [arm])
                        ]
                    )
                ),
                "phase_elapsed_seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )
    return result


def run_phase(phase: str) -> None:
    verify_inputs()
    directory = OUTPUT / phase
    directory.mkdir(exist_ok=False)
    calibration = phase == "calibration"
    if not calibration:
        prior = json.loads((OUTPUT / "calibration-summary.json").read_text())
        if not calibration_passes(prior["runs"]):
            raise RuntimeError(
                "Calibration did not qualify; main comparison prohibited"
            )
    split, probe_seed = (
        ("calibration_test", 99101) if calibration else ("test", 20261008)
    )
    probe = probe_examples(
        json.loads((OUTPUT / f"latent-{split}-{probe_seed}.json").read_text())
    )
    save(directory / "evaluation-inputs.json", probe)
    started = time.monotonic()
    results, status = [], {"complete": False, "phase": phase, "new_paid_calls": 0}
    try:
        for seed in ([101] if calibration else SEEDS):
            train_split = "calibration_train" if calibration else "train"
            cases = json.loads(
                (OUTPUT / f"latent-{train_split}-{seed}.json").read_text()
            )
            for arm in (DOMAINS if calibration else ARMS):
                results.append(
                    fit(arm, seed, cases, probe, directory / f"{arm}-{seed}", started)
                )
                save(OUTPUT / f"{phase}-partial.json", results)
        status["complete"] = True
        if calibration:
            status["passed"] = calibration_passes(results)
            if not status["passed"]:
                status["blocker"] = (
                    "At least one declared specialist failed independent source or held-out-template learnability gates; no main comparison or tuning permitted"
                )
    except Exception as exc:
        status["stop_reason"] = str(exc)
        raise
    finally:
        verify_inputs()
        status.update(
            {
                "runs": results,
                "prior_evidence_and_source_unchanged": True,
                "wall_seconds": time.monotonic() - started,
                "resources": resources(),
            }
        )
        save(OUTPUT / f"{phase}-summary.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "calibration", "main"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    prepare() if args.phase == "prepare" else run_phase(args.phase)
