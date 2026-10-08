"""Post-hoc frozen forward-pass analysis; no training, browser or API calls."""

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.run_interactive_teacher import RULE, SEEDS, NoTeacher, build, save
from scripts.run_local_learning import digest_state

OUTPUT = ROOT / "logs/cpu-browser-recovery"


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = OUTPUT / "posthoc-probabilities.json"
    if output.exists():
        raise FileExistsError("Preserve the existing analysis")
    preflight = json.loads((OUTPUT / "preflight.json").read_text())
    template = preflight["observations"]["left"]
    results = []
    for seed in SEEDS:
        saved = json.loads((OUTPUT / f"results/trained-{seed}-1536.json").read_text())
        for arm in ["fresh", "trained"]:
            agent = build(None, seed, NoTeacher(), "forward-pass-only")
            if arm == "trained":
                agent.learner.load(
                    str(ROOT / f"checkpoints/cpu-browser-recovery/seed-{seed}.pt")
                )
            agent.set_training(False)
            before = digest_state(agent.learner)
            subgoal = torch.from_numpy(agent.subgoal_encoder.encode(RULE)).unsqueeze(0)
            mask = torch.zeros(1, 32)
            mask[:, :7] = 1
            rows = []
            for index, ep in enumerate(saved["episodes"]):
                for row in ep["browser_trials"]:
                    probabilities = {}
                    for cue in ["left", "right"]:
                        observation = dict(template)
                        observation["dom_text"] = (
                            observation["dom_text"]
                            .replace("Current cue: left", "Current cue: " + cue)
                            .replace(
                                "Page practice, trial 1 of 24.",
                                f"Page held-out-{index}, trial {row['trial'] + 1} of 24.",
                            )
                        )
                        if cue == row["cue"]:
                            actual_hash = hashlib.sha256(
                                json.dumps(observation, sort_keys=True).encode()
                            ).hexdigest()
                            assert (
                                actual_hash == row["observation_sha256"]
                            ), "Reconstructed observation differs from actual held-out input"
                        state = torch.from_numpy(
                            agent.state_encoder.encode(observation)
                        ).unsqueeze(0)
                        with torch.no_grad():
                            p = (
                                agent.learner.policy(state, subgoal, mask)
                                .softmax(-1)[0]
                                .numpy()
                            )
                        probabilities[cue] = p[:7].tolist()
                    p = probabilities[row["cue"]]
                    answer = 0 if row["cue"] == "left" else 1
                    rows.append(
                        {
                            "page_index": index,
                            "trial": row["trial"],
                            "cue": row["cue"],
                            "correct_probability": p[answer],
                            "click_probability": p[0] + p[1],
                            "greedy_correct": int(np.argmax(p)) == answer,
                            "left_action_probability_cue_left": probabilities["left"][
                                0
                            ],
                            "left_action_probability_cue_right": probabilities["right"][
                                0
                            ],
                        }
                    )
            assert digest_state(agent.learner) == before
            results.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "mean_correct_probability": float(
                        np.mean([r["correct_probability"] for r in rows])
                    ),
                    "mean_click_probability": float(
                        np.mean([r["click_probability"] for r in rows])
                    ),
                    "greedy_accuracy": float(
                        np.mean([r["greedy_correct"] for r in rows])
                    ),
                    "cue_effect_on_left_action_probability": float(
                        np.mean(
                            [
                                r["left_action_probability_cue_left"]
                                - r["left_action_probability_cue_right"]
                                for r in rows
                            ]
                        )
                    ),
                    "frozen_state_unchanged": True,
                    "reconstructed_inputs_match_saved_hashes": True,
                    "cases": rows,
                }
            )
    save(
        output,
        {
            "post_hoc_diagnostic": True,
            "purpose": "Distinguish selecting buttons from using the visible cue; not additional training or a new task",
            "results": results,
        },
    )
    print(
        json.dumps(
            [{k: v for k, v in r.items() if k != "cases"} for r in results], indent=2
        )
    )


if __name__ == "__main__":
    main()
