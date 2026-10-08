"""Read-only MiniWoB probe using local assets and a fresh headless browser."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

from agents.hybrid import HybridAgent
from envs.browsergym_client import BrowserGymEnvWrapper
from llm.coach import CoachDirective


class OfflineCoach:
    def advise(self, **kwargs):
        return CoachDirective(subgoal="Follow the displayed instruction.")


def main():
    env = BrowserGymEnvWrapper(
        "browsergym/miniwob.click-checkboxes", max_episode_steps=10
    )
    try:
        obs, info = env.reset(seed=100, return_info=True)
        agent = HybridAgent(env, OfflineCoach(), max_steps=10)
        agent._extract_checkbox_targets(obs)
        print(
            json.dumps(
                {
                    "keys": list(obs),
                    "goal": obs.get("goal"),
                    "catalog": agent._action_catalog(obs)[1],
                    "info": info,
                    "targets": list(agent._checkbox_targets),
                },
                default=str,
            )
        )
        (ROOT / "logs/audit/browser-observation.json").write_text(
            json.dumps(obs, default=str)
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
