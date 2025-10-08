FLAPPY v0.1
============

FLAPPY is a research platform for continually learning web agents on MiniWoB++ tasks via BrowserGym. It compares three baselines:

* Pure reinforcement learning agent (PPO + Random Network Distillation)
* Coach-guided random agent (LLM coach for masks/subgoals, random policy)
* Hybrid coach/learner agent (LLM coach + PPO/RND driver policy)

The platform targets PyTorch 2.x, Python 3.11, and runs headless Chrome via BrowserGym.

Repository layout
-----------------

```
envs/        # BrowserGym wrappers and selector helpers
llm/         # GPT-5 mini client, coach prompts, episodic memory
rl/          # PPO+RND learner, context encoders, buffers
agents/      # Agent interfaces (RL baseline, hybrid coach-driven)
eval/        # Task lists, metrics, evaluation harness
scripts/     # CLI entrypoints for exploration, eval, ablations
configs/     # YAML configs for seeds and hyperparameters
tests/       # Unit tests
```

Quick start
-----------

1. Install system dependencies (Python 3.11, poetry or pip, Chrome/Chromium).
2. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
3. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Launch MiniWoB++ via BrowserGym (see `scripts/bootstrap_browsergym.sh` once added) and run dry evals:

   ```bash
   python scripts/run_eval.py --agent coach_random --env miniwob/click-checkboxes
   python scripts/run_eval.py --agent hybrid --env miniwob/click-checkboxes
   ```

5. Monitor TensorBoard logs in `runs/`.

Core components
---------------

* **BrowserGym integration** via `envs/browsergym_client.py`, with DOM-derived action candidates from `envs/selectors.py`.
* **LLM coach** in `llm/coach.py` with advisory prompts in `llm/prompts.py`, and episodic memory in `llm/memory.py`.
* **PPO+RND learner** in `rl/rnd_ppo_agent.py`, with observation embeddings from `rl/features.py` and masked policy utilities in `rl/policy.py`.
* **Agent interfaces** under `agents/` implementing the Pure RL and Hybrid coach-driven controllers.
* **Evaluation harness** in `eval/harness.py` with metrics defined in `eval/metrics.py` and task configs in `eval/tasks.yaml`.

Development setup
-----------------

* Format with `ruff` and `black` (configured via `pyproject.toml`, forthcoming).
* Run unit tests with `pytest`.
* Use `configs/default.yaml` to control seeds, logging dirs, and training params.

Roadmap
-------

* v0.1: MiniWoB++ support, PPO+RND, Reflexion memory.
* v0.2: WebArena backend, advanced curiosity modules, Dreamer-style world models.

License
-------

See `LICENSE`.
