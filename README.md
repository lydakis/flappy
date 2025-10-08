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
2. Copy `.env.example` to `.env` and set required keys, including `OPENAI_API_KEY` and
   a reachable `MINIWOB_URL` (see BrowserGym docs for self-hosting). The CLI scripts
   load this file automatically via `python-dotenv`.
3. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Launch MiniWoB++ via the Farama miniwob-plusplus assets. Clone and serve the HTML root:

   ```bash
   git clone git@github.com:Farama-Foundation/miniwob-plusplus.git
   cd miniwob-plusplus/miniwob/html
   python3 -m http.server 8890
   ```

   Set `MINIWOB_URL=http://127.0.0.1:8890/miniwob/` in `.env` (note the `/miniwob/` suffix) and run dry evals:

   ```bash
   python scripts/run_eval.py --agent coach_random --env browsergym/miniwob.click-checkboxes
   python scripts/run_eval.py --agent hybrid --env browsergym/miniwob.click-checkboxes
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
Training & evaluation workflow
------------------------------

1. **Train / explore** with the hybrid agent. The example below collects 200k steps, logs progress, and writes checkpoints/metrics:

   ```bash
   python scripts/run_explore.py \
     --agent hybrid \
     --env browsergym/miniwob.click-checkboxes \
     --steps 200000 \
     --save-path checkpoints/hybrid_click.pt \
     --save-every 50000 \
     --log-interval 5 \
     --log-file logs/hybrid_click.csv \
     --tensorboard logs/tb/hybrid \
     --action-trace-file logs/hybrid_click_trace.jsonl \
     --no-headless
   ```

   Use `--resume-from checkpoints/hybrid_click.pt` to continue training from an existing checkpoint.

   Episode metrics are emitted to stdout, the CSV (if provided), and TensorBoard summaries:

   * `reward` – extrinsic task reward per episode (should rise toward success)
   * `intrinsic_reward` – RND curiosity bonus (decays as the policy covers the DOM)
   * `success` – 1 if the task succeeded, 0 otherwise
   * `coach_interventions` – coach calls per episode (drops as the policy internalises subgoals)

   Launch TensorBoard in another terminal to inspect the curves live:

   ```bash
   tensorboard --logdir logs/tb/hybrid
   ```

   If you supply `--action-trace-file`, each episode’s low-level actions (clicks, types, etc.) are appended as JSON lines so you can audit what the agent attempted.

2. **Evaluate** any agent (including the trained hybrid) with frozen weights:

   ```bash
   python scripts/run_eval.py \
     --agent hybrid \
     --env browsergym/miniwob.click-checkboxes \
     --episodes 100 \
     --checkpoint checkpoints/hybrid_click.pt \
     --frozen \
     --no-headless
   ```

3. **Compare baselines** by swapping `--agent` for `coach_random` or `baseline_rl`.
