# Repository Guidelines

## Project Structure & Module Organization
Source for interactive agents lives in `agents/`, with the PPO+RND learner and utilities in `rl/`, BrowserGym adapters inside `envs/`, and LLM prompts plus memory tooling under `llm/`. Evaluation harnesses sit in `eval/`, reusable configs in `configs/`, and runnable entrypoints (explore, eval, ablations) in `scripts/`. Keep generated checkpoints under `checkpoints/` and logs, CSVs, and TensorBoard runs beneath `logs/`. Tests mirror the code layout inside `tests/`.

## Build, Test, and Development Commands
Run `pip install -r requirements.txt` after cloning or updating dependencies. Launch exploratory training with `python scripts/run_explore.py --agent hybrid --env browsergym/miniwob.click-checkboxes --steps 200000 --tensorboard logs/tb/hybrid`. Evaluate checkpoints via `python scripts/run_eval.py --agent hybrid --checkpoint checkpoints/hybrid_click.pt --episodes 100 --frozen`. Execute targeted smoke tests using `pytest tests/agents/test_hybrid.py`; run the full regression suite with `pytest`. Start live metric dashboards by running `tensorboard --logdir logs/tb`.

## Coding Style & Naming Conventions
Target Python 3.11 with 4-space indentation and explicit type hints on new public APIs. Use snake_case for variables/functions, PascalCase for classes, and keep module filenames lowercase. Apply `ruff check .` and `black .` before raising a pull request; project defaults are configured through `pyproject.toml`. Keep docstrings concise, explaining intent and inputs for coach hooks, encoders, and environment wrappers.

## Testing Guidelines
Add or update tests in the package-matching directory (e.g., `tests/rl/` when touching `rl/`). Seed randomness via helpers in `configs/` and prefer mocking BrowserGym IO where deterministic outcomes are required. Name tests `test_<behavior>()` to clarify coverage. Run `pytest --maxfail=1 --disable-warnings -q` locally and ensure new logging, checkpointing, or resume paths gain regression coverage.

## Commit & Pull Request Guidelines
Follow the existing imperative commit style (`Add tensorboard dependency`, `Improve exploration tooling with logging and resume support`). Group related edits into atomic commits and avoid bundling generated artifacts. Pull requests should state motivation, list reproduction or validation commands, call out configuration updates (`.env`, `configs/*.yaml`), and include screenshots or log snippets for telemetry-facing changes. Reference relevant issues and request reviews from maintainers owning the touched modules.

## Configuration & Agent Operations
Copy `.env.example` to `.env` and populate secrets locally; never commit credentials. Ensure `MINIWOB_URL` targets the served MiniWoB++ root with the `/miniwob/` suffix. Write checkpoints into `checkpoints/`, append telemetry to `logs/`, and keep long-form traces in JSONL via `--action-trace-file`. Use `--resume-from <checkpoint>` and verify continuity in TensorBoard under `logs/tb/` when restarting experiments.
