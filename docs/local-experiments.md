# Local learning experiments

These CPU diagnostics separate actual optimization from scaffolding and frozen
evaluation. The default browser controller is a small feed-forward PPO policy,
not a language model. See [the architecture inventory](architectures.md).

## Fixes

- Frozen hybrid evaluation disables PPO transitions, RND optimizer/statistic
  updates and reflection writes, then restores the caller's settings.
- Sampling errors stop visibly instead of silently replacing the learner with a
  random controller. Update counters, losses and failure counts are logged.
- Zero-valued action masks prohibit actions; padded action slots stay masked.
  The legacy mask head is disabled by default because it has no training loss.
- RND uses a running variance rather than an accumulated squared-deviation sum.
  Running statistics and optimizer/update counters survive checkpoints.
- Episode-local intervention and guardrail state reset, rewards are accumulated,
  and the agent's finite horizon ends its rollout trajectory.
- Browser clicks use Playwright actionability checks. Keyboard presses omit an
  unsupported timeout argument. The regression reproduces previously lost clicks.
- Pure-RL evaluation no longer constructs an API client.

## Setup and checks

Use Python 3.11 in an isolated environment. The audit dependency snapshot records
the versions used for the historical runs; it is not a promise of compatibility
with arbitrary newer versions.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r configs/local-audit-requirements.txt
export PYTHON_DOTENV_DISABLED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m pytest --maxfail=1 --disable-warnings -q
```

The normal tests use synthetic data and mocked API clients. They do not require
API keys or saved provider-account records. The optional real-browser regression
requires an already installed Playwright Chromium:

```sh
RUN_LOCAL_BROWSER_TESTS=1 .venv/bin/python -m pytest -q tests/test_browsergym_click_delivery.py
```

Set `PLAYWRIGHT_BROWSERS_PATH` if using a non-default installation. The test opens
only the local fixture page; it does not train or call a tutor.

## Offline reproduction commands

Run these only when intentionally starting a new experiment. Preparation refuses
to overwrite existing output. Generated files belong in ignored `logs/` and
`checkpoints/`; archive previous results rather than resetting a study to hunt
for a favorable seed. Each runner limits CPU threads and checks resource use.
The examples below are independent alternatives, not an instruction to run all
experiments. On a shared machine use `nice -n 15` and inspect resource limits first.

For a delayed cue, compare identical four-screen information in an MLP and a
two-layer transformer. The current-screen control lacks the earlier cue:

```sh
.venv/bin/python scripts/run_local_learning.py --arch history --delay 3 --steps 8192 --seeds 7 19 43 --max-seconds 600 --output logs/cue-history
.venv/bin/python scripts/run_local_learning.py --arch transformer --delay 3 --steps 8192 --seeds 7 19 43 --max-seconds 600 --output logs/cue-transformer
```

The offline coach supplies a constant subgoal and no answer mask. Fresh and
random controls get the same permitted observations/scaffolding. Frozen
evaluations check model, optimizer and RNG state. This tests within-episode
memory, not continual retention.

Other entrypoints:

| Question | Prepare | Execute | Analyze |
|---|---|---|---|
| Local browser recovery | `run_cpu_browser_recovery.py prepare` | `run_cpu_browser_recovery.py run` | `summarize_cpu_browser_recovery.py` |
| Easy curriculum | `run_adaptive_curriculum.py prepare` | `run_adaptive_curriculum.py verify`, then `run` if the gate passes | `analyze_adaptive_curriculum.py` |
| Harder curriculum | `run_harder_curriculum.py prepare` | `run_harder_curriculum.py calibrate`, then `run` if the gate passes | `analyze_harder_curriculum.py` |
| Representation transfer | `run_representation_transfer.py prepare` | `run_representation_transfer.py calibration`, then `main` if the gate passes | `analyze_representation_transfer.py` |
| Feedback replay | `run_offline_tutor_repair.py prepare` | `run_offline_tutor_repair.py run` | `analyze_offline_tutor_repair.py` |
| Task board | `run_task_board_study.py prepare` | `run_task_board_study.py run` | `analyze_task_board_study.py` |

Prefix entries with `.venv/bin/python scripts/`. All these studies use an
explicit deterministic oracle or no coach, not a real LLM. Run tests before
preparation so the resulting source/data manifest freezes tested code.
The final two studies disable socket connections in the execution process.
Do not treat passing unit tests as passing a learnability/retention gate.

The board analyzer also reads an existing evidence ZIP using `--bundle PATH
--output-dir PATH`, without training or extracting large logs. Its source/data
hash checks require the exact source saved with that run; publication formatting
does not retroactively change historical hashes.

## Historical results and limits

These are descriptive, small-seed results, not benchmark or significance claims.
Publication did not rerun experiments or change their acceptance thresholds.

- CPU PPO updates were observable after the evaluation/sampling fixes. On a
  delayed cue, a history MLP matched the tiny transformer. This did not establish
  a need for a GPU or language understanding.
- A browser controller could improve its reward while collapsing to one button;
  success must be checked per cue and against fresh/random controls.
- Easy and harder supervised curricula were near ceiling and did not establish
  the declared adaptive-curriculum advantage.
- Representation transfer failed its independent learnability gate. The main
  comparison was stopped; no favorable result was substituted.
- The original continual-tutor stream exposed a positional label pattern.
  Its query-value comparison is invalid. The corrected generator shuffles packet
  order; the later repair samples classes independently and hides label metadata.
- Balanced past-feedback replay improved retained skill accuracy, but three of
  nine runs still failed the original retention gate. That failure is preserved.
- A separately specified 72-stream board study maintained skills but found no
  useful board or tutor advantage. Returning-job accuracy was 96.82% current-only
  versus 96.85% with the board, without tutoring. The paired-seed mean difference
  was +0.026 percentage points, exploratory 95% t interval [-2.98, +3.03], based
  on only three seeds. All controls had less than five points of headroom, so none
  could meet the five-point board-improvement criterion. No maintenance loss
  exceeded ten points in 76 assessment returns; the maximum was 2.34 points.
  This extra-practice schedule does not overturn the earlier failed gate.
- Perfect-oracle tutor labels did not achieve the board study's required net
  synthetic-income advantage after query fees. That does not justify a paid
  tutor or GPU extension of this design.

The public checkout contains source, tests, fixed scientific plans and this
summary. Private conversations, approvals, account ledgers, API response logs,
machine paths, checkpoints and raw evidence bundles are intentionally excluded.
The original archival checkout and evidence remain unchanged.

## Historical API code

The generic budget modules preserve useful reservation-before-send, locking,
no-retry, bounded-output and fail-closed tests. Prior ledger hashes are explicit
caller inputs and tests construct fictional ledgers. Rate/model constants are
historical fixtures, not current pricing or spending permission.

The historical teacher collection, live-browser run and continual `real`
entrypoints now reject execution before credential access. Cached-label training
and offline simulations remain available. No launcher automatically reads a
personal credential path, releases an old reservation, or grants a new budget.
