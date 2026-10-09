# Curious student

## Aim

A deployed model that keeps improving from its own experience. It decides what to
learn from, when to practice, and when to pay for help. No human curates its
training data or schedule. Over time it should pick up new skills and new task
types, and keep the ones it already has.

This is not how language models work today. Deployed models are frozen, and they
improve between releases through offline training that people curate. Memory
features add notes to the context window but don't change the weights. Learning
from deployment is still mostly research, for example:

- test-time training
- models that write their own fine-tuning data
- agents that grow skill libraries in code or memory rather than in weights

Four problems stand in the way:

1. **No reliable reward signal.** Production has no hidden tests that say an
   answer was right.
2. **Forgetting.** Training on recent traffic erodes older skills.
3. **Poisoning.** Users can teach the model bad behaviour.
4. **Cost and regressions.** Every update has to be checked.

## The sandbox

`world/`, `student/` and `llm/tutor_ledger.py` model that setting at small
scale:

| Production concept | Sandbox |
|---|---|
| Incoming work | Paid jobs on the board in `world/board.py` |
| Verifiable signals (tests, task completion) | Graders for chat, code (hidden tests in a subprocess) and buttons |
| Expensive fallback (frontier model or human) | Tutor (`gpt-6-luna`), paid in credits and capped in USD by the ledger |
| Choosing what to learn | Learning progress per skill (`student/curiosity.py`) |
| Not forgetting | Cross-skill replay in the hybrid learner |
| Help budget | Wallet credits plus the tutor ledger |

The student is Qwen2.5-0.5B-Instruct with a LoRA adapter. It learns in one of three ways:

- `sft`: fine-tuning on verified answers
- `grpo`: RL on grader rewards over groups of 4 samples, with a KL penalty to the base model
- `hybrid`: picks a mode per practice group
  - mixed scores: a GRPO update
  - all four fail: the student is stuck and may buy tutor help
  - all four pass: no GRPO update
  - always: replay weighted by how weak each skill still is

Tutor help has two modes:

- `retry` (default): the tutor gives feedback without the answer. The student
  retries, and only its own passing retries are used for training.
- `imitate`: graded tutor answers become training targets directly.

## What is already known

- **Fine-tuning on your own verified successes** (STaR, rejection-sampling
  fine-tuning, ReST) is a strong baseline.
- **GRPO** (DeepSeekMath, DeepSeek-R1) mostly sharpens abilities the model
  already has.
  - For small models, distillation from a stronger model beats RL.
  - Groups whose answers all score the same give no learning signal.
- **Mixing imitation of a stronger model with RL** has several recent
  versions (for example LUFFY), as does learning from feedback through
  retries (Reflexion, learning from natural-language feedback).
- **Choosing practice by learning progress** comes from intrinsic-motivation
  and automatic-curriculum work: Oudeyer and Kaplan 2007, Graves 2017,
  Matiisen 2017, Portelas 2019.
- **Deciding when to ask a teacher on a budget** comes from action advising
  ("teaching on a budget", Torrey and Taylor 2013) and learning when to ask
  for help (Nguyen and Daumé 2019).

What could be new here is the combination: one learner that serves a stream of
work, chooses what to learn from that work, asks for help only when it is stuck,
and does not forget.

## Results so far

Held-out greedy evaluation: 240 tasks, 60 per skill, about ±6 points per skill.
Arm: `progress`. Budget: 200 ticks per run.

The table shows the mean change from before to after training, averaged over
the seeds where every learner reached close to 200 ticks:

| Learner | Buttons | Arithmetic | Instructions | Code | All |
|---|---|---|---|---|---|
| sft | +0.31 | −0.03 | +0.22 | +0.11 | +0.15 |
| grpo | +0.29 | −0.02 | +0.10 | −0.12 | +0.07 |
| hybrid (imitate) | +0.40 | +0.02 | +0.18 | −0.05 | +0.14 |

What these numbers say:

- SFT is the strongest and cheapest learner, as the literature predicts.
- Hybrid's stuck-time tutor help works. It gives the best buttons gains, and
  every tutor answer passed the grader.
- The GRPO term costs code accuracy.
- Arithmetic and instruction groups were all-pass or all-fail 80 to 100% of
  the time, so GRPO gets little signal on those skills.
- Mac and GPU agree on seed 0. SFT went 0.48 → 0.64 on the Mac and
  0.48 → 0.65 on an A10.

Full tables are posted as comments on the pull request.

## Next experiment: a deployment stream

The question: can a model keep acquiring skills from its own experience,
choosing what to learn and when to ask, without forgetting and without curated
training?

1. **A stream it does not control.** Jobs arrive regardless of readiness.
   Skills appear, fade and return, and new task types are introduced partway
   through. The student must keep doing paid work while it learns.
2. **Self-curated learning.** The student chooses which past attempts to train
   on, when to practice variants of its failures, and when to buy help.
3. **A promotion gate.** An adapter update is kept only if it does not regress
   a held-out check. This is the production-safety piece.
4. **Metrics:**
   - performance on the live stream over time
   - time to acquire a new skill
   - retention when an old skill returns
   - tutor cost
5. **Baselines:**
   - a frozen model
   - training on everything it sees
   - a fixed schedule
   - the curious student

Known limits:

- Graders give a cleaner signal than real production has. A later step should
  use weaker signals: accept or reject decisions, or the model judging itself.
- At 0.5B the experiment shows whether the mechanism works, not that the
  student beats a large frozen model.

## Operating rules

Tutor calls:

- Every call reserves its worst-case cost before it is sent.
- The ledger ceiling is $20. Raising it is an explicit `raise-ceiling` step
  that records a reason.
- Each remote machine gets its own capped ledger. Its allowance is recorded in
  the local ledger first.
- The key reaches remote jobs only through `errand --env-file`, and the runner
  removes it from its environment after reading.

GPU leases:

- Keep a lease up only if the next job will be submitted within about 15 minutes.
- Otherwise fetch every job and release the lease explicitly with
  `errand leases release`.
- Fetch or collect every finished job: retained changes nobody has fetched
  keep a lease alive for up to 12 hours.

```sh
.venv/bin/python scripts/run_curious_student.py --arm progress --learner hybrid --tutor-mode retry --ticks 200 --eval-per-cell 12
```
