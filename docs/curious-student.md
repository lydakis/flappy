# Curious student

## Aim

A deployed model that keeps improving from its own experience. It decides what to
learn from, when to practice, and when to pay for help. No human curates its
training data or schedule. Over time it should pick up new skills and new task
types, and keep the ones it already has.

The point is not to beat a larger model, and not only to cut training cost. The
question is whether a learner's decisions about its own education are worth
something.

There are two kinds of learning to keep apart:

- **Learning to perform:** updates that make the student solve more tasks.
- **Learning to learn:** better decisions about which experiences, practice
  and tutor interactions make those updates worthwhile.

Good SFT results are evidence for the first, not for the second.

### Why this is open

Deployed language models are frozen. They improve between releases through
offline training that people curate. Memory features add notes to the context
window but don't change the weights. Close research:

- **SEAL** (Zweiger et al., NeurIPS 2025): the model writes its own fine-tuning
  data and update directives. An RL outer loop rewards post-update performance.
  The paper reports forgetting under sequential self-edits.
- **Self-Distillation Fine-Tuning** (Shenfeld et al., ICML 2026): a
  demonstration-conditioned copy of the model teaches itself. It gains
  sequential skills with less forgetting than SFT. In their setup the smallest
  model lagged SFT because its in-context learning was too weak.
- **Test-time training**, and agents that grow skill libraries in code or memory
  rather than in weights (Voyager).

What this project adds to those is the combination:

- a persistent stream of work it doesn't choose
- choosing learning actions on its own
- a budget for tutor help
- the tradeoff between learning new skills and keeping old ones

Four problems stand in the way of learning from deployment:

1. **No reliable reward.** Production has no hidden tests that say an answer was
   right.
2. **Forgetting.** Training on recent traffic erodes older skills.
3. **Poisoning.** Users can teach the model bad behaviour.
4. **Cost and regressions.** Every update has to be checked.

The first experiments keep clean graders on purpose. That isolates the decision
problem. Weaker signals come later, as a separate question.

## The sandbox

| Production concept | Sandbox |
|---|---|
| Incoming work | Paid jobs in `world/board.py` |
| Verifiable signals | Graders for chat, code (hidden tests in a subprocess) and buttons |
| Expensive fallback (frontier model or human) | Tutor (`gpt-6-luna`), paid in credits and capped in USD by `llm/tutor_ledger.py` |
| Not forgetting | Cross-skill replay (hybrid learner) |

The student is Qwen2.5-0.5B-Instruct with a LoRA adapter.

**Learners** (`--learner`):

- `sft`: fine-tunes on verified answers.
- `grpo`: RL on grader rewards over groups of 4 samples, with a KL penalty to the
  base model.
- `hybrid`: decides per practice group.
  - Mixed scores: a GRPO update.
  - All four fail: the student is stuck and may ask for help.
  - All four pass: no GRPO update.
  - Every step adds replay weighted by how weak each skill still is.

**Tutor modes** (`--tutor-mode`):

- `retry` (default): the tutor gives feedback without the answer. The student
  retries, and only its own passing retries are trained on, stored against the
  bare prompt.
  - Feedback that itself passes the grader is discarded.
  - Evaluation always uses bare prompts, with no hints or feedback.
- `imitate`: graded tutor answers become training targets.
- `retry_blank`: a control for retry. Same retry decisions and attempt counts,
  but the prompt only says the answer was wrong, and no tutor call is made.

## How the student decides today

These are fixed rules with two small learned estimates. They are baselines that
a learned policy would have to beat, not evidence of learned self-direction.

**What it sees:**

- per-skill success over its last 16 attempts
- learning progress per skill (recent success minus older success)
- per (skill, difficulty) success counts
- the job board's skill, difficulty and pay
- its wallet
- per-skill estimates of how useful each kind of help has been

| Decision | Rule | Learned? |
|---|---|---|
| Which job to take | Highest optimistic expected pay | Success estimates only |
| What to practice | Sample skills by absolute learning progress, then the lowest difficulty below 70% success | Progress statistics only |
| Whether to ask | Only if progress has plateaued or all attempts failed, and a Thompson sample says help's expected gain × the value of 5 future jobs exceeds its price | Beta posterior of help usefulness per skill and help kind |
| What to train on | Verified answers: own successes, passing retries, and verified tutor answers in imitate mode | No |

**The question this doesn't capture yet:** which action now most improves
future work? Two failures can look the same, a one-off format quirk and a
missing concept that affects hundreds of later tasks, yet help is worth very
different amounts in each. Fast progress on easy work can also crowd out a hard
prerequisite that pays off later. Learning to invest in future competence is
the interesting part. A learned controller should come only after we know
whether the current choices matter at all.

## Results so far: pilot

Evaluation: 240 held-out tasks (60 per skill), greedy decoding, bare prompts.

At 60 tasks, ±6 points is one binomial standard error for a single run at
about 50% accuracy. It does not include variance between seeds.

What a tick is:

- one paid attempt, plus one practice attempt
- `grpo` and `hybrid` practice samples 4 answers, so they generate and train
  more per tick than `sft`
- 200 ticks are therefore not equal compute

Mean change, averaged over two seeds (the seeds where every learner reached
about 200 ticks):

| Learner | Buttons | Arithmetic | Instructions | Code | All |
|---|---|---|---|---|---|
| sft | +0.31 | −0.03 | +0.22 | +0.11 | +0.15 |
| grpo | +0.29 | −0.02 | +0.10 | −0.12 | +0.07 |
| hybrid (imitate) | +0.40 | +0.02 | +0.18 | −0.05 | +0.14 |

How to read it:

- The training loop improves held-out performance. SFT is a simple, sound
  default for the next experiment.
- These numbers don't establish a robust ranking. A one-point gap between SFT
  and hybrid is noise.
- Configurations that include GRPO showed code regressions. Blaming the GRPO
  term specifically would need a controlled ablation.
- In DeepSeek-R1's setup, distillation beat RL for small models. That is a
  result in their setting, not a general rule.
- Seed 0 agrees across hardware. SFT went 0.48 → 0.64 on the Mac and
  0.48 → 0.65 on an A10.

Full tables and the GPU runs are posted as comments on the pull request.

## Next experiment: a deployment stream

**First milestone:** under a shifting stream of verifiable work and fixed
resource limits, show that self-selected learning actions improve cumulative
unaided performance, and acquisition of new tasks, over strong fixed and
failure-triggered policies, while retention is measured independently.

**The stream:**

- Order A → B → A returns → C appears.
  - A and B test adaptation and retention.
  - C is a task family the student has never seen, with new conventions (for
    example a newly specified tool API or a small transformation language),
    evaluated on unseen compositions.
- Vary the ordering and arrival times across runs.
- Never reveal phase boundaries to the controller.
- Jobs arrive regardless of readiness, so the student can't select comfortable
  work.

**Main comparison:** the same student, SFT optimizer, replay capacity,
promotion gate, stream and resource limits, under two policies:

- curious policy vs strong simple policy: verified-success SFT, sensible
  replay, and fixed or failure-triggered rules for practice and asking
- also a frozen model, and the strong simple policy plus an "always ask"
  variant with defined behaviour once its budget runs out
- "train on everything" means every verified success, not failed answers

**Budgets and compute:**

- Equal budget ceilings, not forced equal spending. Spending less is itself an
  outcome.
- Account for student generation, training and validation compute as well as
  tutor dollars, so extra local compute can't pass as better tutor judgment.

**Outcomes:**

- **Primary:** cumulative unaided performance on arriving jobs, measured on
  each job before learning from it.
- **Secondary:**
  - how fast the student acquires C
  - retention when A returns
  - tutor dependence within comparable task groups
  - gate acceptance rate
- **Tutor target:** dependence should fall on familiar work, while the student
  still asks for help on unfamiliar work. Report assisted success and later
  unaided success separately.

**The promotion gate:**

- An adapter update is kept only if it passes per-skill tolerances against
  both the previous checkpoint and a fixed reference.
- Keep three sets separate:
  - learning feedback
  - promotion validation, which becomes adaptively reused
  - an independent audit set that never influences training, tutor use or
    promotion
- Report acceptance rate and old-skill performance before and after the gate,
  and ablate the gate. Otherwise "stable" might mean the gate filtered out bad
  updates, or that nothing was learned.
- Call this regression screening within the benchmark, not a production-safety
  solution.

**Do choices create value?** Occasionally fork the same checkpoint, apply two
different learning actions under equal budgets, and compare their effect on the
same later evaluation tasks. Keep these diagnostics out of the deployed
controller.

**Positive control:** before testing autonomy on a skill, check that the
student can learn it under generous, well-supervised training. If it can't, a
failed autonomous run says little about the policy.

## Operating rules

Tutor calls:

- Every call reserves its worst-case cost before it is sent.
- The ledger ceiling is $20. Raising it is an explicit `raise-ceiling` step
  with a recorded reason.
- Each remote run gets its own capped ledger. Its allowance is recorded in the
  local ledger first.
- The key reaches remote jobs only through `errand --env-file`, and the runner
  removes it from its environment after reading.

GPU leases:

- Keep a lease up only if the next job will be submitted within about 15
  minutes.
- Otherwise fetch every job and release the lease explicitly with
  `errand leases release`.
- Unfetched job changes keep a lease alive for up to 12 hours.

```sh
.venv/bin/python scripts/run_curious_student.py --arm progress --learner hybrid --tutor-mode retry --ticks 200 --eval-per-cell 12
```
