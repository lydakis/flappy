# Fixed offline task-board protocol

Compare learner-selected practice with and without a truthful two-contract
forecast, crossed with availability of perfect-oracle labels. The full settings
are in [configs/task-board-study.json](../configs/task-board-study.json).
Retention is an outcome of this protocol; the earlier replay study's failed gate
remains a separate result.

Use seeds 307, 401 and 503, four conditions, four training streams and two
assessment streams per controller: 72 streams total. Allocator weights carry
between its streams; answer model, replay memory and wallet restart each stream.
Freeze the allocator during assessment while the answer learner keeps updating.

Each stream has eight contracts of 32 cycles, four contracts per skill, with
each skill returning after at least one absence. Sample six distinct valid
schedules per seed, using four for training and two for assessment. A cycle has
one eight-case practice packet and one eight-case paid packet. Select practice
before seeing candidate cases. Commit answers before revealing feedback or
purchased labels, then update. Score paid answers before their feedback/update.

Actions are practice A, practice B, and each with labels. The initial purchase
probability is 5%, independent of the forecast. A correct paid A answer earns
0.25 synthetic credit; B earns one. Practice earns no wallet income. Start with
four credits; labels cost two per packet, with twelve queries maximum.
Unaffordable requests earn no labels and incur no charge. No-tutor arms mask
purchase actions entirely.

The allocator receives only current public contract information, past completed
contracts, observed performance history, wallet/query allowance and (in board
arms) the next two contracts' skill/pay/availability. It receives no candidate
inputs, targets, identifiers, future outcomes or recommended lesson. The answer
model sees only numeric position, velocity and horizon.

Match paid packets, practice candidate pools, initializations, opportunities,
seeds and optimizer-step counts across paired arms. Selected practice examples
may differ as a consequence of the action. Use separate latent-tuple hash
partitions for training paid work, training practice, assessment paid work,
assessment practice and probes. Sample target classes independently; audit
position-only and public-metadata-only label predictions against a 35% cap.

All conditions share a past-feedback reservoir holding inputs, selected actions
and correctness bits, grouped only by observed public horizon: 128 records each,
32 replay samples per update. No future cards or oracle labels enter replay.
The learner makes four optimizer steps per packet: 2,048 per stream. Its loss is
PPO/value/entropy plus correctness replay, with label cross-entropy only when a
query is granted. The allocator uses four PPO epochs every sixteen cycles.

Primary readiness is accuracy on the first four paid packets of a returning
skill. The preparation criterion requires at least a five-percentage-point
board gain in two of three seeds, nonnegative pooled net-income change, and
mean per-skill readiness at least 80%. Tutor value requires a 5% net-income
advantage in two seeds with the fixed macro-accuracy guard. Report first
appearances, maintenance losses, query counts and forecast sensitivity too.
Do not change thresholds if controls are already near ceiling. Analyze seed
pairs as the independent units, not individual packets.

No API, browser, credentials or pretrained model is required. Execution uses one
CPU thread with a 600-second preparation-plus-run wall-clock guard, peak process
RSS below 1,800 MiB, at least 8 GiB free disk and output below 100 MiB. Stop on
integrity failure, resource limit or the fixed endpoint. There is no automatic
paid extension.
