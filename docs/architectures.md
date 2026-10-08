# Architecture inventory

Counts below come from module parameters, including unused heads and frozen RND
targets where stated. All local networks start with random weights. None is an
autoregressive language model.

| Family | Architecture and stored parameter count | Representation and context |
|---|---|---|
| Original / browser controller | Actor: 2304→128 ReLU→128 ReLU→32, plus legacy 128→128→32 mask head; **336,320** total, **315,680** excluding the unused mask branch. Independent 2304→128→128→1 critic: **311,681**. | Current DOM: 2,048-bin fixed SHA1 unigram/bigram counts; subgoal: 256-bin fixed hashes; L2 normalized. Lowercase whitespace tokenization; no learned token embeddings. Up to 32 action slots, masked to available actions; seven in the local cue page. No observation-sequence context. |
| RND | Each network 2048→256 ReLU→256: **590,336**. | Predictor learns feature prediction with MSE; random target stays frozen. RND is not a semantic embedding model. |
| Small cue MLP | Current actor **13,252**, critic **8,897**; history actor **25,540**, critic **21,185**. Each actor includes a **4,290**-parameter unused mask branch. | 64-bin fixed hashes per screen plus eight subgoal features; one or four screens. Two ReLU layers of width 64, two actions. |
| Tiny cue transformer | Separate actor **19,378** / critic **19,337**. Each: 64→32 projection, four learned positional vectors, two causal transformer layers, two attention heads, feedforward width 64, no dropout; final token plus eight subgoal features feeds output head. | Four hashed **screens**, not four words. Actor outputs two actions; critic one scalar. No vocabulary or next-token objective. |
| Numeric teacher comparison | Actor **9,992** including **4,420** unused mask parameters; critic **5,377**. Separate 17→64 ReLU→64 ReLU trunks. | Thirteen visible normalized item fields plus four zero subgoal features; four actions; current case only. |
| Easy/harder curriculum | **4,738**: 6→64 ReLU→64 ReLU→2. | Five numeric/task-family features and one zero subgoal feature. No tokenizer, learned embedding or sequence model. |
| Representation transfer | **13,394**: 30×16 learned embeddings, GRU(input 17, hidden 48), masked mean pooling, 48→64 ReLU→2. | Fixed regex tokenizer and 30-symbol vocabulary; numbers use a shared numeric token plus a value/2 channel. Maximum 22 tokens of constrained motion records, English templates or program text. No text generation. |
| Continual/replay answer learner | **1,349**: 3→32 tanh→32 tanh, four-action head and scalar value head. | Position/2, velocity/2 and horizon−1.5. No tokenizer, embedding table or within-case history. |
| Continual ASK controller | **2,083**: 61→32 tanh, two-action and scalar-value heads. | Eight current cases, their answer probabilities, wallet/query allowance, past accuracy/income and progress. ASK/SKIP; no future board. |
| Board allocator | **1,701**: 47→32 tanh, four-action and scalar-value heads. | Public current contract, padded past contract list, past per-skill statistics and up to two future public cards. Selects practice A/B with/without labels. |

Original actor/critic updates use clipped PPO, GAE, value MSE and entropy bonus,
with Adam. Default rollout length is 2,048, four epochs, minibatches of 256.
Smaller diagnostics explicitly lower the rollout. RND predictor updates occur
during training even in diagnostics where intrinsic reward weight is zero.
Its target and fixed hash encoders never change. The unused mask-head parameters
remain stored for checkpoint compatibility but receive no gradient.

The numeric teacher comparison adds 128 behavior-cloning updates on 128
demonstrations before PPO. Easy/harder curricula instead use label
cross-entropy with replay and engineered help/curriculum rules; those rules have
no trainable policy. Representation transfer trains embedding, GRU and classifier
with cross-entropy, not RL; its main comparison stopped at failed calibration.

Continual answer learning makes four PPO/value/entropy updates per eight-case
packet, adding label cross-entropy when labels are purchased. The replay repair
adds binary cross-entropy on the probability of past selected actions and their
observed correctness. The learned ASK head uses PPO every sixteen packets;
heuristic and no-tutor controls do not update ASK weights.

The board uses the same answer learner and balanced feedback replay, with 2,048
student optimizer steps per stream. Its allocator makes 64 optimizer steps per
training stream and zero during assessment. Answer learning continues throughout
assessment; held-out probes mutate neither model nor optimizer. Student weights
reset only between complete streams, so this is not cross-stream skill retention.

The real tutor used an external pretrained API model. Only local learner weights
were optimized; the tutor's weights were never trained here. Its tokenizer,
internal layers and parameter count are not established by the repository.
Historical paid runners are retired in this public reproduction.

Source: [PPO/RND](../rl/rnd_ppo_agent.py),
[policy](../rl/policy.py), [cue controls](../scripts/run_local_learning.py),
[representation](../scripts/run_representation_transfer.py),
[continual learner](../scripts/run_continual_tutor.py),
[board](../scripts/run_task_board_study.py).
