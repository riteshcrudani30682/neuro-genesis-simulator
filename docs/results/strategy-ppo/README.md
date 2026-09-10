# Measured memory/strategy/PPO milestone

Environment: CPU, Python 3.12, PyTorch 2.14 CPU. Training: seed 42, 16 founders,
128-tick episode horizon, 20 on-policy updates, deterministic rule strategy.
See `training-seed42.jsonl`, `heldout-seed42.json`, and `demo-checkpoint.pt`.
The demo checkpoint is for reproducing this smoke result, not a recommended policy.

Mean evolutionary fitness on heldout seeds 101, 202, 303:

| Controller | Fitness |
|---|---:|
| Random | 19.418 |
| Plain heuristic | 90.462 |
| Strategy heuristic, memory disabled | 92.957 |
| Strategy heuristic, memory enabled | 92.257 |
| Untrained PPO | 21.596 |
| PPO after 20 updates | 23.079 |

PPO changed its weights and modestly improved in this run; it remains far behind
the heuristic. Memory did not improve the matched strategy-heuristic baseline in
this test. No statistical significance or LLM performance is claimed. The only
LLM tests use a simulated API transport; the real viewer screenshot uses rules.

The optional viewer was executed with SDL dummy drivers and its actual renderer
captured. Windows-native interaction and live local-Ollama inference were not
available in this environment.
