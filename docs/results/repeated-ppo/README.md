# Repeated PPO: measured CPU results

Five independent initializations; 20 updates each; 16 creatures; 128 steps per episode.
Training seeds: 11, 22, 33, 44, 55. Heldout worlds: 101, 202, 303, 404, 505.
Runtime: Python 3.12.14, PyTorch 2.14.0+cpu, Linux, two Torch threads.

| Training seed | Untrained PPO | Trained PPO | Difference |
| --- | ---: | ---: | ---: |
| 11 | 19.017 | 24.800 | +5.783 |
| 22 | 20.273 | 24.269 | +3.996 |
| 33 | 21.315 | 23.092 | +1.778 |
| 44 | 20.776 | 21.713 | +0.937 |
| 55 | 20.622 | 22.916 | +2.294 |

| Policy | Mean heldout fitness |
| --- | ---: |
| random | 18.872 |
| heuristic | 89.937 |
| strategy_no_memory | 94.532 |
| memory_strategy_heuristic | 90.952 |
| ppo_untrained | 20.400 |
| ppo_trained | 23.358 |

PPO improved in 5/5 independent training runs: mean paired gain **+2.958**.
Descriptive 95% bootstrap interval across training-run means: **[+1.545, +4.625]**,
conditional on these fixed heldout worlds. Trained PPO is still **66.579 points below**
the plain heuristic. This is only 20 updates per seed; no hyperparameter tuning was
performed against these heldout scores.

The memory-enabled rule strategy scored **3.580 points below** the same strategy
with memory disabled (world-paired bootstrap interval **[-6.877, -0.518]**).
Memory did not help this benchmark; persistence working is not evidence of benefit.
The five repeated copies of non-training baselines are not five independent samples.

`report.json` contains all raw per-world fitness values and summary methodology.
Each seed directory includes 20 training log rows, frozen evaluation and its final
checkpoint. All five checkpoints are retained without cherry-picking. Checkpoints
load through the existing `PPOTrainer.load` with `weights_only=True`.

For example, run the seed-11 model (chosen by seed order, not heldout ranking):

```bash
python -m experiments.strategy_runner --render --ppo-checkpoint docs/results/repeated-ppo/seed-11/checkpoint.pt
```

The viewer enables ongoing reproduction and persistent memory, unlike this frozen
benchmark, so its displayed fitness is not directly comparable to the table.

Live Ollama/Windows validation is still pending. Use `VALIDATE_OLLAMA.bat` on the PC.
See [protocol and reproduction instructions](../../REPEATED_PPO_OLLAMA.md).
