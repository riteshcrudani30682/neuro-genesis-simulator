# Repeated PPO training and live Ollama validation

This extends the existing V2 core. NBB is unchanged. The measured Linux CPU
benchmark is complete. The user has supplied a passing Windows/Ollama report; see
[actual PC results](results/ollama-windows/README.md).

## Windows: validate an installed Ollama model

1. Update this repository (`git pull` in its folder).
2. Open the Ollama application.
3. Double-click `VALIDATE_OLLAMA.bat` in the repository folder.
4. Enter the number of an installed chat model from the printed list.
5. Read the final status and open the `report.json` path printed in the console.
   Share that report for interpretation. It is not uploaded automatically.

The launcher tries `.venv`, `venv`, `py -3`, then `python`. Validation needs only
Python's standard library, with no PyTorch/Pygame installation. It never downloads
models, changes model files, or modifies Ollama configuration. The Windows batch
launcher itself has not been executed on Windows in this development environment.

Equivalent command (replace `YOUR_INSTALLED_MODEL` with an exact listed name):

```bash
python -m experiments.ollama_validation --model YOUR_INSTALLED_MODEL
```

Default: three requests, 60 simulated steps, at most 60 seconds per chat request.
The first request can include model loading. The validator uses one creature in a
benign seeded world, no hazards/reproduction, fresh memory, and lower movement costs
to keep it alive for the diagnostic. It requests a goal at ticks 0, 20 and 40;
low-level movement still happens every step. It saves memory and verifies an exact
reload. This checks integration, not strategic intelligence or speed under population load.
Its fixed 20-tick diagnostic gate is deliberately more frequent than the normal
demo's 80-tick maximum-refresh gate.

Reports include:

- Python/OS, Ollama version, selected model name/digest/size.
- Each accepted goal or explicit fallback, elapsed request time and known token counts.
- Actual request count, cooldown, world transitions and memory round-trip checks.
- Loaded model size/VRAM/context fields when Ollama exposes them.
- Whether requests fit the normal demo's default 10-second timeout.

Exit codes: `0=passed`, `1=failed`, `2=blocked`.
**A fallback is never counted as successful LLM validation.**
Unavailable Ollama, missing models or no selection produce `blocked`. Invalid JSON,
confidence below 0.4, server errors or timeouts produce `failed`, even if the offline
controller completes the simulation. Token counts remain accounted for when a
response contains usage but fails the goal schema. Unknown usage is omitted per
event; totals include known values only.

If blocked, open Ollama and rerun; check the exact installed model name. If timed
out, use an installed smaller/faster chat model or warm the chosen model first.
If JSON or confidence fails, inspect `events` in the report. All network calls are
synchronous: slow models can pause the simulation. Discovery calls time out after
5 seconds; chat timeouts can be adjusted with `--timeout` up to 60 seconds.

After validation, launch the visible simulation with the same installed model:

```bash
python -m experiments.strategy_runner --render --planner ollama --model YOUR_INSTALLED_MODEL --llm-timeout 60 --max-calls 10
```

GPU diagnostics use Ollama's own loaded-model report, not a CUDA assumption. A pass
does not require GPU execution. The API implementation follows Ollama's official
[model listing](https://docs.ollama.com/api/tags),
[chat](https://docs.ollama.com/api/chat), and
[running-model diagnostics](https://docs.ollama.com/api/ps).

## Reproduce the benchmark

Install `requirements-ppo.txt` in the project's Python environment if needed, then:

```bash
python -m experiments.repeated_ppo --training-seeds 11 22 33 44 55 --heldout-seeds 101 202 303 404 505 --updates 20 --population 16 --steps 128 --output runs/repeated-ppo
```

Each training seed initializes a new model, optimizer and RNG, trains for 20 updates
on new worlds, and saves its checkpoint/log before frozen evaluation. All six
policies use the same heldout seeds, world settings and evaluation policy sampling
seeds. The untrained PPO comparison uses that run's original initialization.
Evaluation memories start empty; strategy is offline rules; reproduction is disabled.
All model seeds are retained, without selecting a winner on heldout scores.

Training world seed: `-(training_seed * 1000003 + update_index + 1)`, starting at
update index zero. Sign alone does **not** isolate Python RNG worlds; the runner
rejects overlapping absolute seed ranges, duplicate seeds and invalid counts.

Use `--resume` with exactly the same configuration to reuse completed training seeds.
An interrupted, incomplete seed restarts from its initialization and replaces only
that seed's partial log. This is seed-level restart, not mid-update continuation.
The existing single-run `experiments.ppo_runner --resume` remains available for
checkpoint-level continuation. New runs require an empty output directory.

`report.json` includes every training-seed × world-seed score, aggregate mean/median,
training-seed spread and paired deltas. PPO intervals bootstrap the five independent
training-run means (4,000 resamples, fixed bootstrap seed 7301). They condition on
these five heldout worlds and are descriptive with such a small sample.
Memory-vs-no-memory intervals resample the five world pairs once: duplicate baseline
runs are **not** counted as additional independent evidence. Creatures in the same
world are also not treated as independent experiments.

The [committed results](results/repeated-ppo/README.md) report improvement against
untrained PPO, a large remaining gap to heuristic control, and worse results with
this memory weighting. These are a baseline study, not evidence of general intelligence.

## Validation evidence

Automated tests use an explicitly fake local HTTP server to exercise the actual
HTTP adapter: discovery, requests, JSON parsing, failures, token accounting,
cooldowns and memory persistence. These fixtures are not an Ollama model.
The development workspace's real localhost probe returned `blocked` because no
Ollama service was available. No direct Windows PC connector was available either.
A subsequently supplied Windows report passes all checks with `ornith-9b-32k:latest`.
Its three requests took 50.47, 1.63 and 1.65 seconds. This closes the reported live
protocol/integration check; it is not an independent PC execution by Codex or a
strategy-quality benchmark. See [the report and limits](results/ollama-windows/README.md).
