# Neuro-Genesis V2 Foundation

V2 turns the project into one shared research platform with two tracks:

1. **Artificial Life Lab** — RL, evolution, multi-agent behavior, memory and an optional LLM strategy layer.
2. **NBB Market Lab** — historical market replay and safe shadow research for new learning/decision architectures.

## Non-breaking migration rule

The current Pygame simulator remains the working legacy/reference implementation while V2 is built beside it. Do not rewrite or remove the existing simulator until equivalent V2 behavior has regression coverage.

## Shared transition model

Every V2 world should follow one explicit loop:

`reset -> state -> agent.act -> environment.step -> reward + next_state -> agent.observe -> repeat`

The first shared primitives live in:

- `core/contracts.py` — Environment, Agent and StepResult contracts.
- `core/replay.py` — reusable Transition and ReplayBuffer.
- `experiments/runner.py` — generic headless episode runner.

## Target architecture

```text
core/
  contracts.py
  replay.py
  metrics.py          # next
  seeding.py          # next

worlds/
  creature/           # first migration target
  market/             # later NBB replay environment

brains/
  dqn.py
  ppo.py
  llm.py
  hybrid.py

evolution/
  genome.py
  mutation.py
  selection.py

experiments/
  runner.py
  configs/

ui/
  pygame_view.py
  dashboard.py
```

## Immediate V2 milestone

Migrate the creature simulation first. Correct the current RL transition semantics so one environment step applies one coherent action, then captures the actual resulting next state. Keep Pygame as a renderer/controller rather than embedding learning logic in rendering/event code.

Required properties:

- deterministic reset with seed
- explicit observation/state schema
- explicit action schema
- one action -> one environment transition
- reward components logged separately
- terminal/truncation semantics
- headless episodes
- replay buffer populated from real `(state, action, reward, next_state, done)` transitions
- tests proving state/action/reward alignment

## Artificial Life track

After the creature migration is correct, add multi-creature experiments, genomes, mutation, selection, reproduction, fitness metrics, persistent memories and a slow high-level LLM strategy layer. The LLM should set goals/strategy, not issue per-frame movement commands.

## NBB Market Lab track

The market track must remain isolated from live order execution. Historical/paper market data becomes an environment; actions may include `CE`, `PE`, `WAIT`, `EXIT`; reward must account for trade quality, drawdown, overtrading and correct waiting. Compare architectures using walk-forward/out-of-sample evaluation.

Live NBB deterministic risk controls remain authoritative. Neuro-Genesis V2 is research/shadow infrastructure until separately validated.

## Implemented: Evolution + Multi-Agent V2

The creature environment and population milestone are now implemented using the
shared contracts above. See [Evolution V2](EVOLUTION_V2.md) for actual behavior,
commands and remaining research limitations. NBB Market Lab remains untouched.
