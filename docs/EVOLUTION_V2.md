# Evolution + Multi-Agent V2

## Starting point and compatibility

Implemented on `642e933`, which contained `core.contracts`, replay, and the generic
runner, but no `worlds/creature` implementation. The missing Creature environment
was added alongside the legacy code, using those existing contracts. No legacy
saves, simulator files, tests, or NBB/market code were changed or removed.
`CreatureEnvironment` adapts the same world to single-agent `reset(seed)` and
`step(action) -> StepResult`; `MultiCreatureEnvironment` takes a dictionary of
agent IDs to actions. This is a grid artificial-life experiment, not a biological
brain model or evidence of emergent intelligence.

## Run on Windows or Linux

From the repository root, the random/heuristic population runner needs only the
Python standard library. It does not import Pygame, NumPy, or Torch.

```bash
python -m experiments.population_runner --generations 20 --population 50 --seed 42 --compare-baselines
python -m experiments.population_runner --mode continuous --population 200 --max-population 200 --steps 120 --evaluation-seeds 11 --snapshot runs/continuous/world.svg --output runs/continuous
python -m experiments.population_runner --resume runs/population/checkpoint.json --generations 5 --output runs/resumed --compare-baselines
python -m pytest -q
```

`--generations` means additional generations on resume. Checkpoint configuration
wins over CLI population/seed/policy settings when resuming. Use a new output
folder for independent runs: JSONL files append, while checkpoints and snapshots
replace atomically (snapshots replace normally). Avoid appending a replay of an
old checkpoint into an already-completed run's output directory.

## Coherent world transition

1. Capture all pre-step observations. Collect every living agent's action.
2. Validate the complete action mapping before changing the world or RNG.
3. Propose destinations. Conflicting movers get a seeded lottery over sorted IDs.
4. A stationary occupant keeps its cell; blocked chains propagate to a fixed
   point. Swaps and closed movement cycles are allowed. There is no overlap.
5. Apply positions simultaneously. Each creature ages exactly once and pays a
   basal cost plus attempted-movement cost, including failed movements.
6. Consume food, apply hazards, cap energy, and mark starvation/age deaths.
   Food can rescue an agent within this tick before the final death check.
7. Reproduce from the eligible survivors in a seeded shuffled order.
8. Regrow food once, then return immutable next observations and true transitions.

Loop/input dictionary order cannot decide who wins. IDs break presentation and
elite-ranking ties; equal-fitness tournament ties are randomized. Newborns enter
next observations but never act in their birth tick. Dead agents have a terminal
next observation in their transition; they are absent from the next active map.
World horizon is a truncation, separate from individual death. The world stops
when no agents remain or its configured horizon is reached.

## Local observation schema and actions

`Action` is an IntEnum: `UP=0, DOWN=1, LEFT=2, RIGHT=3, STAY=4`.
A body is currently one occupied grid cell.

`Observation` contains normalized own energy and age and an immutable local square
sensor window (default radius 2). Each row-major sensor tuple is
`(dx, dy, food, hazard, other_creature, boundary)`. The four channels are binary.
Own occupancy is excluded. The agent receives no world coordinates, distant
resources, other genomes, population fitness, or lineage data. `.vector()` emits
energy, age, then the four channels in dy/dx order: **102 values at radius 2**.

## Genomes, life histories and reproduction

Every creature has an ID, genome, policy identity, position, age, energy,
biological generation, parent ID, birth/death tick, mutation summary and lifetime
counters. Policy instances live in the episode runner, with independent seeded
RNGs; environment logic never invokes Torch or a policy.

| Gene | Bounds | Role |
|---|---|---|
| movement_tendency | 0–1 | Heuristic propensity to move |
| exploration_tendency | 0–1 | Stochastic exploration; also neural adapter epsilon |
| food_attraction | 0–3 | Local heuristic food scoring |
| hazard_avoidance | 0–3 | Local heuristic hazard scoring |
| reproduction_threshold | 30–95 | Continuous-world birth eligibility |
| mutation_scale | 0.001–0.2 | Gaussian mutation standard deviation as range fraction |

Defaults: min birth energy 60, parent age at least 15, cooldown 20 ticks, parent
cost 30, child energy 25, mutation probability 0.2 per gene, cap 200 living agents.
Effective birth threshold is max(config minimum, genome threshold). A vacant
orthogonal non-food/non-hazard neighbor is required. Failure costs no birth
energy. Every birth records child ID, parent ID, generation=parent+1, birth tick
and changed genes with before/after values. The immutable parent's genome stays
unchanged. Mutation clips explicit finite scalar values to documented bounds.
Population cap, finite resources, energy expenditure and max age limit growth.

## Two distinct modes

**Continuous:** all creatures share a world and may reproduce during their lives.
Metrics include dead creatures and children. Biological generations may coexist.
There is no tournament reset in this mode. Configurable `max_steps` bounds a run.

**Generational:** start an episode from each candidate genome, evaluate all
founders in the same competitive world across distinct selection seeds, aggregate
fitness by mean/median, tournament-select parents, mutate, and start the next
population. Two elites survive as unchanged genomes by default, with new IDs and
parent links. In-world reproduction is disabled in this mode, so every candidate
gets comparable episode exposure. Reproduction-threshold fitness is neutral in
this mode; the gene remains useful in continuous life. Reproduction fitness and
`offspring_count` are zero during these episodes; breeding across generations is
recorded in next-generation members' parent IDs, rather than fabricated as food
or survival reward. There is no assertion that all genes improve under all modes.

## Policies, replay and neural integration

- `RandomAgent`: genome-independent uniform random baseline.
- `HeuristicAgent`: local food/hazard scoring controlled by genome.
- `NeuralAgentAdapter`: accepts an existing PyTorch Q-network. Instantiate it for
  the 102-value observation (or configured radius) with five outputs. An explicit
  four-action map can reuse legacy QNetwork classes with four outputs. Arbitrary
  legacy weights trained on a different state schema are not compatible.

Adapters follow `act`/`observe` and record real `core.replay.Transition` entries.
The neural adapter supports `act_batch`; default population policies have
independent networks/RNGs, so the runner does not batch unrelated networks.
Use `run_population_episode(..., policy_factory=...)` to inject a shared-contract
agent. A supplied agent can implement learning in `observe`; this milestone's
neural adapter is frozen inference plus replay, not a claim of DQN training.
Evolution changes scalar genomes, not arbitrary Python or neural objects.

`set_goal` defines validated future goal metadata (`seek_food`,
`avoid_competition`, `explore`, `protect_energy`). There are no LLM calls,
strategy execution, persistent memory or PPO in this milestone.

## Fitness and metrics

Immediate rewards: survival +0.01, food +2, hazard -2, blocked movement -0.05,
death -1. These are logged separately and do not define evolutionary fitness.

Fitness sums these visible components:

- Survival: 0.02 × lifetime.
- Food: 3 × food eaten.
- Hazard avoidance: 0.05 × moves away from an adjacent hazard without hitting one.
- Hazard damage: -2 × hazards hit.
- Energy efficiency: food eaten / (1 + energy spent).
- Reproduction: 2 × offspring count.
- Blocked movement: -0.1 × blocked move attempts.

This is an engineered objective and can be exploited; the components are exposed
for inspection. `hazards_avoided` is an observable movement event, not an inference
about intention. Energy spent includes attempted movement and reproduction costs.

`creatures.jsonl` records one complete life record per agent per episode/seed,
including every requested metric, fitness components, genome, policy, position,
birth/death tick and mutation history. `generations.jsonl` records selection
summaries; `heldout_comparison.json` records matched baselines. Continuous mode
uses separate `continuous_*` files. No database is used.

Diversity is mean population variance over genes normalized to their bounds.
`genetic_diversity` is founder diversity; `final_genetic_diversity_mean` averages
surviving populations' diversity over seeds. Zero means clones, singleton, or
extinction (use survival/population metrics to distinguish). In continuous
summaries `population_size` counts founders; `final_population_mean` counts
survivors including descendants. `survival_rate` includes all lives;
`founder_survival_rate` avoids newborn exposure bias. `offspring_count` sums births
across the listed seeds. Founder mean fitness and all-lives per-seed mean fitness
can differ in continuous mode.

## Research evaluation and safe checkpoints

Selection defaults to seeds 11/22/33. Optional heldout comparison uses disjoint
101/202/303, with the same initial positions/resources and policy RNG slots for
evolved, frozen initial heuristic and random populations. Heldout evaluations do
not influence breeding. Report seed-wise scores and their standard deviation;
repeat the entire evolution with different `--seed` values before drawing broad
conclusions. Larger mean scores in one run do not establish general intelligence.

`checkpoint.json` is versioned atomic JSON at **generation boundaries**. It stores
the next population, initial baseline population, generation, next ID, config,
aggregate metrics and the full selection/mutation RNG state. Loading and running
the next generation matches uninterrupted execution exactly on the tested Python
runtime. It does not serialize a mid-tick continuous world or external neural
weights/policy optimizer state. Legacy pickle saves are untouched.

Optional SVG rendering is passive; it displays food, hazards, IDs, generation
colors and a selected/best creature. With repeated continuous seeds the snapshot
shows the last evaluated seed. No GUI or Pygame is required for snapshots.
