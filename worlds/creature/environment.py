"""Simultaneous shared-grid environment. No renderer, policy, or Torch dependency."""
from dataclasses import dataclass, replace
import random

from core.contracts import StepResult
from core.replay import Transition
from evolution.genome import Genome
from evolution.mutation import mutate
from evolution.fitness import fitness, fitness_components
from .entities import Action, Creature, DELTAS, WorldConfig
from .observations import observe
from .rewards import reward_components


@dataclass(frozen=True)
class PopulationStep:
    observations: dict
    rewards: dict
    terminated: dict
    truncated: bool
    births: tuple
    deaths: tuple
    transitions: dict
    reward_components: dict

    @property
    def done(self):
        return self.truncated or not self.observations


class MultiCreatureEnvironment:
    def __init__(self, config=None):
        self.config = config or WorldConfig()
        self.creatures = {}
        self.food = set()
        self.hazards = set()
        self.rng = random.Random()
        self.step_count = 0
        self.next_id = 0
        self.finished = True

    @property
    def alive(self):
        return {i: c for i, c in sorted(self.creatures.items()) if c.alive}

    def reset(self, seed=None, *, genomes=None, founders=None, policy_name='heuristic'):
        self.rng = random.Random(seed)
        self.step_count = 0
        self.finished = False
        self.creatures = {}
        config = self.config
        genomes = list(genomes) if genomes is not None else [Genome() for _ in range(config.population)]
        if len(genomes) != config.population or not all(isinstance(g, Genome) for g in genomes):
            raise ValueError('one valid genome is required per initial creature')
        if founders is not None and (len(founders) != len(genomes) or
                                    len({f['id'] for f in founders}) != len(founders)):
            raise ValueError('founder identities must be unique and aligned with genomes')
        positions = self.rng.sample([(x, y) for y in range(config.height) for x in range(config.width)],
                                    config.population + config.hazard_count + config.food_count)
        for index, genome in enumerate(genomes):
            metadata = dict(founders[index]) if founders is not None else {'id': index}
            if type(metadata['id']) is not int or metadata['id'] < 0:
                raise ValueError('founder ID must be a nonnegative integer')
            c = Creature(position=positions[index], energy=config.initial_energy, genome=genome,
                         policy_name=policy_name, **metadata)
            self.creatures[c.id] = c
        self.next_id = max(self.creatures) + 1
        start = config.population
        self.hazards = set(positions[start:start + config.hazard_count])
        self.food = set(positions[start + config.hazard_count:])
        return self.observations()

    def observations(self):
        occupied = {c.position for c in self.alive.values()}
        return {i: observe(c, self.config, self.food, self.hazards, occupied)
                for i, c in self.alive.items()}

    def _resolve(self, actions, agents):
        """Seeded lottery for common targets; blockers propagate, swaps/cycles succeed."""
        destinations = {}
        contenders = {}
        occupied = {c.position: i for i, c in agents.items()}
        blocked = set()
        for i, c in agents.items():
            dx, dy = DELTAS[actions[i]]
            target = (c.position[0] + dx, c.position[1] + dy)
            if not (0 <= target[0] < self.config.width and 0 <= target[1] < self.config.height):
                target = c.position
                blocked.add(i)
            destinations[i] = target
            if target != c.position:
                contenders.setdefault(target, []).append(i)
        for target in sorted(contenders):
            ids = sorted(contenders[target])
            if len(ids) > 1:
                winner = self.rng.choice(ids)
                for i in ids:
                    if i != winner:
                        destinations[i] = agents[i].position
                        blocked.add(i)
        # A stationary occupant has right of occupancy. Block chains to fixed point.
        changed = True
        while changed:
            changed = False
            for i in agents:
                target = destinations[i]
                occupant = occupied.get(target)
                if target != agents[i].position and occupant is not None and destinations[occupant] == target:
                    destinations[i] = agents[i].position
                    blocked.add(i)
                    changed = True
        return destinations, blocked

    def _reproduce(self):
        if not self.config.reproduction_enabled:
            return []
        cfg = self.config
        eligible = [c for c in self.alive.values()
                    if c.energy >= max(cfg.min_reproduction_energy, c.genome.reproduction_threshold)
                    and c.age >= cfg.min_reproduction_age
                    and self.step_count - c.last_birth_step >= cfg.reproduction_cooldown]
        self.rng.shuffle(eligible)  # Seeded fairness, never iteration-order priority.
        occupied = {c.position for c in self.alive.values()}
        births = []
        for parent in eligible:
            if len(occupied) >= cfg.max_population:
                break
            x, y = parent.position
            spaces = [(x + dx, y + dy) for dx, dy in list(DELTAS.values())[:4]
                      if 0 <= x + dx < cfg.width and 0 <= y + dy < cfg.height
                      and (x + dx, y + dy) not in occupied | self.hazards | self.food]
            if not spaces:
                continue
            position = self.rng.choice(spaces)
            genome, summary = mutate(parent.genome, self.rng, cfg.mutation_rate)
            child = Creature(id=self.next_id, position=position, energy=cfg.child_energy,
                             genome=genome, policy_name=parent.policy_name,
                             generation=parent.generation + 1, parent_id=parent.id,
                             birth_step=self.step_count, mutation_history=summary)
            self.next_id += 1
            parent.energy -= cfg.reproduction_cost
            parent.energy_spent += cfg.reproduction_cost
            parent.last_birth_step = self.step_count
            parent.offspring_count += 1
            self.creatures[child.id] = child
            occupied.add(position)
            births.append(child.id)
        return births

    def step(self, actions):
        if self.finished:
            raise RuntimeError('reset required before stepping a completed world')
        agents = self.alive
        if set(actions) != set(agents):
            raise ValueError('provide exactly one action for every living creature')
        if any(type(a) is bool or not isinstance(a, (int, Action)) for a in actions.values()):
            raise ValueError('actions must be Action members or integer values')
        actions = {i: Action(actions[i]) for i in agents}  # Validate before mutating any state.
        previous = self.observations()
        destinations, blocked = self._resolve(actions, agents)
        self.step_count += 1
        components = {}
        deaths = []
        for i, c in agents.items():
            old_position = c.position
            c.position = destinations[i]
            moved = c.position != old_position
            c.age += 1
            c.movements += int(moved)
            c.blocked_moves += int(i in blocked)
            # Charge attempted movement, including failed boundary/collision moves.
            cost = self.config.basal_cost + self.config.movement_cost * (actions[i] != Action.STAY)
            c.energy_spent += cost
            c.energy -= cost
            ate = c.position in self.food
            hit = c.position in self.hazards
            if ate:
                self.food.remove(c.position)
                c.food_eaten += 1
                c.energy = min(self.config.max_energy, c.energy + self.config.food_energy)
            if hit:
                c.hazards_hit += 1
                c.energy -= self.config.hazard_damage
            # Count only actual movement away from an adjacent hazard, not repeated idle reward.
            if moved and not hit and any(abs(hx-old_position[0]) + abs(hy-old_position[1]) == 1
                                        for hx, hy in self.hazards):
                c.hazards_avoided += 1
            c.energy = max(0.0, c.energy)
            if c.energy <= 0 or c.age >= self.config.max_age:
                c.alive = False
                c.death_step = self.step_count
                deaths.append(i)
            components[i] = reward_components(food=ate, hazard=hit, blocked=i in blocked, died=not c.alive)
            c.total_reward += sum(components[i].values())
        births = self._reproduce()
        # Regrow resources once per world tick, after all consumption/births.
        occupied = {c.position for c in self.alive.values()}
        count = min(self.config.food_regrowth, self.config.food_count - len(self.food))
        if count > 0:
            unavailable = occupied | self.food | self.hazards
            free = [(x, y) for y in range(self.config.height) for x in range(self.config.width)
                    if (x, y) not in unavailable]
            self.food.update(self.rng.sample(free, min(count, len(free))))
        observations = self.observations()
        truncated = self.step_count >= self.config.max_steps
        transitions = {}
        for i, c in agents.items():
            next_state = observations.get(i) or observe(c, self.config, self.food, self.hazards, occupied)
            transitions[i] = Transition(previous[i], actions[i], sum(components[i].values()),
                                        next_state, not c.alive or truncated)
        for c in self.creatures.values():
            c.fitness = fitness(c)
        self.finished = truncated or not observations
        return PopulationStep(observations, {i: t.reward for i, t in transitions.items()},
                              {i: not c.alive for i, c in agents.items()}, truncated,
                              tuple(births), tuple(deaths), transitions, components)

    def creature_metrics(self):
        return [{**c.metrics(), 'fitness_components': fitness_components(c)}
                for _, c in sorted(self.creatures.items())]


class CreatureEnvironment:
    """Single-agent adapter preserving the shared Environment/StepResult contract."""
    def __init__(self, config=None):
        self.world = MultiCreatureEnvironment(replace(config or WorldConfig(), population=1,
                                                     max_population=1, reproduction_enabled=False))

    def reset(self, seed=None):
        return self.world.reset(seed=seed)[0]

    def step(self, action):
        result = self.world.step({0: action})
        t = result.transitions[0]
        return StepResult(t.next_state, t.reward, result.terminated[0], result.truncated,
                          {'reward_components': result.reward_components[0]})
