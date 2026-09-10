"""Explicit world configuration, agent identity, and life-history state."""
from dataclasses import dataclass, field, asdict
from enum import IntEnum
import math
from evolution.genome import Genome


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4


DELTAS = {Action.UP: (0, -1), Action.DOWN: (0, 1), Action.LEFT: (-1, 0),
          Action.RIGHT: (1, 0), Action.STAY: (0, 0)}


@dataclass(frozen=True)
class WorldConfig:
    width: int = 32
    height: int = 32
    population: int = 50
    max_population: int = 200
    food_count: int = 180
    hazard_count: int = 40
    food_regrowth: int = 3
    sense_radius: int = 2
    initial_energy: float = 50.0
    max_energy: float = 100.0
    basal_cost: float = 0.25
    movement_cost: float = 0.35
    food_energy: float = 12.0
    hazard_damage: float = 15.0
    max_age: int = 500
    max_steps: int = 300
    reproduction_enabled: bool = True
    min_reproduction_energy: float = 60.0
    reproduction_cost: float = 30.0
    child_energy: float = 25.0
    min_reproduction_age: int = 15
    reproduction_cooldown: int = 20
    mutation_rate: float = 0.2

    def __post_init__(self):
        integers = ('width', 'height', 'population', 'max_population', 'food_count',
                    'hazard_count', 'food_regrowth', 'sense_radius', 'max_age',
                    'max_steps', 'min_reproduction_age', 'reproduction_cooldown')
        for name in integers:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if min(self.width, self.height, self.population, self.max_age, self.max_steps) < 1:
            raise ValueError('world dimensions, population and horizons must be positive')
        if not self.population <= self.max_population <= self.width * self.height:
            raise ValueError('invalid population cap')
        if self.population + self.food_count + self.hazard_count > self.width * self.height:
            raise ValueError('initial population and resources do not fit')
        if not 1 <= self.sense_radius <= 5:
            raise ValueError('sense_radius must be between 1 and 5')
        for name in ('initial_energy', 'max_energy', 'basal_cost', 'movement_cost',
                     'food_energy', 'hazard_damage', 'min_reproduction_energy',
                     'reproduction_cost', 'child_energy', 'mutation_rate'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not 0 < self.initial_energy <= self.max_energy:
            raise ValueError('invalid initial energy')
        if not 0 < self.child_energy <= self.reproduction_cost < self.min_reproduction_energy <= self.max_energy:
            raise ValueError('reproduction must conserve energy and leave the parent alive')
        if self.mutation_rate > 1:
            raise ValueError('mutation rate exceeds 1')


@dataclass
class Creature:
    id: int
    position: tuple
    energy: float
    genome: Genome
    policy_name: str = 'heuristic'
    generation: int = 0
    parent_id: object = None
    birth_step: int = 0
    mutation_history: list = field(default_factory=list)
    age: int = 0
    alive: bool = True
    last_birth_step: int = -1_000_000
    food_eaten: int = 0
    hazards_hit: int = 0
    hazards_avoided: int = 0
    movements: int = 0
    blocked_moves: int = 0
    energy_spent: float = 0.0
    total_reward: float = 0.0
    offspring_count: int = 0
    fitness: float = 0.0
    death_step: object = None

    def metrics(self):
        return {**asdict(self), 'lifetime': self.age, 'final_energy': self.energy}
