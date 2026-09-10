"""Explicit scalar genotype; bounds also define diversity normalization."""
from dataclasses import asdict, dataclass
import math
from typing import ClassVar


@dataclass(frozen=True)
class Genome:
    movement_tendency: float = 0.8
    exploration_tendency: float = 0.15
    food_attraction: float = 1.0
    hazard_avoidance: float = 1.5
    reproduction_threshold: float = 65.0
    mutation_scale: float = 0.05

    BOUNDS: ClassVar[dict] = {
        'movement_tendency': (0.0, 1.0),
        'exploration_tendency': (0.0, 1.0),
        'food_attraction': (0.0, 3.0),
        'hazard_avoidance': (0.0, 3.0),
        'reproduction_threshold': (30.0, 95.0),
        'mutation_scale': (0.001, 0.2),
    }

    def __post_init__(self):
        for name, (low, high) in self.BOUNDS.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f'{name} must be numeric')
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'{name} must be within [{low}, {high}]')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if set(data) != set(cls.BOUNDS):
            raise ValueError('Genome must contain exactly the documented fields')
        return cls(**data)

    @classmethod
    def random(cls, rng):
        return cls(**{name: rng.uniform(low, high) for name, (low, high) in cls.BOUNDS.items()})
