"""Mutate only declared scalar genes using the caller's RNG."""
import math
from .genome import Genome


def mutate(parent, rng, rate=0.2):
    if not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError('mutation rate must be within [0, 1]')
    values = parent.to_dict()
    changes = []
    for name, (low, high) in Genome.BOUNDS.items():
        if rng.random() < rate:
            before = values[name]
            after = min(high, max(low, before + rng.gauss(0, parent.mutation_scale * (high - low))))
            values[name] = after
            if before != after:
                changes.append({'gene': name, 'before': before, 'after': after})
    return Genome(**values), changes
