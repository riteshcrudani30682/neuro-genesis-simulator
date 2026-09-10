"""Immutable local sensors. No absolute position or full-world state is exposed."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    energy: float
    age: float
    # Row-major (dy, dx, food, hazard, other_creature, boundary) within square radius.
    cells: tuple

    def vector(self):
        return (self.energy, self.age) + tuple(v for cell in self.cells for v in cell[2:])


def observe(creature, config, food, hazards, occupied):
    x, y = creature.position
    cells = []
    r = config.sense_radius
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            point = (x + dx, y + dy)
            boundary = not (0 <= point[0] < config.width and 0 <= point[1] < config.height)
            cells.append((dx, dy, int(point in food), int(point in hazards),
                          int(point in occupied and point != creature.position), int(boundary)))
    return Observation(creature.energy / config.max_energy, creature.age / config.max_age, tuple(cells))
