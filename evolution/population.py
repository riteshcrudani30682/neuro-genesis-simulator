"""Generational breeding is independent of in-world energy-based reproduction."""
from dataclasses import dataclass, field, asdict
from .genome import Genome
from .mutation import mutate
from .selection import tournament


@dataclass(frozen=True)
class Member:
    id: int
    genome: Genome
    generation: int = 0
    parent_id: object = None
    mutation_history: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data['genome'] = Genome.from_dict(data['genome'])
        return cls(**data)

    def founder(self):
        return {k: v for k, v in self.to_dict().items() if k != 'genome'}


def breed(members, scores, rng, *, next_id, elitism=2, mutation_rate=0.2, tournament_size=3):
    if not 0 <= elitism <= len(members):
        raise ValueError('elitism must fit the population')
    if next_id <= max(m.id for m in members):
        raise ValueError('child IDs must be new')
    ranked = sorted(members, key=lambda m: (-scores[m.id], m.id))
    children = []
    for index in range(len(members)):
        elite = index < elitism
        parent = ranked[index] if elite else tournament(members, scores, rng, tournament_size)
        genome, history = (parent.genome, []) if elite else mutate(parent.genome, rng, mutation_rate)
        children.append(Member(next_id + index, genome, parent.generation + 1, parent.id, history))
    return children
