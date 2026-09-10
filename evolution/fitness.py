"""Transparent lifetime fitness, with individually logged contributions."""
from statistics import mean, pvariance
from .genome import Genome


def fitness_components(creature):
    return {
        'survival': 0.02 * creature.age,
        'food': 3.0 * creature.food_eaten,
        'hazard_avoidance': 0.05 * creature.hazards_avoided,
        'hazard_damage': -2.0 * creature.hazards_hit,
        'energy_efficiency': creature.food_eaten / (1.0 + creature.energy_spent),
        'reproduction': 2.0 * creature.offspring_count,
        'blocked_movement': -0.1 * creature.blocked_moves,
    }


def fitness(creature):
    return sum(fitness_components(creature).values())


def genetic_diversity(genomes):
    """Mean normalized population variance; zero for clones/singletons."""
    genomes = list(genomes)
    if len(genomes) < 2:
        return 0.0
    return mean(pvariance([(getattr(g, name) - low) / (high - low) for g in genomes])
                for name, (low, high) in Genome.BOUNDS.items())
