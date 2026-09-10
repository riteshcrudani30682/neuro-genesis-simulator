"""Local-sensor baselines using independent seeded policy RNGs."""
import random
from worlds.creature.entities import Action, DELTAS
from evolution.genome import Genome


class RandomAgent:
    def __init__(self, genome=None, seed=0):
        self.genome = genome or Genome()
        self.rng = random.Random(seed)
        self.goal = None

    def set_goal(self, goal):
        """Future strategy boundary; validated metadata only, no LLM integration."""
        if goal not in (None, 'seek_food', 'avoid_competition', 'explore', 'protect_energy'):
            raise ValueError('unknown high-level goal')
        self.goal = goal

    def act(self, state, explore=True):
        # Intentionally genome-independent null baseline.
        return self.rng.choice(list(Action))

    def observe(self, state, action, reward, next_state, done):
        pass


class HeuristicAgent(RandomAgent):
    def act(self, state, explore=True):
        g = self.genome
        if explore and self.rng.random() < g.exploration_tendency:
            return self.rng.choice(list(Action))
        if self.rng.random() > g.movement_tendency:
            return Action.STAY
        cells = {(c[0], c[1]): c[2:] for c in state.cells}
        foods = [(c[0], c[1]) for c in state.cells if c[2]]
        scores = {}
        for action, (dx, dy) in DELTAS.items():
            _, hazard, occupied, boundary = cells[(dx, dy)]
            if boundary or occupied:
                scores[action] = -float('inf')
                continue
            attraction = max((1 / (1 + abs(x-dx) + abs(y-dy)) for x, y in foods), default=0)
            scores[action] = g.food_attraction * attraction - g.hazard_avoidance * hazard
            if action == Action.STAY:
                scores[action] -= 0.01  # weak tie preference for exploration
        best = max(scores.values())
        return self.rng.choice([a for a, value in scores.items() if value == best])
