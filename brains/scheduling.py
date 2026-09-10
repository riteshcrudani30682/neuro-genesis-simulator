"""Deterministic shared LLM admission, separate from local action selection."""
import hashlib
import math


class FairCallScheduler:
    def __init__(self, budget, *, quota, horizon, seed=0, served=None, min_gap=20):
        if type(quota) is not int or quota < 0 or horizon < 1 or min_gap < 1:
            raise ValueError('invalid strategy schedule')
        self.budget, self.quota = budget, quota
        self.gap = max(min_gap, math.ceil(horizon / max(1, quota)))
        self.seed = seed
        self.served = served if served is not None else {}
        self.waiting = {}
        self.used = 0
        self.next_tick = 0
        self.tick = 0
        self.selected = None

    def prepare(self, observations, agents, world_tick):
        """Inspect ALL eligible local controllers before any creature acts."""
        self.tick, self.selected = world_tick, None
        candidates = {}
        for i in sorted(observations):
            agent = agents[i]
            controller = agent.strategy
            controller.scheduler, controller.world_tick = self, world_tick
            if controller.is_request_due(observations[i], agent.tick):
                owner = agent.memory.owner
                self.waiting.setdefault(owner, world_tick)
                candidates[owner] = controller
        self.waiting = {owner: since for owner, since in self.waiting.items() if owner in candidates}
        if self.used >= self.quota or self.budget.used >= self.budget.maximum or world_tick < self.next_tick:
            return
        def priority(owner):
            tie = hashlib.sha256(f'{self.seed}:{owner}'.encode()).hexdigest()
            return self.served.get(owner, 0), self.waiting[owner], tie
        if candidates:
            self.selected = min(candidates, key=priority)

    def take(self, owner):
        if owner != self.selected or self.used >= self.quota or not self.budget.take():
            return False
        self.selected = None
        self.used += 1  # Failed requests count too; no automatic retry burst.
        self.served[owner] = self.served.get(owner, 0) + 1
        self.waiting.pop(owner, None)
        self.next_tick = self.tick + self.gap
        return True

    def fallback_source(self):
        if self.budget.used >= self.budget.maximum:
            return 'budget_fallback'
        return 'episode_budget_fallback' if self.used >= self.quota else 'scheduled_fallback'
