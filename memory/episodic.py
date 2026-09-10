"""Learn reusable local action-outcome associations, never an omniscient map."""
from collections import OrderedDict
import json
import math
from pathlib import Path
import tempfile

from worlds.creature.entities import Action


class ExperienceMemory:
    SCHEMA = 1
    SENSOR_SCHEMA = 'creature-local-v1'

    def __init__(self, owner, *, capacity=128, episode_capacity=16, enabled=True):
        if not isinstance(owner, str) or not owner or len(owner) > 200:
            raise ValueError('owner must be a nonempty short namespace/identity')
        if type(capacity) is not int or not 1 <= capacity <= 4096:
            raise ValueError('memory capacity must be 1..4096')
        if type(episode_capacity) is not int or not 1 <= episode_capacity <= 256:
            raise ValueError('episode capacity must be 1..256')
        if type(enabled) is not bool:
            raise ValueError('enabled must be boolean')
        self.enabled = enabled
        self.owner = owner
        self.capacity = capacity
        self.episode_capacity = episode_capacity
        self.patterns = OrderedDict()
        self.episodes = []
        self.transitions = 0

    @staticmethod
    def signature(state):
        # Own energy bucket and immediate local cross; no ID, location or hidden world.
        local = [tuple(c[2:]) for c in state.cells if abs(c[0]) + abs(c[1]) <= 1]
        return json.dumps([min(3, int(state.energy * 4)), local], separators=(',', ':'))

    def remember(self, state, action, reward, next_state, done):
        if not self.enabled:
            return
        action = Action(action)
        if not math.isfinite(reward) or not -100 <= reward <= 100:
            raise ValueError('reward must be finite and within the memory schema range')
        key = self.signature(state)
        if key not in self.patterns:
            self.patterns[key] = [[0, 0.0] for _ in Action]
        self.patterns.move_to_end(key)
        item = self.patterns[key][int(action)]
        item[0] += 1
        item[1] += (float(reward) - item[1]) / item[0]
        while len(self.patterns) > self.capacity:
            self.patterns.popitem(last=False)
        self.transitions += 1

    def action_values(self, state):
        """Confidence-shrunk empirical reward; unseen actions remain zero."""
        if not self.enabled:
            return (0.0,) * len(Action)
        stats = self.patterns.get(self.signature(state), [[0, 0.0] for _ in Action])
        return tuple(max(-1.0, min(1.0, avg / 3)) * min(count / 5, 1.0) for count, avg in stats)

    def finish_episode(self, *, reward, steps, outcome):
        if not math.isfinite(reward) or type(steps) is not int or steps < 0:
            raise ValueError('invalid episode summary')
        if outcome not in ('done', 'interrupted', 'died', 'time_limit', 'ended_unspecified'):
            raise ValueError('invalid outcome')
        if not self.enabled:
            return
        self.episodes.append({'reward': float(reward), 'steps': steps, 'outcome': outcome})
        self.episodes = self.episodes[-self.episode_capacity:]

    def context(self, state):
        """Small structured retrieval, not a growing conversation transcript."""
        return {'schema': 'empirical-movement-memory-v2', 'transitions_seen': self.transitions,
                'movement_action_values': {a.name: v for a, v in zip(Action, self.action_values(state))},
                'recent_episodes': [{'reward': e['reward'], 'steps': e['steps'],
                                     'end_reason': 'ended_unspecified' if e['outcome'] == 'done' else e['outcome']}
                                    for e in self.episodes[-3:]]}

    def to_dict(self):
        return {'schema_version': self.SCHEMA, 'sensor_schema': self.SENSOR_SCHEMA,
                'owner': self.owner, 'enabled': self.enabled, 'capacity': self.capacity, 'episode_capacity': self.episode_capacity,
                'patterns': list(self.patterns.items()), 'episodes': self.episodes,
                'transitions': self.transitions}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = None
        try:
            with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as handle:
                temp = Path(handle.name)
                json.dump(self.to_dict(), handle, allow_nan=False)
            temp.replace(path)
        finally:
            if temp and temp.exists():
                temp.unlink()

    @classmethod
    def load(cls, path, *, owner):
        path = Path(path)
        if path.stat().st_size > 4_000_000:
            raise ValueError('memory file is too large')
        data = json.loads(path.read_text(encoding='utf-8'))
        if (data.get('schema_version') != cls.SCHEMA or data.get('sensor_schema') != cls.SENSOR_SCHEMA
                or data.get('owner') != owner):
            raise ValueError('memory schema/owner mismatch')
        memory = cls(owner, capacity=data['capacity'], episode_capacity=data['episode_capacity'], enabled=data.get('enabled', True))
        if len(data['patterns']) > memory.capacity or len(data['episodes']) > memory.episode_capacity:
            raise ValueError('memory capacity exceeded')
        for key, stats in data['patterns']:
            if not isinstance(key, str) or len(key) > 512 or key in memory.patterns or len(stats) != len(Action):
                raise ValueError('invalid memory pattern')
            for count, avg in stats:
                if type(count) is not int or count < 0 or not math.isfinite(avg) or not -100 <= avg <= 100:
                    raise ValueError('invalid action statistics')
            memory.patterns[key] = stats
        for episode in data['episodes']:
            memory.finish_episode(**episode)
        if type(data['transitions']) is not int or data['transitions'] < 0:
            raise ValueError('invalid transition count')
        memory.transitions = data['transitions']
        return memory
