"""A slow high-level goal planner above local controllers. No arbitrary actions/code."""
from dataclasses import dataclass
import json
import math
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from worlds.creature.entities import Action, DELTAS
from .baseline import RandomAgent

GOALS = ('seek_food', 'avoid_competition', 'explore', 'protect_energy')
STATIC_INSTRUCTIONS = (
    'You select one high-level goal for a simulated creature. Input contains only local senses '
    'and empirical memory, not a world map. Return JSON with exactly goal, confidence, reason. '
    'goal must be seek_food, avoid_competition, explore, or protect_energy. '
    'confidence is 0..1; reason is a short plain summary. Do not return movement actions, '
    'code, tools, commands, or claims of intelligence.'
)


@dataclass(frozen=True)
class GoalDecision:
    goal: str
    confidence: float
    reason: str

    @classmethod
    def parse(cls, data):
        if isinstance(data, str):
            if len(data) > 2048:
                raise ValueError('oversized strategy response')
            data = json.loads(data)
        if not isinstance(data, dict) or set(data) != {'goal', 'confidence', 'reason'}:
            raise ValueError('unexpected strategy fields')
        if data['goal'] not in GOALS:
            raise ValueError('invalid goal')
        confidence = data['confidence']
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError('confidence must be numeric')
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('invalid confidence')
        if not isinstance(data['reason'], str) or not 1 <= len(data['reason']) <= 240:
            raise ValueError('reason must contain 1..240 characters')
        return cls(**data)


class RulePlanner:
    """Explicit offline baseline and fallback, never presented as an LLM."""
    def plan(self, context):
        senses = context['senses']
        if senses['energy'] < .3:
            goal = 'seek_food' if senses['food'] else 'protect_energy'
        elif senses['creatures'] > 3:
            goal = 'avoid_competition'
        elif senses['food']:
            goal = 'seek_food'
        else:
            goal = 'explore'
        return GoalDecision(goal, 1.0, 'Deterministic local-sensor rule')


class OllamaPlanner:
    """Opt-in local inference via Ollama's documented /api/chat protocol."""
    def __init__(self, model, *, endpoint='http://127.0.0.1:11434', timeout=10, transport=None):
        parsed = urlparse(endpoint)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('invalid Ollama endpoint')
        if not model or len(model) > 200 or not 0 < timeout <= 60:
            raise ValueError('explicit model and timeout in (0,60] required')
        self.model, self.endpoint, self.timeout = model, endpoint.rstrip('/'), timeout
        self.transport = transport
        self.last_usage = {}

    def plan(self, context):
        payload = {'model': self.model, 'stream': False, 'format': 'json', 'think': False,
                   'keep_alive': '5m', 'options': {'temperature': 0, 'num_predict': 96, 'num_ctx': 2048},
                   'messages': [{'role': 'system', 'content': STATIC_INSTRUCTIONS},
                                {'role': 'user', 'content': json.dumps(context, allow_nan=False)}]}
        raw = json.dumps(payload).encode()
        if len(raw) > 8000:
            raise ValueError('strategy input budget exceeded')
        if self.transport:
            response = self.transport(payload)
        else:
            request = Request(self.endpoint+'/api/chat', data=raw, headers={'Content-Type': 'application/json'})
            with urlopen(request, timeout=self.timeout) as handle:
                data = handle.read(65537)
            if len(data) > 65536:
                raise ValueError('oversized Ollama response')
            response = json.loads(data)
        self.last_usage = {key: response[key] for key in ('prompt_eval_count', 'eval_count')
                           if type(response.get(key)) is int and response[key] >= 0}
        return GoalDecision.parse(response['message']['content'])


@dataclass
class CallBudget:
    maximum: int = 10
    used: int = 0

    def __post_init__(self):
        if type(self.maximum) is not int or self.maximum < 0:
            raise ValueError('call budget must be a nonnegative integer')

    def take(self):
        if self.used >= self.maximum:
            return False
        self.used += 1
        return True


class StrategyController:
    def __init__(self, planner=None, *, budget=None, min_interval=20, max_interval=80):
        if not 1 <= min_interval <= max_interval:
            raise ValueError('strategy interval must be positive and ordered')
        self.planner = planner or RulePlanner()
        self.budget = budget or CallBudget()
        self.min_interval, self.max_interval = min_interval, max_interval
        self.goal = 'explore'
        self.last_tick = None
        self.last_signature = None
        self.events = []

    def choose(self, state, memory, tick):
        senses = {'energy': round(state.energy, 3), 'age': round(state.age, 3),
                  'food': sum(c[2] for c in state.cells), 'hazards': sum(c[3] for c in state.cells),
                  'creatures': sum(c[4] for c in state.cells)}
        signature = (state.energy < .3, bool(senses['food']), bool(senses['hazards']), senses['creatures'] > 3)
        elapsed = 0 if self.last_tick is None else tick - self.last_tick
        due = self.last_tick is None or (elapsed >= self.min_interval and
                                        (signature != self.last_signature or elapsed >= self.max_interval))
        if not due:
            return self.goal
        self.last_tick, self.last_signature = tick, signature  # Failed calls also consume cooldown.
        context = {'senses': senses, 'current_goal': self.goal, 'memory': memory.context(state)}
        source, error = 'rule', None
        planner = self.planner
        if not isinstance(planner, RulePlanner):
            if self.budget.take():
                source = 'llm'
            else:
                planner, source = RulePlanner(), 'budget_fallback'
        attempted = source == 'llm'
        try:
            decision = planner.plan(context)
            # Do not trust even a custom provider to bypass the output schema.
            decision = GoalDecision.parse(vars(decision))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            error = type(exc).__name__  # Avoid logging endpoints/credentials/large remote responses.
            decision, source = RulePlanner().plan(context), 'error_fallback'
        if decision.confidence < .4:
            decision, source = RulePlanner().plan(context), 'confidence_fallback'
        self.goal = decision.goal
        self.events.append({'tick': tick, 'goal': decision.goal, 'confidence': decision.confidence,
                            'reason': decision.reason, 'source': source, 'error': error,
                            'request_attempted': attempted,
                            'usage': getattr(planner, 'last_usage', {}) if attempted and not error else {}})
        self.events = self.events[-128:]
        return self.goal


def strategy_features(state, goal, memory):
    return state.vector() + tuple(float(goal == g) for g in GOALS) + memory.action_values(state)


class MemoryStrategyAgent(RandomAgent):
    """Agent-contract wrapper: episodic memory + slow strategy + local low-level policy."""
    def __init__(self, memory, *, seed=0, genome=None, planner=None, budget=None, low_level=None):
        super().__init__(genome, seed)
        self.memory = memory
        self.strategy = StrategyController(planner, budget=budget)
        self.low_level = low_level
        self.tick = 0
        self.episode_reward = 0.0
        self.episode_finished = False

    def features(self, state):
        goal = self.strategy.choose(state, self.memory, self.tick)
        self.goal = goal
        return strategy_features(state, goal, self.memory)

    def act(self, state, explore=True):
        features = self.features(state)
        if self.low_level is not None:
            return self.low_level.act_features(features, explore=explore)
        cells = {(c[0], c[1]): c[2:] for c in state.cells}
        foods = [(c[0], c[1]) for c in state.cells if c[2]]
        others = [(c[0], c[1]) for c in state.cells if c[4]]
        memory_values = self.memory.action_values(state)
        scores = {}
        for action, (dx, dy) in DELTAS.items():
            _, hazard, occupied, boundary = cells[(dx, dy)]
            if boundary or occupied:
                continue
            food = max((1/(1+abs(x-dx)+abs(y-dy)) for x, y in foods), default=0)
            crowd = sum(1/(1+abs(x-dx)+abs(y-dy)) for x, y in others)
            score = self.genome.food_attraction * food - 2*self.genome.hazard_avoidance*hazard
            if self.goal == 'seek_food':
                score += 2*food
            elif self.goal == 'avoid_competition':
                score -= .5*crowd
            elif self.goal == 'protect_energy':
                score += .4 if action == Action.STAY else -.1
            elif self.goal == 'explore':
                score += .05 if action != Action.STAY else 0
            scores[action] = score + .3*memory_values[action]
        if explore and self.goal == 'explore' and self.rng.random() < self.genome.exploration_tendency:
            return self.rng.choice(list(scores))
        best = max(scores.values())
        return self.rng.choice([a for a, score in scores.items() if score == best])

    def observe(self, state, action, reward, next_state, done):
        self.memory.remember(state, action, reward, next_state, done)
        self.tick += 1
        self.episode_reward += reward
        if done:
            self.finish_episode()

    def finish_episode(self, interrupted=False):
        if not self.episode_finished:
            self.memory.finish_episode(reward=self.episode_reward, steps=self.tick,
                                       outcome='interrupted' if interrupted else 'done')
            self.episode_finished = True
