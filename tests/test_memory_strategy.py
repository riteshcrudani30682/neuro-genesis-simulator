from dataclasses import replace
import json
import subprocess
import sys

import pytest

from brains.strategy import (CallBudget, GoalDecision, MemoryStrategyAgent, OllamaPlanner,
                             StrategyController, strategy_features)
from memory.episodic import ExperienceMemory
from worlds.creature.environment import MultiCreatureEnvironment
from worlds.creature.entities import Action, WorldConfig
from experiments.strategy_runner import run_session


def world():
    env = MultiCreatureEnvironment(WorldConfig(population=1, max_population=4, max_steps=5))
    env.reset(seed=42)
    return env


def test_memory_learns_action_outcomes_and_persists(tmp_path):
    state = world().observations()[0]
    memory = ExperienceMemory('test:0')
    for _ in range(5):
        memory.remember(state, Action.RIGHT, 2, state, False)
    assert memory.action_values(state)[Action.RIGHT] == pytest.approx(2/3)
    assert memory.action_values(state)[Action.LEFT] == 0
    memory.finish_episode(reward=10, steps=5, outcome='done')
    path = tmp_path/'memory.json'
    memory.save(path)
    restored = ExperienceMemory.load(path, owner='test:0')
    assert restored.to_dict() == memory.to_dict()
    with pytest.raises(ValueError):
        ExperienceMemory.load(path, owner='someone_else')


def test_memory_is_bounded_and_finite(tmp_path):
    state = world().observations()[0]
    memory = ExperienceMemory('test', capacity=2, episode_capacity=2)
    for energy in (.1, .3, .6, .9):
        memory.remember(replace(state, energy=energy), Action.STAY, 1, state, False)
        memory.finish_episode(reward=1, steps=1, outcome='done')
    assert len(memory.patterns) == 2 and len(memory.episodes) == 2
    with pytest.raises(ValueError):
        memory.remember(state, Action.STAY, float('nan'), state, False)
    assert memory.transitions == 4


def test_bad_memory_schema_rejected(tmp_path):
    path = tmp_path/'memory.json'
    path.write_text('{"schema_version":99}')
    with pytest.raises(ValueError):
        ExperienceMemory.load(path, owner='test')


@pytest.mark.parametrize('data', [
    {'goal': 'UP', 'confidence': .8, 'reason': 'x'},
    {'goal': 'explore', 'confidence': float('nan'), 'reason': 'x'},
    {'goal': 'explore', 'confidence': True, 'reason': 'x'},
    {'goal': 'explore', 'confidence': .8, 'reason': 'x', 'command': 'anything'},
    {'goal': 'explore', 'confidence': .8, 'reason': 'x'*241},
    'not JSON',
])
def test_goal_schema_rejects_invalid_output(data):
    with pytest.raises((ValueError, TypeError)):
        GoalDecision.parse(data)


class CountingPlanner:
    def __init__(self, fail=False):
        self.contexts = []
        self.fail = fail
    def plan(self, context):
        self.contexts.append(context)
        if self.fail:
            raise TimeoutError('simulated timeout')
        return GoalDecision('protect_energy', .9, 'Preserve energy')


def test_strategy_gate_never_calls_every_frame():
    state = world().observations()[0]
    planner = CountingPlanner()
    controller = StrategyController(planner)
    for tick in range(101):
        controller.choose(state, ExperienceMemory('test'), tick)
    assert len(planner.contexts) == 2  # t=0 and t=80, not 101 calls.
    assert controller.goal == 'protect_energy'
    assert set(planner.contexts[0]) == {'senses', 'current_goal', 'memory'}
    assert 'position' not in str(planner.contexts[0])


def test_event_gate_respects_minimum_interval():
    state = world().observations()[0]
    planner = CountingPlanner()
    controller = StrategyController(planner)
    memory = ExperienceMemory('test')
    controller.choose(state, memory, 0)
    low = replace(state, energy=.1)
    controller.choose(low, memory, 1)
    assert len(planner.contexts) == 1
    controller.choose(low, memory, 20)
    assert len(planner.contexts) == 2


def test_timeout_and_global_budget_fallback():
    state = world().observations()[0]
    planner = CountingPlanner(fail=True)
    budget = CallBudget(1)
    a, b = StrategyController(planner, budget=budget), StrategyController(planner, budget=budget)
    a.choose(state, ExperienceMemory('a'), 0)
    b.choose(state, ExperienceMemory('b'), 0)
    assert budget.used == 1 and len(planner.contexts) == 1
    assert a.events[-1]['source'] == 'error_fallback'
    assert b.events[-1]['source'] == 'budget_fallback'
    a.choose(state, ExperienceMemory('a'), 1)
    assert len(planner.contexts) == 1


def test_ollama_payload_and_parse_without_real_network():
    captured = []
    def transport(payload):
        captured.append(payload)
        return {'message': {'content': '{"goal":"seek_food","confidence":0.8,"reason":"Food visible"}'}}
    planner = OllamaPlanner('installed-test-model', transport=transport)
    assert planner.plan({'senses': {}}).goal == 'seek_food'
    assert captured[0]['stream'] is False
    assert captured[0]['options']['num_predict'] == 160
    assert captured[0]['format']['properties']['goal']['enum'] == ['seek_food', 'avoid_competition', 'explore', 'protect_energy']
    assert captured[0]['format']['additionalProperties'] is False
    assert len(captured[0]['messages']) == 2


def test_goals_change_low_level_behavior():
    env = world()
    env.creatures[0].position = (10, 10)
    env.food.clear()
    env.hazards.clear()
    state = env.observations()[0]
    agent = MemoryStrategyAgent(ExperienceMemory('test'), planner=CountingPlanner())
    assert agent.act(state, explore=False) == Action.STAY
    assert len(agent.features(state)) == 111
    assert strategy_features(state, 'explore', agent.memory) != strategy_features(state, 'seek_food', agent.memory)


def test_memory_survives_sessions_and_episode_summary_is_once(tmp_path):
    config = replace(world().config, reproduction_enabled=False)
    run_session(config, episodes=1, output=tmp_path, seed=42)
    files = list((tmp_path/'memory').glob('*.json'))
    first = json.loads(files[0].read_text())
    run_session(config, episodes=1, output=tmp_path, seed=42)
    second = json.loads(files[0].read_text())
    assert second['transitions'] == 2*first['transitions']
    assert len(second['episodes']) == 2
    assert len(list((tmp_path/'memory').glob('*.json'))) == 1


def test_memory_strategy_headless_import_has_no_torch_or_pygame():
    subprocess.run([sys.executable, '-c', "import experiments.strategy_runner; import sys; assert 'torch' not in sys.modules; assert 'pygame' not in sys.modules"], check=True)


def test_viewer_handles_newborn_before_controller_creation(monkeypatch):
    monkeypatch.setenv('SDL_VIDEODRIVER', 'dummy')
    monkeypatch.setenv('SDL_AUDIODRIVER', 'dummy')
    from ui.creature_view import CreatureView
    env = world()
    # No controller yet, as with a child born in the just-completed world tick.
    viewer = CreatureView(env.config, {}, CallBudget())
    try:
        viewer(env, None)
    finally:
        viewer.close()


def test_low_confidence_response_uses_labeled_fallback():
    class Uncertain:
        def plan(self, context):
            return GoalDecision('explore', .1, 'Unsure')
    controller = StrategyController(Uncertain())
    controller.choose(world().observations()[0], ExperienceMemory('test'), 0)
    assert controller.events[-1]['source'] == 'confidence_fallback'


def test_ppo_strategy_session_uses_persistent_memory(tmp_path):
    from brains.ppo import PPOTrainer, PPOConfig
    trainer = PPOTrainer(PPOConfig(hidden=8, epochs=1))
    summary = run_session(replace(world().config, max_steps=2), episodes=1, output=tmp_path, trainer=trainer)
    assert summary['policy'] == 'ppo' and summary['llm_calls'] == 0
    assert list((tmp_path/'memory').glob('*.json'))


def test_disabled_memory_is_an_actual_ablation():
    state = world().observations()[0]
    memory = ExperienceMemory('ablation', enabled=False)
    memory.remember(state, Action.RIGHT, 2, state, True)
    memory.finish_episode(reward=2, steps=1, outcome='done')
    assert memory.action_values(state) == (0, 0, 0, 0, 0)
    assert not memory.patterns and not memory.episodes and memory.transitions == 0
