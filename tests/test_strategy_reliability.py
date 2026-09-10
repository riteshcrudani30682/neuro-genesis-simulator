"""Scheduling tests use fake planners; they do not claim measured LLM improvement."""
from dataclasses import replace
import json

import pytest

from brains.scheduling import FairCallScheduler
from brains.strategy import (CallBudget, GoalDecision, MemoryStrategyAgent, OllamaPlanner,
                             StrategyController, STATIC_INSTRUCTIONS)
from experiments.strategy_runner import run_session
from memory.episodic import ExperienceMemory
from worlds.creature.entities import WorldConfig, Action
from worlds.creature.environment import MultiCreatureEnvironment


class Planner:
    def plan(self, context):
        return GoalDecision('seek_food', .8, 'Test fixture')


def state():
    return MultiCreatureEnvironment(WorldConfig(population=1)).reset(seed=11)[0]


def scheduled_run(reverse=False, fail=False):
    class Provider(Planner):
        def plan(self, context):
            if fail:
                raise TimeoutError('fixture timeout')
            return super().plan(context)
    budget = CallBudget(4)
    scheduler = FairCallScheduler(budget, quota=4, horizon=80, seed=42)
    agents = {i: MemoryStrategyAgent(ExperienceMemory(f'agent:{i}'), planner=Provider(), budget=budget)
              for i in range(4)}
    observations = {i: state() for i in agents}
    ids = list(reversed(agents)) if reverse else list(agents)
    for tick in range(80):
        scheduler.prepare({i: observations[i] for i in ids}, {i: agents[i] for i in ids}, tick)
        for i in ids:
            agents[i].features(observations[i])
            agents[i].observe(observations[i], Action.STAY, 0, observations[i], False)
    calls = sorted((e['world_tick'], i, e['source']) for i, agent in agents.items()
                   for e in agent.strategy.events if e['request_attempted'])
    return calls, scheduler, agents


def test_scheduler_spreads_calls_across_agents_and_time_independent_of_action_order():
    calls, scheduler, _ = scheduled_run()
    reverse, _, _ = scheduled_run(reverse=True)
    assert calls == reverse
    assert [tick for tick, _, _ in calls] == [0, 20, 40, 60]
    assert len({i for _, i, _ in calls}) == 4
    assert scheduler.used == scheduler.budget.used == 4


def test_failed_calls_consume_slots_without_retry_bursts():
    calls, scheduler, _ = scheduled_run(fail=True)
    assert len(calls) == 4 and scheduler.budget.used == 4
    assert all(source == 'error_fallback' for _, _, source in calls)
    assert [tick for tick, _, _ in calls] == [0, 20, 40, 60]


def test_dead_candidate_does_not_block_living_candidates_and_newborns_can_join():
    budget = CallBudget(2)
    scheduler = FairCallScheduler(budget, quota=2, horizon=40)
    a = MemoryStrategyAgent(ExperienceMemory('dead'), planner=Planner(), budget=budget)
    b = MemoryStrategyAgent(ExperienceMemory('child'), planner=Planner(), budget=budget)
    scheduler.prepare({0: state()}, {0: a}, 0)
    # Creature disappears before the next scheduling snapshot; no stale queue reservation.
    scheduler.prepare({1: state()}, {1: b}, 1)
    b.features(state())
    assert scheduler.used == 1 and b.strategy.events[-1]['source'] == 'llm'


def test_episode_quotas_and_run_logs_do_not_merge(tmp_path):
    config = WorldConfig(population=6, max_population=6, max_steps=80,
                         initial_energy=100, hazard_count=0, reproduction_enabled=False)
    first = run_session(config, episodes=2, planner=Planner(), max_calls=4, output=tmp_path)
    first_rows = [json.loads(s) for s in (tmp_path/'latest_strategy.jsonl').read_text().splitlines()]
    attempts = [r for r in first_rows if r['request_attempted']]
    assert [(e['episode'], e['world_tick']) for e in sorted(attempts, key=lambda e: (e['episode'], e['world_tick']))] == [(1, 0), (1, 40), (2, 0), (2, 40)]
    assert len({e['owner'] for e in attempts}) == 4
    assert all(r['run_id'] == first['run_id'] for r in first_rows)
    assert all(e['sources']['attempts'] == 2 for e in first['episodes'])
    second = run_session(config, episodes=1, planner=Planner(), max_calls=0, output=tmp_path)
    latest = [json.loads(s) for s in (tmp_path/'latest_strategy.jsonl').read_text().splitlines()]
    assert first['run_id'] != second['run_id']
    assert all(r['run_id'] == second['run_id'] and not r['request_attempted'] for r in latest)
    aggregate = [json.loads(s) for s in (tmp_path/'strategy.jsonl').read_text().splitlines()]
    assert len(aggregate) == len(first_rows) + len(latest)
    assert (tmp_path/'sessions'/first['run_id']/'strategy.jsonl').exists()


def test_scheduled_calls_respect_controller_cooldown_and_total_cap():
    budget = CallBudget(2)
    scheduler = FairCallScheduler(budget, quota=8, horizon=80)
    agent = MemoryStrategyAgent(ExperienceMemory('only'), planner=Planner(), budget=budget)
    for tick in range(80):
        observation = replace(state(), energy=.1 if tick % 2 else .8)
        scheduler.prepare({0: observation}, {0: agent}, tick)
        agent.features(observation)
        agent.tick += 1
    attempts = [e for e in agent.strategy.events if e['request_attempted']]
    assert len(attempts) == 2
    assert attempts[1]['tick'] - attempts[0]['tick'] >= 20


def test_legacy_memory_done_means_unknown_and_movement_values_are_named(tmp_path):
    memory = ExperienceMemory('legacy')
    memory.finish_episode(reward=25, steps=20, outcome='done')
    memory.remember(state(), Action.RIGHT, 2, state(), True)
    path = tmp_path/'memory.json'
    memory.save(path)
    loaded = ExperienceMemory.load(path, owner='legacy')
    assert loaded.to_dict() == memory.to_dict()
    context = loaded.context(state())
    assert context['recent_episodes'][0]['end_reason'] == 'ended_unspecified'
    assert set(context['movement_action_values']) == {a.name for a in Action}
    assert 'explore' not in context['movement_action_values']
    assert 'NEVER success' in STATIC_INSTRUCTIONS


@pytest.mark.parametrize('terminated,truncated,expected', [(True, False, 'died'), (False, True, 'time_limit'), (True, True, 'died')])
def test_memory_distinguishes_death_and_time_limit(terminated, truncated, expected):
    agent = MemoryStrategyAgent(ExperienceMemory('ending'))
    agent.observe(state(), Action.STAY, 1, state(), True, terminated=terminated, truncated=truncated)
    agent.finish_episode()
    assert len(agent.memory.episodes) == 1
    assert agent.memory.episodes[0]['outcome'] == expected


@pytest.mark.parametrize('content,reason,code', [
    ('{"goal":"explore","confidence":1,"reason":"' + 'x'*241 + '"}', 'stop', 'reason_length_or_type'),
    ('{"goal":', 'length', 'incomplete_generation'),
    ('not json', 'stop', 'malformed_json'),
    ('{"goal":"explore","confidence":1,"reason":"ok","command":"x"}', 'stop', 'unexpected_fields'),
])
def test_rejected_outputs_have_bounded_exact_diagnostics(content, reason, code):
    def transport(payload):
        assert payload['format']['additionalProperties'] is False
        return {'message': {'content': content}, 'done_reason': reason, 'eval_count': 160}
    controller = StrategyController(OllamaPlanner('fixture', transport=transport))
    controller.choose(state(), ExperienceMemory('test'), 0)
    event = controller.events[-1]
    assert event['source'] == 'error_fallback' and event['error_code'] == code
    assert event['diagnostics']['rejected_excerpt'] == content[:512]
    assert len(event['diagnostics']['rejected_excerpt']) <= 512
    assert event['usage']['eval_count'] == 160
    assert event['context']['senses']['energy'] == .5


def test_low_confidence_keeps_original_proposal_for_audit():
    planner = OllamaPlanner('fixture', transport=lambda _: {'message': {'content': '{"goal":"explore","confidence":0.1,"reason":"Unsure"}'}})
    controller = StrategyController(planner)
    controller.choose(state(), ExperienceMemory('test'), 0)
    event = controller.events[-1]
    assert event['source'] == 'confidence_fallback'
    assert event['proposed_decision']['confidence'] == .1
    assert event['error_code'] == 'low_confidence'


def test_event_sink_retains_events_beyond_viewer_history_bound():
    controller = StrategyController()
    captured = []
    controller.event_sink = captured.append
    for tick in range(0, 80*140, 80):
        controller.choose(state(), ExperienceMemory('test'), tick)
    assert len(controller.events) == 128
    assert len(captured) == 140
