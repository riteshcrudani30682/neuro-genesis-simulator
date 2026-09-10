"""World invariants and experimental correctness, including regression contracts."""
from dataclasses import replace
import json
import random
import subprocess
import sys

import pytest

from core.replay import ReplayBuffer
from experiments.runner import run_episode
from experiments.population_runner import (EvolutionExperiment, ExperimentConfig,
                                            evaluate_population, run_population_episode)
from evolution.genome import Genome
from evolution.mutation import mutate
from evolution.population import Member, breed
from evolution.selection import tournament
from evolution.fitness import fitness, fitness_components, genetic_diversity
from brains.baseline import RandomAgent, HeuristicAgent
from worlds.creature.entities import Action, Creature, WorldConfig
from worlds.creature.environment import CreatureEnvironment, MultiCreatureEnvironment


def config(**changes):
    base = WorldConfig(width=8, height=8, population=2, max_population=8,
                       food_count=0, hazard_count=0, food_regrowth=0,
                       max_steps=20, reproduction_enabled=False)
    return replace(base, **changes)


def world(**changes):
    env = MultiCreatureEnvironment(config(**changes))
    env.reset(seed=7)
    return env


def stay(env):
    return {i: Action.STAY for i in env.alive}


def reproducible_state(env):
    return env.creature_metrics(), sorted(env.food), sorted(env.hazards), env.rng.getstate()


def test_deterministic_reset_and_steps():
    a, b = world(food_count=8, hazard_count=3), world(food_count=8, hazard_count=3)
    for _ in range(8):
        assert reproducible_state(a) == reproducible_state(b)
        assert a.step(stay(a)) == b.step(stay(b))
    a.reset(seed=7)
    b.reset(seed=7)
    assert reproducible_state(a) == reproducible_state(b)


def test_all_creatures_act_one_world_tick():
    env = world()
    env.creatures[0].position = (1, 1)
    env.creatures[1].position = (5, 5)
    result = env.step({0: Action.RIGHT, 1: Action.UP})
    assert env.step_count == 1
    assert env.creatures[0].position == (2, 1)
    assert env.creatures[1].position == (5, 4)
    assert all(c.age == 1 for c in env.alive.values())
    assert len(result.transitions) == 2


def test_action_and_entity_order_do_not_change_conflicts():
    a, b = world(), world()
    for env in (a, b):
        env.creatures[0].position = (1, 1)
        env.creatures[1].position = (3, 1)
    b.creatures = dict(reversed(list(b.creatures.items())))
    first = a.step({0: Action.RIGHT, 1: Action.LEFT})
    second = b.step({1: Action.LEFT, 0: Action.RIGHT})
    assert first == second
    assert reproducible_state(a) == reproducible_state(b)
    assert sum(c.position == (2, 1) for c in a.alive.values()) == 1


def test_swap_and_stationary_occupant():
    env = world()
    env.creatures[0].position, env.creatures[1].position = (1, 1), (2, 1)
    env.step({0: Action.RIGHT, 1: Action.LEFT})
    assert env.creatures[0].position == (2, 1)
    assert env.creatures[1].position == (1, 1)
    env.step({0: Action.LEFT, 1: Action.STAY})
    assert env.creatures[0].position == (2, 1)
    assert env.creatures[0].blocked_moves == 1


def test_blocked_chain_propagates_without_overlap():
    env = world(population=3)
    for i in range(3):
        env.creatures[i].position = (i+1, 1)
    env.step({0: Action.RIGHT, 1: Action.RIGHT, 2: Action.STAY})
    assert [c.position for c in env.alive.values()] == [(1, 1), (2, 1), (3, 1)]


@pytest.mark.parametrize('action, cost', [(Action.STAY, .25), (Action.UP, .6)])
def test_energy_cost(action, cost):
    env = world()
    env.step({0: action, 1: Action.STAY})
    assert env.creatures[0].energy == pytest.approx(50-cost)


def test_food_and_transition_alignment():
    env = world()
    env.creatures[0].position, env.creatures[1].position = (1, 1), (6, 6)
    env.food = {(2, 1)}
    previous = env.observations()[0]
    result = env.step({0: Action.RIGHT, 1: Action.STAY})
    c = env.creatures[0]
    assert c.food_eaten == 1
    assert c.energy == pytest.approx(61.4)
    assert result.rewards[0] == 2.01
    assert (2, 1) not in env.food
    t = result.transitions[0]
    assert t.state == previous and t.action == Action.RIGHT
    assert t.next_state == result.observations[0] and t.next_state != previous


def test_hazard_penalty_and_death():
    env = world()
    env.creatures[0].position, env.creatures[1].position = (1, 1), (6, 6)
    env.creatures[0].energy = 10
    env.hazards = {(2, 1)}
    result = env.step({0: Action.RIGHT, 1: Action.STAY})
    assert env.creatures[0].hazards_hit == 1
    assert result.rewards[0] == -3
    assert result.deaths == (0,)
    assert result.terminated[0] and result.transitions[0].done
    assert result.transitions[0].next_state.energy == 0
    assert 0 not in result.observations


@pytest.mark.parametrize('reason', ['energy', 'age'])
def test_death_sources(reason):
    env = world(population=1, max_age=2)
    if reason == 'energy':
        env.creatures[0].energy = .1
    else:
        env.creatures[0].age = 1
    result = env.step(stay(env))
    assert result.done and not env.alive
    with pytest.raises(RuntimeError):
        env.step({})


def reproductive_world(**changes):
    env = world(population=1, reproduction_enabled=True, initial_energy=90,
                min_reproduction_age=1, **changes)
    env.creatures[0].position = (3, 3)
    return env


def test_reproduction_cost_lineage_generation_and_mutation():
    env = reproductive_world(mutation_rate=1)
    parent = env.creatures[0]
    result = env.step(stay(env))
    assert len(result.births) == 1
    child = env.creatures[result.births[0]]
    assert parent.energy == pytest.approx(59.75)
    assert child.energy == 25 and parent.offspring_count == 1
    assert child.generation == parent.generation + 1
    assert child.parent_id == parent.id and child.id != parent.id
    assert child.birth_step == 1 and child.age == 0
    assert child.mutation_history
    assert parent.energy + child.energy < 90
    assert child.id in result.observations and child.id not in result.transitions


@pytest.mark.parametrize('condition', ['age', 'energy', 'space', 'cooldown', 'cap', 'genome'])
def test_reproduction_requires_all_conditions(condition):
    env = reproductive_world()
    parent = env.creatures[0]
    if condition == 'age':
        env.config = replace(env.config, min_reproduction_age=10)
    elif condition == 'energy':
        parent.energy = 40
    elif condition == 'space':
        env.hazards = {(3, 2), (3, 4), (2, 3), (4, 3)}
    elif condition == 'cooldown':
        parent.last_birth_step = 0
    elif condition == 'cap':
        env.config = replace(env.config, max_population=1)
    elif condition == 'genome':
        parent.genome = replace(parent.genome, reproduction_threshold=95)
    energy = parent.energy
    result = env.step(stay(env))
    assert not result.births
    assert parent.energy == pytest.approx(energy-.25)


def test_population_cap_across_many_fertile_parents():
    env = world(population=4, max_population=5, initial_energy=90, reproduction_enabled=True,
                min_reproduction_age=1, reproduction_cooldown=1)
    for _ in range(4):
        env.step(stay(env))
        assert len(env.alive) <= 5
    assert len(env.creatures) == 5


def test_same_seed_reproduction_and_mapping_order():
    a, b = reproductive_world(), reproductive_world()
    for _ in range(5):
        x = a.step(stay(a))
        y = b.step(dict(reversed(list(stay(b).items()))))
        assert x == y
        assert reproducible_state(a) == reproducible_state(b)


@pytest.mark.parametrize('seed', [0, 42, 999])
def test_mutation_seed_and_bounds(seed):
    parent = Genome()
    first = mutate(parent, random.Random(seed), rate=1)
    assert first == mutate(parent, random.Random(seed), rate=1)
    for _ in range(50):
        parent, _ = mutate(parent, random.Random(seed), rate=1)
        for name, (low, high) in Genome.BOUNDS.items():
            assert low <= getattr(parent, name) <= high
    assert mutate(parent, random.Random(seed), rate=0) == (parent, [])


def test_serializable_genome_and_diversity():
    genome = Genome()
    assert Genome.from_dict(json.loads(json.dumps(genome.to_dict()))) == genome
    assert genetic_diversity([genome]*5) == 0
    assert genetic_diversity([genome, replace(genome, food_attraction=3)]) > 0
    with pytest.raises(ValueError):
        Genome(exploration_tendency=1.1)
    with pytest.raises(ValueError):
        Genome(mutation_scale=float('nan'))


def test_transparent_fitness_not_raw_reward():
    c = Creature(0, (0, 0), 50, Genome(), age=10, food_eaten=2, hazards_hit=1,
                 hazards_avoided=4, offspring_count=1, blocked_moves=3, energy_spent=9, total_reward=999)
    parts = fitness_components(c)
    assert parts == pytest.approx({'survival': .2, 'food': 6, 'hazard_avoidance': .2,
                                  'hazard_damage': -2, 'energy_efficiency': .2,
                                  'reproduction': 2, 'blocked_movement': -.3})
    assert fitness(c) == pytest.approx(6.3)


def test_selection_and_elitism():
    members = [Member(i, Genome(food_attraction=i)) for i in range(3)]
    scores = {0: 1, 1: 10, 2: -5}
    assert tournament(members, scores, random.Random(2), size=3).id == 1
    children = breed(members, scores, random.Random(3), next_id=3, elitism=1, mutation_rate=1)
    assert children[0].genome == members[1].genome
    assert not children[0].mutation_history
    assert children[0].parent_id == 1
    assert {c.id for c in children} == {3, 4, 5}
    assert all(c.generation == 1 for c in children)


def experiment():
    return EvolutionExperiment(ExperimentConfig(config(food_count=12, hazard_count=3, max_steps=8),
                                                evaluation_seeds=(11, 22), heldout_seeds=(33, 44), elitism=1))


def test_generations_repeated_evaluation_and_metrics(tmp_path):
    exp = experiment()
    first = exp.advance(tmp_path)
    exp.advance(tmp_path)
    assert exp.generation == 2 and all(m.generation == 2 for m in exp.members)
    assert first['evaluation_seeds'] == [11, 22]
    assert len(first['seed_mean_fitness']) == 2
    rows = [json.loads(line) for line in (tmp_path/'creatures.jsonl').read_text().splitlines()]
    assert len(rows) == 8
    assert {'parent_id', 'fitness_components', 'lifetime', 'seed'} <= set(rows[0])
    assert len((tmp_path/'generations.jsonl').read_text().splitlines()) == 2


def test_checkpoint_exact_resume(tmp_path):
    uninterrupted = experiment()
    uninterrupted.advance()
    path = tmp_path/'checkpoint.json'
    uninterrupted.save(path)
    resumed = EvolutionExperiment.load(path)
    assert resumed.members == uninterrupted.members
    assert resumed.rng.getstate() == uninterrupted.rng.getstate()
    assert resumed.advance() == uninterrupted.advance()
    assert resumed.members == uninterrupted.members
    assert resumed.compare_baselines() == uninterrupted.compare_baselines()


def test_checkpoint_rejects_bad_schema(tmp_path):
    path = tmp_path/'bad.json'
    path.write_text('{"schema_version": 99}')
    with pytest.raises(ValueError):
        EvolutionExperiment.load(path)


def test_baselines_use_separate_heldout_seeds():
    exp = experiment()
    report = exp.compare_baselines()
    assert [r['baseline'] for r in report] == ['evolved', 'fixed_initial_heuristic', 'random']
    assert all(r['evaluation_seeds'] == [33, 44] for r in report)
    with pytest.raises(ValueError):
        replace(exp.config, heldout_seeds=(11,))


def test_local_observation_hides_distant_food_and_agents():
    env = world()
    env.creatures[0].position, env.creatures[1].position = (1, 1), (7, 7)
    before = env.observations()[0]
    env.food.add((7, 6))
    env.creatures[1].position = (6, 7)
    assert env.observations()[0] == before
    assert len(before.vector()) == 102
    assert any(c[-1] for c in before.cells)


def test_invalid_action_is_atomic():
    env = world()
    before = reproducible_state(env)
    for bad in ({0: Action.UP}, {0: 99, 1: Action.STAY}, {0: True, 1: Action.STAY}):
        with pytest.raises(ValueError):
            env.step(bad)
        assert reproducible_state(env) == before
        assert env.step_count == 0


def test_single_creature_reuses_contract_and_replay():
    class RecordingAgent(RandomAgent):
        def __init__(self):
            super().__init__(seed=4)
            self.replay = ReplayBuffer(100)
        def observe(self, state, action, reward, next_state, done):
            from core.replay import Transition
            self.replay.append(Transition(state, action, reward, next_state, done))
    agent = RecordingAgent()
    env = CreatureEnvironment(config(max_steps=3))
    result = run_episode(env, agent, seed=4)
    assert result.truncated and result.steps == 3
    assert len(agent.replay) == 3
    assert sum(t.done for t in agent.replay.sample(3)) == 1


def test_runner_replay_includes_terminal_transitions():
    replay = ReplayBuffer(100)
    rows = run_population_episode(config(max_steps=2), [Member(i, Genome()) for i in range(2)], 42,
                                  replay=replay)
    assert len(replay) == 4 and len(rows) == 2
    assert sum(t.done for t in replay.sample(4)) == 2


def test_heuristic_uses_food_and_hazard_genes():
    env = world()
    env.creatures[0].position, env.creatures[1].position = (3, 3), (7, 7)
    env.food = {(4, 3)}
    env.hazards = {(3, 2)}
    agent = HeuristicAgent(Genome(movement_tendency=1, exploration_tendency=0), seed=42)
    assert agent.act(env.observations()[0], explore=False) == Action.RIGHT


def test_repeated_seed_aggregation():
    exp = experiment()
    scores, summary, rows = evaluate_population(exp.members, replace(exp.config.world, reproduction_enabled=False),
                                               (11, 22), aggregation='median')
    for m in exp.members:
        values = [r['fitness'] for r in rows if r['id'] == m.id]
        assert scores[m.id] == pytest.approx(sum(values)/2)
    assert summary['fitness_aggregation'] == 'median'


def test_headless_import_has_no_torch_or_pygame():
    code = "import experiments.population_runner; import sys; assert 'torch' not in sys.modules; assert 'pygame' not in sys.modules"
    subprocess.run([sys.executable, '-c', code], check=True)


def test_neural_adapter_batches_existing_qnetwork_and_records_replay():
    from brains.dqn import NeuralAgentAdapter
    from neuro_genesis_sim import QNetwork
    env = world()
    states = list(env.observations().values())
    net = QNetwork(len(states[0].vector()), action_dim=5)
    agent = NeuralAgentAdapter(net)
    actions = agent.act_batch(states, explore=False)
    assert len(actions) == 2 and all(a in Action for a in actions)
    result = env.step(dict(enumerate(actions)))
    t = result.transitions[0]
    agent.observe(t.state, t.action, t.reward, t.next_state, t.done)
    assert agent.replay.sample(1)[0] == t


def test_passive_renderer_does_not_advance_or_draw_random_numbers(tmp_path):
    from worlds.creature.rendering import render_svg
    env = world()
    before = reproducible_state(env)
    path = tmp_path/'world.svg'
    render_svg(env, path)
    assert reproducible_state(env) == before and env.step_count == 0
    assert '<svg' in path.read_text() and 'generation' in path.read_text()


def test_resource_regrowth_occurs_once_not_per_agent():
    env = world(food_count=10, food_regrowth=2)
    env.food.clear()
    env.step(stay(env))
    assert len(env.food) == 2
    assert not env.food & {c.position for c in env.alive.values()}


def test_population_200_steps_preserve_unique_occupancy():
    env = MultiCreatureEnvironment(WorldConfig(population=200, max_population=200, max_steps=5))
    env.reset(seed=42)
    rng = random.Random(9)
    for _ in range(5):
        env.step({i: rng.choice(list(Action)) for i in env.alive})
        occupied = [c.position for c in env.alive.values()]
        assert len(occupied) == len(set(occupied))
        assert all(0 <= c.energy <= env.config.max_energy for c in env.alive.values())


@pytest.mark.parametrize('field, value', [('next_id', 0), ('generation', 99)])
def test_checkpoint_rejects_inconsistent_counters(tmp_path, field, value):
    exp = experiment()
    path = tmp_path/'checkpoint.json'
    exp.save(path)
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        EvolutionExperiment.load(path)


def test_selection_rejects_nonfinite_fitness():
    with pytest.raises(ValueError):
        tournament([Member(0, Genome())], {0: float('nan')}, random.Random(0))
