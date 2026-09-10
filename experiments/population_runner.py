"""Headless continuous-life and repeated-seed generational experiments."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
import random
from statistics import mean, median
import tempfile

from brains.baseline import HeuristicAgent, RandomAgent
from evolution.fitness import genetic_diversity
from evolution.genome import Genome
from evolution.population import Member, breed
from worlds.creature.environment import MultiCreatureEnvironment
from worlds.creature.entities import WorldConfig


POLICIES = {'random': RandomAgent, 'heuristic': HeuristicAgent}


@dataclass(frozen=True)
class ExperimentConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    seed: int = 42
    evaluation_seeds: tuple = (11, 22, 33)
    heldout_seeds: tuple = (101, 202, 303)
    elitism: int = 2
    tournament_size: int = 3
    policy: str = 'heuristic'
    aggregation: str = 'mean'

    def __post_init__(self):
        if type(self.seed) is not int or not self.evaluation_seeds or not self.heldout_seeds:
            raise ValueError('integer seed and nonempty evaluation/heldout seeds required')
        if any(type(s) is not int for s in (*self.evaluation_seeds, *self.heldout_seeds)):
            raise ValueError('evaluation seeds must be integers')
        if len(set(self.evaluation_seeds)) != len(self.evaluation_seeds) or len(set(self.heldout_seeds)) != len(self.heldout_seeds):
            raise ValueError('repeated seeds must be distinct')
        if set(self.evaluation_seeds) & set(self.heldout_seeds):
            raise ValueError('selection and heldout seeds must be disjoint')
        if not 0 <= self.elitism <= self.world.population or self.tournament_size < 1:
            raise ValueError('invalid selection configuration')
        if self.policy not in POLICIES or self.aggregation not in ('mean', 'median'):
            raise ValueError('unknown policy or fitness aggregation')


def run_population_episode(config, members, seed, *, policy='heuristic', policy_factory=None,
                           replay=None, frame_callback=None):
    """Policies see the same pre-step world. Training hooks receive true transitions.

    A custom factory receives (Creature, policy_seed) and returns a core.Agent.
    Dead policies are discarded; newborns receive independent policy instances.
    """
    env = MultiCreatureEnvironment(config)
    observations = env.reset(seed=seed, genomes=[m.genome for m in members],
                             founders=[m.founder() for m in members], policy_name=policy)
    policies = {}
    policy_slots = {m.id: index for index, m in enumerate(members)}
    next_slot = len(members)
    factory = policy_factory or (lambda c, s: POLICIES[policy](c.genome, s))
    for _ in range(config.max_steps):
        for i in sorted(observations):
            if i not in policies:
                if i not in policy_slots:
                    policy_slots[i] = next_slot
                    next_slot += 1
                # Reset episode seeds per slot, not lineage ID, for fair generation comparisons.
                policies[i] = factory(env.creatures[i], seed * 1_000_003 + policy_slots[i])
        actions = {i: policies[i].act(observations[i], explore=True) for i in sorted(observations)}
        result = env.step(actions)
        for i, t in result.transitions.items():
            policies[i].observe(t.state, t.action, t.reward, t.next_state, t.done)
            if replay is not None:
                replay.append(t)
            if result.terminated[i]:
                del policies[i]
        if frame_callback:
            frame_callback(env, result)
        observations = result.observations
        if result.done:
            break
    return env.creature_metrics()


def summarize(rows, members, scores):
    spent = sum(c['energy_spent'] for c in rows)
    return {
        'generation': members[0].generation, 'population_size': len(members),
        'survival_rate': mean(float(c['alive']) for c in rows),
        'mean_fitness': mean(scores.values()), 'best_fitness': max(scores.values()),
        'median_fitness': median(scores.values()),
        'food_efficiency': sum(c['food_eaten'] for c in rows) / max(1.0, spent),
        'mean_lifetime': mean(c['lifetime'] for c in rows),
        'offspring_count': sum(c['offspring_count'] for c in rows),
        'genetic_diversity': genetic_diversity(m.genome for m in members),
    }


def evaluate_population(members, config, seeds, *, policy='heuristic', aggregation='mean', frame_callback=None):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('evaluation needs distinct, nonempty seeds')
    if aggregation not in ('mean', 'median'):
        raise ValueError('unknown fitness aggregation')
    rows = []
    for seed in seeds:
        rows.extend({**row, 'seed': seed} for row in run_population_episode(
            config, members, seed, policy=policy, frame_callback=frame_callback))
    # Continuous descendants may be present in rows; selection scores belong to founders only.
    aggregate = mean if aggregation == 'mean' else median
    scores = {m.id: aggregate(r['fitness'] for r in rows if r['id'] == m.id) for m in members}
    summary = summarize(rows, members, scores)
    per_seed = [mean(r['fitness'] for r in rows if r['seed'] == s) for s in seeds]
    summary.update({'evaluation_seeds': list(seeds), 'seed_mean_fitness': per_seed,
                    'final_population_mean': sum(r['alive'] for r in rows) / len(seeds),
                    'founder_survival_rate': mean(float(r['alive']) for r in rows if r['id'] in scores),
                    'max_biological_generation': max(r['generation'] for r in rows),
                    'fitness_aggregation': aggregation,
                    'fitness_seed_std': (mean((v-mean(per_seed))**2 for v in per_seed))**0.5})
    return scores, summary, rows


def write_json(path, data):
    """Atomic explicit JSON persistence. A partial write cannot replace a checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as handle:
            temp = Path(handle.name)
            json.dump(data, handle, allow_nan=False, indent=2)
        temp.replace(path)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def append_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + '\n')


class EvolutionExperiment:
    """Checkpoint boundary is between generations, after breeding the next population."""
    def __init__(self, config=None):
        self.config = config or ExperimentConfig()
        self.rng = random.Random(self.config.seed)
        self.members = [Member(i, Genome.random(self.rng)) for i in range(self.config.world.population)]
        self.initial_members = list(self.members)
        self.generation = 0
        self.next_id = len(self.members)
        self.metrics = []

    def advance(self, output=None):
        world = replace(self.config.world, reproduction_enabled=False)
        scores, summary, rows = evaluate_population(self.members, world, self.config.evaluation_seeds,
                                                    policy=self.config.policy, aggregation=self.config.aggregation)
        if output:
            append_rows(Path(output) / 'creatures.jsonl', rows)
            append_rows(Path(output) / 'generations.jsonl', [summary])
        self.members = breed(self.members, scores, self.rng, next_id=self.next_id,
                             elitism=self.config.elitism, mutation_rate=world.mutation_rate,
                             tournament_size=self.config.tournament_size)
        self.next_id += len(self.members)
        self.generation += 1
        self.metrics.append(summary)
        return summary

    def compare_baselines(self, output=None):
        """Matched heldout worlds. Neither comparisons nor heldout seeds drive selection."""
        world = replace(self.config.world, reproduction_enabled=False)
        comparisons = []
        for name, members, policy in (
            ('evolved', self.members, self.config.policy),
            ('fixed_initial_heuristic', self.initial_members, 'heuristic'),
            ('random', self.initial_members, 'random'),
        ):
            _, summary, _ = evaluate_population(members, world, self.config.heldout_seeds,
                                                policy=policy, aggregation=self.config.aggregation)
            comparisons.append({'baseline': name, **summary})
        if output:
            write_json(Path(output) / 'heldout_comparison.json', comparisons)
        return comparisons

    def save(self, path):
        write_json(path, {'schema_version': 1, 'mode': 'generational', 'generation': self.generation,
                          'config': asdict(self.config), 'next_id': self.next_id,
                          'members': [m.to_dict() for m in self.members],
                          'initial_members': [m.to_dict() for m in self.initial_members],
                          'rng_state': self.rng.getstate(), 'metrics': self.metrics})

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('schema_version') != 1 or data.get('mode') != 'generational':
            raise ValueError('unsupported checkpoint schema or mode')
        config = dict(data['config'])
        config['world'] = WorldConfig(**config['world'])
        config['evaluation_seeds'] = tuple(config['evaluation_seeds'])
        config['heldout_seeds'] = tuple(config['heldout_seeds'])
        experiment = cls(ExperimentConfig(**config))
        experiment.members = [Member.from_dict(m) for m in data['members']]
        experiment.initial_members = [Member.from_dict(m) for m in data['initial_members']]
        experiment.generation = data['generation']
        experiment.next_id = data['next_id']
        experiment.metrics = data['metrics']
        ids = [m.id for m in experiment.members]
        if (len(ids) != experiment.config.world.population or len(ids) != len(set(ids))
                or not all(type(i) is int and i >= 0 for i in ids)
                or type(experiment.generation) is not int or experiment.generation < 0
                or any(m.generation != experiment.generation for m in experiment.members)
                or experiment.next_id <= max(ids) or len(experiment.metrics) != experiment.generation):
            raise ValueError('inconsistent checkpoint population/generation')
        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, list) else value
        experiment.rng.setstate(tuples(data['rng_state']))
        return experiment


def seed_list(value):
    try:
        return tuple(int(s) for s in value.split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('expected comma-separated integer seeds') from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['generational', 'continuous'], default='generational')
    parser.add_argument('--generations', type=int, default=20)
    parser.add_argument('--population', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--max-population', type=int, default=200)
    parser.add_argument('--policy', choices=POLICIES, default='heuristic')
    parser.add_argument('--evaluation-seeds', type=seed_list, default=(11, 22, 33))
    parser.add_argument('--heldout-seeds', type=seed_list, default=(101, 202, 303))
    parser.add_argument('--aggregation', choices=['mean', 'median'], default='mean')
    parser.add_argument('--elitism', type=int, default=2)
    parser.add_argument('--output', default='runs/population')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--compare-baselines', action='store_true')
    parser.add_argument('--snapshot', type=Path, help='Passive final SVG snapshot (continuous mode only)')
    args = parser.parse_args(argv)
    if args.snapshot and args.mode != 'continuous':
        parser.error('--snapshot is available in continuous mode only')
    if args.generations < 1:
        parser.error('--generations must be positive')
    if args.resume and args.mode != 'generational':
        parser.error('checkpoint resume is supported at generational boundaries only')
    try:
        if args.resume:
            experiment = EvolutionExperiment.load(args.resume)
        else:
            world = WorldConfig(population=args.population, max_population=args.max_population, max_steps=args.steps)
            experiment = EvolutionExperiment(ExperimentConfig(world, args.seed, args.evaluation_seeds,
                                                               args.heldout_seeds, args.elitism,
                                                               policy=args.policy, aggregation=args.aggregation))
        if args.mode == 'continuous':
            def capture(world, result):
                if args.snapshot and result.done:
                    from worlds.creature.rendering import render_svg
                    render_svg(world, args.snapshot)
            scores, summary, rows = evaluate_population(experiment.members, experiment.config.world,
                                                        experiment.config.evaluation_seeds,
                                                        policy=experiment.config.policy,
                                                        aggregation=experiment.config.aggregation, frame_callback=capture)
            append_rows(Path(args.output) / 'continuous_creatures.jsonl', rows)
            write_json(Path(args.output) / 'continuous_summary.json', summary)
            print(json.dumps({'mode': 'continuous', **summary}))
            return
        for _ in range(args.generations):
            summary = experiment.advance(args.output)
            experiment.save(Path(args.output) / 'checkpoint.json')
            print(f"Generation {summary['generation']:3d} | mean={summary['mean_fitness']:.3f} "
                  f"best={summary['best_fitness']:.3f} survival={summary['survival_rate']:.1%} "
                  f"diversity={summary['genetic_diversity']:.4f}", flush=True)
        if args.compare_baselines:
            for row in experiment.compare_baselines(args.output):
                print(f"Heldout {row['baseline']}: mean={row['mean_fitness']:.3f} "
                      f"seed_std={row['fitness_seed_std']:.3f}")
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
