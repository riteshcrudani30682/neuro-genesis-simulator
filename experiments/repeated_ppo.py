"""Independent PPO training seeds with paired, frozen heldout evaluation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import random
from statistics import mean, median, pstdev

from experiments.population_runner import append_rows, write_json


def paired_summary(differences, *, unit='training_seed_mean_over_heldout_worlds'):
    """Resample independent run/world means, never individual creatures."""
    if len(differences) < 2:
        raise ValueError('at least two independent training seeds are required')
    rng = random.Random(7301)
    samples = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(4000))
    return {'unit': unit, 'n': len(differences),
            'mean_delta': mean(differences), 'median_delta': median(differences),
            'std_delta': pstdev(differences), 'positive_runs': sum(x > 0 for x in differences),
            'per_unit_delta': differences,
            'bootstrap_95_percent_interval': [samples[100], samples[3899]]}


def aggregate(runs):
    policies = [r['policy'] for r in runs[0]['evaluation']]
    by_seed = [{r['policy']: r for r in run['evaluation']} for run in runs]
    summaries = []
    for policy in policies:
        values = [run[policy]['mean_fitness'] for run in by_seed]
        summaries.append({'policy': policy, 'mean_fitness': mean(values),
                          'median_fitness': median(values), 'training_seed_std': pstdev(values),
                          'per_training_seed_mean': values})
    comparisons = {}
    for a, b in [('ppo_trained', 'ppo_untrained'), ('ppo_trained', 'heuristic')]:
        differences = [run[a]['mean_fitness'] - run[b]['mean_fitness'] for run in by_seed]
        comparisons[f'{a}_minus_{b}'] = paired_summary(differences)
    # These baselines do not train: five copies must not pretend to be five new experiments.
    a, b = by_seed[0]['memory_strategy_heuristic'], by_seed[0]['strategy_no_memory']
    comparisons['memory_strategy_heuristic_minus_strategy_no_memory'] = paired_summary(
        [x-y for x, y in zip(a['seed_fitness'], b['seed_fitness'])], unit='heldout_world_seed')
    return {'policies': summaries, 'paired_comparisons': comparisons,
            'limitations': ['Small training-seed count; bootstrap intervals are descriptive.',
                           'Intervals condition on the fixed heldout worlds, not all possible worlds.',
                           'Heuristic and memory baselines are repeated controls, not independent training runs.',
                           'Fresh evaluation memories; no LLM; no reproduction; no heldout tuning.']}


def run_benchmark(*, training_seeds=(11, 22, 33, 44, 55), heldout_seeds=(101, 202, 303, 404, 505),
                  updates=20, world=None, ppo_config=None, device='cpu', output=Path('runs/repeated-ppo'), resume=False):
    import torch
    from brains.ppo import PPOConfig, PPOTrainer
    from experiments.ppo_runner import collect_episode, evaluate, validate_seed_split
    from worlds.creature.entities import WorldConfig
    training_seeds, heldout_seeds = list(training_seeds), list(heldout_seeds)
    if (len(training_seeds) < 2 or len(set(training_seeds)) != len(training_seeds)
            or any(type(s) is not int or s < 0 for s in training_seeds)):
        raise ValueError('use at least two distinct nonnegative training seeds')
    if (len(heldout_seeds) < 2 or len(set(heldout_seeds)) != len(heldout_seeds)
            or any(type(s) is not int or s <= 0 for s in heldout_seeds)):
        raise ValueError('use at least two distinct positive heldout seeds')
    if type(updates) is not int or not 1 <= updates < 1_000_003:
        raise ValueError('updates must be in 1..1000002')
    for seed in training_seeds:
        validate_seed_split(seed, 0, updates, heldout_seeds)
    world = world or WorldConfig(population=16, max_steps=128, reproduction_enabled=False)
    if world.reproduction_enabled:
        raise ValueError('PPO benchmark requires reproduction disabled')
    ppo_config = ppo_config or PPOConfig()
    torch.set_num_threads(2)
    config = {'training_seeds': training_seeds, 'heldout_seeds': heldout_seeds, 'updates': updates,
              'world': asdict(world), 'ppo': asdict(ppo_config), 'device': device, 'torch_threads': 2}
    output = Path(output)
    manifest = output/'config.json'
    if output.exists() and any(output.iterdir()):
        if not resume or not manifest.exists() or json.loads(manifest.read_text()) != config:
            raise ValueError('output exists: use --resume with exactly matching configuration or a new directory')
    write_json(manifest, config)
    runs = []
    for seed in training_seeds:
        directory = output/f'seed-{seed}'
        completed = directory/'result.json'
        if completed.exists():
            result = json.loads(completed.read_text())
        else:
            trainer = PPOTrainer(ppo_config, seed=seed, device=device)
            directory.mkdir(parents=True, exist_ok=True)
            # Incomplete seeds restart; completed seeds are reused. Never append duplicate updates.
            (directory/'training.jsonl').write_text('')
            for _ in range(updates):
                episode_seed = -(seed*1_000_003 + trainer.updates + 1)
                trajectories, rows = collect_episode(trainer, world, episode_seed)
                metrics = trainer.update(trajectories)
                metrics.update({'training_seed': seed, 'episode_seed': episode_seed,
                                'mean_fitness': mean(r['fitness'] for r in rows),
                                'mean_reward': mean(r['total_reward'] for r in rows)})
                append_rows(directory/'training.jsonl', [metrics])
                if trainer.updates % 5 == 0 or trainer.updates == updates:
                    print(f"Seed {seed}: update {trainer.updates}/{updates}, loss={metrics['loss']:.3f}", flush=True)
            trainer.save(directory/'checkpoint.pt', metadata={'world': asdict(world), 'seed': seed})
            result = {'training_seed': seed, 'updates': trainer.updates,
                      'evaluation': evaluate(trainer, world, seeds=heldout_seeds, initial_seed=seed)}
            write_json(completed, result)
        runs.append(result)
        score = {r['policy']: r['mean_fitness'] for r in result['evaluation']}
        print(f"Seed {seed}: PPO {score['ppo_untrained']:.3f} -> {score['ppo_trained']:.3f}; "
              f"heuristic {score['heuristic']:.3f}", flush=True)
    report = {'schema_version': 1, 'config': config, 'runtime': {'python': platform.python_version(),
              'platform': platform.platform(), 'torch': str(torch.__version__)},
              'runs': runs, **aggregate(runs)}
    write_json(output/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-seeds', nargs='+', type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument('--heldout-seeds', nargs='+', type=int, default=[101, 202, 303, 404, 505])
    parser.add_argument('--updates', type=int, default=20)
    parser.add_argument('--population', type=int, default=16)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path, default=Path('runs/repeated-ppo'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    from worlds.creature.entities import WorldConfig
    try:
        world = WorldConfig(population=args.population, max_population=max(200, args.population),
                            max_steps=args.steps, reproduction_enabled=False)
        run_benchmark(training_seeds=args.training_seeds, heldout_seeds=args.heldout_seeds,
                      updates=args.updates, world=world, device=args.device, output=args.output, resume=args.resume)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
