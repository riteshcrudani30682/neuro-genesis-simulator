"""Train shared goal-conditioned PPO on real multi-agent trajectories; no LLM in training."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, pstdev

from brains.ppo import PPOConfig, PPOTrainer, PPOPolicy
from brains.strategy import MemoryStrategyAgent
from evolution.genome import Genome
from evolution.population import Member
from memory.episodic import ExperienceMemory
from worlds.creature.environment import MultiCreatureEnvironment
from worlds.creature.entities import WorldConfig
from experiments.population_runner import append_rows, run_population_episode, write_json


def collect_episode(trainer, config, seed):
    """One shared world, one batched policy call per tick, separate GAE per creature."""
    config = replace(config, reproduction_enabled=False)
    env = MultiCreatureEnvironment(config)
    observations = env.reset(seed=seed)
    agents = {i: MemoryStrategyAgent(ExperienceMemory(f'train:{i}'), seed=seed+i) for i in observations}
    trajectories = {i: [] for i in observations}
    features = {i: agents[i].features(s) for i, s in observations.items()}
    for _ in range(config.max_steps):
        ids = sorted(observations)
        action_list, log_probs, values = trainer.sample([features[i] for i in ids])
        actions = dict(zip(ids, action_list))
        result = env.step(actions)
        next_features = {}
        for i in ids:
            t = result.transitions[i]
            agents[i].observe(t.state, t.action, t.reward, t.next_state, t.done)
            next_features[i] = agents[i].features(t.next_state)
        next_values = trainer.values([next_features[i] for i in ids])
        for index, i in enumerate(ids):
            t = result.transitions[i]
            trajectories[i].append({'policy_version': trainer.updates, 'features': features[i], 'action': actions[i], 'log_prob': log_probs[index],
                                    'reward': t.reward, 'value': values[index], 'next_value': next_values[index],
                                    'terminated': result.terminated[i], 'truncated': result.truncated})
        observations = result.observations
        features = {i: next_features[i] for i in observations}
        if result.done:
            break
    return list(trajectories.values()), env.creature_metrics()


def evaluate(trainer, config, seeds=(101, 202, 303), *, initial_seed=42):
    """Frozen policies with fresh memory per episode on disjoint heldout worlds."""
    world = replace(config, reproduction_enabled=False)
    untrained = PPOTrainer(trainer.config, seed=initial_seed, device=str(trainer.device))
    members = [Member(i, Genome()) for i in range(world.population)]
    report = []
    for label, model in [('random', None), ('heuristic', None), ('strategy_no_memory', None), ('memory_strategy_heuristic', None),
                         ('ppo_untrained', untrained), ('ppo_trained', trainer)]:
        scores, rewards, survivals = [], [], []
        for seed in seeds:
            factory = None
            if label not in ('random', 'heuristic'):
                def factory(c, policy_seed):
                    low = PPOPolicy(model, policy_seed) if model is not None else None
                    return MemoryStrategyAgent(ExperienceMemory(f'eval:{c.id}', enabled=label != 'strategy_no_memory'), seed=policy_seed,
                                               genome=c.genome, low_level=low)
            rows = run_population_episode(world, members, seed, policy='random' if label == 'random' else 'heuristic',
                                          policy_factory=factory)
            scores.append(mean(r['fitness'] for r in rows))
            rewards.append(mean(r['total_reward'] for r in rows))
            survivals.append(mean(float(r['alive']) for r in rows))
        report.append({'policy': label, 'seeds': list(seeds), 'seed_fitness': scores,
                       'mean_fitness': mean(scores), 'fitness_seed_std': pstdev(scores),
                       'mean_reward': mean(rewards), 'survival_rate': mean(survivals)})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updates', type=int, default=20)
    parser.add_argument('--population', type=int, default=16)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path, default=Path('runs/ppo'))
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args(argv)
    if args.updates < 1:
        parser.error('--updates must be positive')
    import torch
    torch.set_num_threads(2)
    if args.resume:
        trainer, metadata = PPOTrainer.load(args.resume, device=args.device)
        world = WorldConfig(**metadata['world'])
        seed = metadata['seed']
    else:
        world = WorldConfig(population=args.population, max_population=max(args.population, 200),
                            max_steps=args.steps, reproduction_enabled=False)
        trainer = PPOTrainer(seed=args.seed, device=args.device)
        seed = args.seed
    for _ in range(args.updates):
        # Negative training seeds cannot overlap default positive heldout seeds.
        episode_seed = -(abs(seed)*1_000_003 + trainer.updates + 1)
        trajectories, rows = collect_episode(trainer, world, episode_seed)
        metrics = trainer.update(trajectories)
        metrics.update({'episode_seed': episode_seed, 'mean_reward': mean(r['total_reward'] for r in rows),
                        'mean_fitness': mean(r['fitness'] for r in rows),
                        'survival_rate': mean(float(r['alive']) for r in rows)})
        append_rows(args.output/'training.jsonl', [metrics])
        trainer.save(args.output/'checkpoint.pt', metadata={'world': asdict(world), 'seed': seed})
        print(f"PPO update {trainer.updates:3d} | reward={metrics['mean_reward']:.3f} "
              f"loss={metrics['loss']:.3f} KL={metrics['approx_kl']:.5f} samples={metrics['transitions']}", flush=True)
    report = evaluate(trainer, world, initial_seed=seed)
    write_json(args.output/'evaluation.json', report)
    for row in report:
        print(f"Heldout {row['policy']}: fitness={row['mean_fitness']:.3f} reward={row['mean_reward']:.3f}")


if __name__ == '__main__':
    main()
