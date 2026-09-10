"""Persistent-memory population demo with opt-in local LLM and optional PPO/viewer."""
import argparse
import hashlib
import json
from pathlib import Path

from brains.strategy import CallBudget, MemoryStrategyAgent, OllamaPlanner
from evolution.genome import Genome
from evolution.population import Member
from memory.episodic import ExperienceMemory
from worlds.creature.entities import WorldConfig
from experiments.population_runner import run_population_episode, append_rows, write_json


def run_session(config, *, episodes=2, seed=42, output=Path('runs/strategy'), namespace='creature-strategy-v1',
                planner=None, max_calls=10, trainer=None, render=False):
    if type(episodes) is not int or episodes < 1:
        raise ValueError('episodes must be positive')
    output = Path(output)
    manifest = output/'session.json'
    session = json.loads(manifest.read_text())['session'] + 1 if manifest.exists() else 0
    write_json(manifest, {'session': session})
    budget = CallBudget(max_calls)
    results = []
    for episode in range(episodes):
        agents, paths = {}, {}
        view = None
        def factory(creature, policy_seed):
            # Founders intentionally retain experiences. Newborn IDs cannot alias between episodes.
            identity = (f'founder:{creature.id}' if creature.parent_id is None
                        else f'session:{session}:episode:{seed+episode}:child:{creature.id}')
            owner = f'{namespace}:{identity}'
            path = output/'memory'/(hashlib.sha256(owner.encode()).hexdigest()+'.json')
            memory = ExperienceMemory.load(path, owner=owner) if path.exists() else ExperienceMemory(owner)
            low = None
            if trainer is not None:
                from brains.ppo import PPOPolicy
                low = PPOPolicy(trainer, policy_seed)
            agent = MemoryStrategyAgent(memory, seed=policy_seed, genome=creature.genome,
                                        planner=planner, budget=budget, low_level=low)
            agents[creature.id], paths[creature.id] = agent, path
            return agent
        try:
            if render:
                from ui.creature_view import CreatureView
                view = CreatureView(config, agents, budget)
            members = [Member(i, Genome()) for i in range(config.population)]
            rows = run_population_episode(config, members, seed+episode, policy_factory=factory, frame_callback=view)
            results.extend({**r, 'episode_seed': seed+episode} for r in rows)
            append_rows(output/'creatures.jsonl', [{**r, 'episode_seed': seed+episode} for r in rows])
        except KeyboardInterrupt:
            for agent in agents.values():
                agent.finish_episode(interrupted=True)
            break
        finally:
            for i, agent in agents.items():
                agent.memory.save(paths[i])
                append_rows(output/'strategy.jsonl', [{**e, 'owner': agent.memory.owner,
                                                      'episode_seed': seed+episode} for e in agent.strategy.events])
            if view:
                view.close()
        print(f"Episode {episode+1}: lives={len(rows)} LLM calls={budget.used} "
              f"persistent memories={len(agents)}", flush=True)
    summary = {'session': session, 'episodes_requested': episodes, 'life_records': len(results), 'llm_calls': budget.used,
               'llm_budget': budget.maximum, 'planner': 'ollama' if isinstance(planner, OllamaPlanner) else 'rules',
               'policy': 'ppo' if trainer else 'memory_strategy_heuristic'}
    write_json(output/'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--population', type=int, default=12)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('runs/strategy'))
    parser.add_argument('--namespace', default='creature-strategy-v1')
    parser.add_argument('--planner', choices=('rules', 'ollama'), default='rules')
    parser.add_argument('--model', help='Explicit installed Ollama model name; no downloads are triggered')
    parser.add_argument('--endpoint', default='http://127.0.0.1:11434')
    parser.add_argument('--max-calls', type=int, default=10)
    parser.add_argument('--ppo-checkpoint', type=Path)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args(argv)
    if args.planner == 'ollama' and not args.model:
        parser.error('--model is required with --planner ollama')
    planner = OllamaPlanner(args.model, endpoint=args.endpoint) if args.planner == 'ollama' else None
    trainer = None
    if args.ppo_checkpoint:
        from brains.ppo import PPOTrainer
        trainer, _ = PPOTrainer.load(args.ppo_checkpoint)
        if trainer.config.input_dim != 111:
            parser.error('checkpoint observation radius does not match this demo')
    config = WorldConfig(population=args.population, max_steps=args.steps)
    run_session(config, episodes=args.episodes, seed=args.seed, output=args.output, namespace=args.namespace,
                planner=planner, max_calls=args.max_calls, trainer=trainer, render=args.render)


if __name__ == '__main__':
    main()
