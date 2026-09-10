"""Persistent-memory population demo with opt-in local LLM and optional PPO/viewer."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import uuid

from brains.strategy import CallBudget, MemoryStrategyAgent, OllamaPlanner, RulePlanner, PROMPT_VERSION
from brains.scheduling import FairCallScheduler
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
    run_id = uuid.uuid4().hex
    write_json(manifest, {'session': session, 'run_id': run_id})
    budget = CallBudget(max_calls)
    served = {}
    run_directory = output/'sessions'/run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    latest_log = output/'latest_strategy.jsonl'
    latest_log.write_text('', encoding='utf-8')
    episode_summaries = []
    results = []
    for episode in range(episodes):
        agents, paths = {}, {}
        view = None
        quota = max_calls // episodes + int(episode < max_calls % episodes)
        scheduler = FairCallScheduler(budget, quota=quota, horizon=config.max_steps,
                                      seed=seed, served=served)
        counts = Counter()
        usage = Counter()
        pending_events = []
        interrupted = False
        rows = []
        metadata = {'run_id': run_id, 'session': session, 'episode': episode+1, 'episode_seed': seed+episode}
        def flush_events():
            if pending_events:
                for path in (output/'strategy.jsonl', latest_log, run_directory/'strategy.jsonl'):
                    append_rows(path, pending_events)
                pending_events.clear()
        def record(event, owner):
            counts[event['source']] += 1
            if event['request_attempted']:
                counts['attempts'] += 1
                usage.update(event['usage'])
            pending_events.append({**event, **metadata, 'owner': owner})
            if len(pending_events) >= 256:
                flush_events()
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
            agent.strategy.event_sink = lambda event: record(event, owner)
            agents[creature.id], paths[creature.id] = agent, path
            return agent
        try:
            if render:
                from ui.creature_view import CreatureView
                view = CreatureView(config, agents, budget)
            members = [Member(i, Genome()) for i in range(config.population)]
            rows = run_population_episode(config, members, seed+episode, policy_factory=factory, frame_callback=view,
                                          before_actions=scheduler.prepare if planner is not None and not isinstance(planner, RulePlanner) else None)
            results.extend({**r, **metadata} for r in rows)
            append_rows(output/'creatures.jsonl', [{**r, **metadata} for r in rows])
        except KeyboardInterrupt:
            interrupted = True
            for agent in agents.values():
                agent.finish_episode(interrupted=True)
        finally:
            flush_events()
            for i, agent in agents.items():
                agent.memory.save(paths[i])
            if view:
                view.close()
        counts.setdefault('attempts', 0)
        counts.setdefault('llm', 0)
        episode_summaries.append({**metadata, 'interrupted': interrupted, 'llm_quota': quota,
                                  'call_gap_world_ticks': scheduler.gap, 'sources': dict(counts),
                                  'known_token_totals': dict(usage)})
        print(f"Episode {episode+1}: lives={len(rows)} LLM attempts={counts['attempts']}/{quota} "
              f"accepted={counts['llm']} session calls={budget.used}/{budget.maximum} "
              f"persistent memories={len(agents)}", flush=True)
        if interrupted:
            break
    summary = {'session': session, 'run_id': run_id, 'episodes_requested': episodes,
               'episodes': episode_summaries, 'life_records': len(results), 'llm_calls': budget.used,
               'llm_budget': budget.maximum, 'planner': 'ollama' if isinstance(planner, OllamaPlanner) else 'rules',
               'policy': 'ppo' if trainer else 'memory_strategy_heuristic', 'prompt_version': PROMPT_VERSION,
               'model': getattr(planner, 'model', None), 'world_config': vars(config),
               'calls_by_owner': served, 'latest_strategy_log': str(latest_log), 'run_directory': str(run_directory)}
    if planner is not None and not isinstance(planner, (RulePlanner, OllamaPlanner)):
        summary['planner'] = 'custom'
    write_json(output/'summary.json', summary)
    write_json(run_directory/'summary.json', summary)
    print(f'Current run log: {latest_log}', flush=True)
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
    parser.add_argument('--llm-timeout', type=float, default=10, help='Per-call timeout in seconds, maximum 60')
    parser.add_argument('--max-calls', type=int, default=10)
    parser.add_argument('--ppo-checkpoint', type=Path)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args(argv)
    if args.planner == 'ollama' and not args.model:
        parser.error('--model is required with --planner ollama')
    planner = OllamaPlanner(args.model, endpoint=args.endpoint, timeout=args.llm_timeout) if args.planner == 'ollama' else None
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
