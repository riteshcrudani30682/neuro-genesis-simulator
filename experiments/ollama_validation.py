"""Live Ollama protocol + creature-loop validation on the machine running this command.

Uses only the Python standard library. Never downloads models or changes Ollama settings.
Fallback is recorded as failure, never as successful LLM validation.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
from statistics import mean
from urllib.request import urlopen

from brains.strategy import CallBudget, MemoryStrategyAgent, OllamaPlanner, StrategyController
from experiments.population_runner import write_json
from memory.episodic import ExperienceMemory
from worlds.creature.environment import MultiCreatureEnvironment
from worlds.creature.entities import WorldConfig


def get_json(endpoint, path, timeout=5):
    with urlopen(endpoint.rstrip('/') + path, timeout=timeout) as response:
        raw = response.read(1_048_577)
    if len(raw) > 1_048_576:
        raise ValueError('oversized Ollama discovery response')
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError('expected Ollama JSON object')
    return data


def select_model(models, name=None, interactive=False):
    names = [m['name'] for m in models if isinstance(m, dict) and isinstance(m.get('name'), str)]
    if name:
        if name not in names:
            raise ValueError('requested model is not installed; use an exact name from the printed list')
        return name
    if not names:
        raise ValueError('no installed Ollama models found')
    for index, model in enumerate(names, 1):
        print(f'  {index}. {model}', flush=True)
    if not interactive:
        raise ValueError('choose an installed model with --model NAME or --interactive')
    index = int(input('Choose model number (no model will be downloaded): ')) - 1
    if not 0 <= index < len(names):
        raise ValueError('invalid model selection')
    return names[index]


def run_validation(*, model=None, endpoint='http://127.0.0.1:11434', timeout=60,
                   requests=3, output=Path('runs/ollama-validation'), interactive=False):
    if type(requests) is not int or not 1 <= requests <= 10:
        raise ValueError('requests must be in 1..10')
    # Validate endpoint/timeout before discovery, using the same rules as the production adapter.
    OllamaPlanner(model or 'discovery', endpoint=endpoint, timeout=timeout)
    output = Path(output)
    report = {'schema_version': 1, 'status': 'blocked', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'runtime': {'system': platform.system(), 'release': platform.release(),
                          'python': platform.python_version(), 'cpu_count': os.cpu_count()},
              'model': model, 'request_budget': requests, 'timeout_seconds': timeout,
              'checks': {}, 'events': [],
              'scope': 'Live protocol and integration check, not an LLM performance benchmark.'}
    stage = 'discovery'
    try:
        report['ollama_version'] = get_json(endpoint, '/api/version').get('version')
        models = get_json(endpoint, '/api/tags')['models']
        if not isinstance(models, list):
            raise ValueError('invalid model list')
        report['available_models'] = [m.get('name') for m in models if isinstance(m, dict)]
        model = select_model(models, model, interactive)
        selected = next(m for m in models if isinstance(m, dict) and m.get('name') == model)
        report.update({'model': model, 'model_digest': selected.get('digest'),
                       'model_size_bytes': selected.get('size')})
        report['checks']['discovery'] = True
        planner = OllamaPlanner(model, endpoint=endpoint, timeout=timeout)
        stage = 'simulation'
        # A benign diagnostic world guarantees the creature lives long enough to check the gate.
        config = WorldConfig(population=1, max_population=1, max_steps=requests*20,
                             initial_energy=100, basal_cost=.1, movement_cost=.2,
                             hazard_count=0, reproduction_enabled=False)
        env = MultiCreatureEnvironment(config)
        observations = env.reset(seed=881)
        memory = ExperienceMemory('ollama-live-validation:0')
        budget = CallBudget(requests)
        agent = MemoryStrategyAgent(memory, seed=881, planner=planner, budget=budget)
        agent.strategy = StrategyController(planner, budget=budget, min_interval=20, max_interval=20)
        for tick in range(config.max_steps):
            state = observations[0]
            before = len(agent.strategy.events)
            action = agent.act(state)
            if len(agent.strategy.events) != before:
                event = agent.strategy.events[-1]
                print(f"Request {budget.used}/{requests}: {event['source']} goal={event['goal']} "
                      f"latency={event['latency_s']:.2f}s error={event['error']}", flush=True)
            result = env.step({0: action})
            t = result.transitions[0]
            agent.observe(t.state, t.action, t.reward, t.next_state, t.done)
            observations = result.observations
            if result.done:
                break
        report['events'] = agent.strategy.events
        memory.save(output/'memory.json')
        restored = ExperienceMemory.load(output/'memory.json', owner=memory.owner)
        events = report['events']
        report['checks'].update({
            'all_requests_accepted_as_llm': len(events) == requests and all(e['source'] == 'llm' for e in events),
            'bounded_calls': budget.used == requests,
            'cooldown': [e['tick'] for e in events] == list(range(0, requests*20, 20)),
            'world_transitions': memory.transitions == config.max_steps,
            'memory_round_trip': restored.to_dict() == memory.to_dict(),
        })
        report['actual_requests'] = budget.used
        report['world_steps'] = memory.transitions
        report['latency_seconds'] = {'first_request': events[0]['latency_s'],
                                     'mean': mean(e['latency_s'] for e in events),
                                     'max': max(e['latency_s'] for e in events)}
        report['default_demo_timeout_compatible'] = all(e['latency_s'] < 10 for e in events)
        report['token_totals'] = {key: sum(e['usage'].get(key, 0) for e in events)
                                  for key in ('prompt_eval_count', 'eval_count')}
        report['status'] = 'passed' if all(report['checks'].values()) else 'failed'
        # Ollama reports the loaded model's VRAM allocation; no Torch/CUDA import is needed.
        try:
            processes = get_json(endpoint, '/api/ps').get('models', [])
            report['loaded_model'] = [{key: p.get(key) for key in ('name', 'size', 'size_vram', 'context_length')}
                                      for p in processes if p.get('name') == model or p.get('model') == model]
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            report['gpu_diagnostic_error'] = type(exc).__name__
    except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError) as exc:
        report['status'] = 'blocked' if stage == 'discovery' else 'failed'
        report['error'] = {'stage': stage, 'type': type(exc).__name__}
        report['next_action'] = ('Open Ollama on this PC and select an installed chat model. '
                                 'Use --model with its exact name. No model is downloaded by this validator.')
    write_json(output/'report.json', report)
    print(f"Validation {report['status'].upper()}: {output/'report.json'}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--endpoint', default='http://127.0.0.1:11434')
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--requests', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    output = args.output or Path('runs')/('ollama-validation-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    try:
        report = run_validation(model=args.model, endpoint=args.endpoint, timeout=args.timeout,
                                requests=args.requests, output=output, interactive=args.interactive)
    except ValueError as exc:
        parser.error(str(exc))
    if report['status'] != 'passed':
        print(report.get('next_action', 'Check events for errors, confidence fallback, or invalid JSON.'))
    elif not report['default_demo_timeout_compatible']:
        print('Some calls took over 10s. Use --llm-timeout 60 in strategy_runner; the viewer pauses during calls.')
    return {'passed': 0, 'failed': 1, 'blocked': 2}[report['status']]


if __name__ == '__main__':
    raise SystemExit(main())
