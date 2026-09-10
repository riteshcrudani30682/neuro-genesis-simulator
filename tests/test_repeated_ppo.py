import json

import pytest

from brains.ppo import PPOConfig
from experiments.repeated_ppo import paired_summary, run_benchmark
from worlds.creature.entities import WorldConfig


def small_run(output, **kwargs):
    return run_benchmark(training_seeds=(11, 22), heldout_seeds=(101, 202), updates=1,
                         world=WorldConfig(width=8, height=8, population=2, max_population=2,
                                           food_count=5, hazard_count=2, max_steps=3,
                                           reproduction_enabled=False),
                         ppo_config=PPOConfig(hidden=8, epochs=1, minibatch=8), output=output, **kwargs)


def test_paired_intervals_use_run_means_and_are_reproducible():
    stats = paired_summary([1, 2, 3, 4, 5])
    assert stats == paired_summary([1, 2, 3, 4, 5])
    assert stats['n'] == 5 and stats['mean_delta'] == 3
    assert stats['positive_runs'] == 5
    lo, hi = stats['bootstrap_95_percent_interval']
    assert 1 <= lo < 3 < hi <= 5
    with pytest.raises(ValueError):
        paired_summary([1])


def test_repeated_training_deterministic_and_resume_preserves_completed_runs(tmp_path):
    a = small_run(tmp_path/'a')
    b = small_run(tmp_path/'b')
    assert a == b
    assert [r['training_seed'] for r in a['runs']] == [11, 22]
    for seed in (11, 22):
        rows = [json.loads(line) for line in (tmp_path/'a'/f'seed-{seed}'/'training.jsonl').read_text().splitlines()]
        assert len(rows) == 1 and rows[0]['episode_seed'] < 0
        assert rows[0]['episode_seed'] not in a['config']['heldout_seeds']
    memory = a['paired_comparisons']['memory_strategy_heuristic_minus_strategy_no_memory']
    assert memory['unit'] == 'heldout_world_seed' and memory['n'] == 2
    completed = tmp_path/'a'/'seed-11'/'checkpoint.pt'
    timestamp = completed.stat().st_mtime_ns
    assert small_run(tmp_path/'a', resume=True) == a
    assert completed.stat().st_mtime_ns == timestamp
    with pytest.raises(ValueError, match='output exists'):
        small_run(tmp_path/'a')
    manifest = tmp_path/'a'/'config.json'
    config = json.loads(manifest.read_text())
    config['updates'] = 99
    manifest.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='matching configuration'):
        small_run(tmp_path/'a', resume=True)


@pytest.mark.parametrize('kwargs', [
    {'training_seeds': (1, 1)}, {'training_seeds': (-1, 1)},
    {'heldout_seeds': (101, 101)}, {'heldout_seeds': (-1, 101)}, {'updates': 0},
    {'training_seeds': (0, 1), 'updates': 101},
])
def test_seed_protocol_rejects_invalid_or_overlapping_worlds(tmp_path, kwargs):
    with pytest.raises(ValueError):
        run_benchmark(output=tmp_path, **kwargs)


def test_python_seed_sign_alias_is_not_a_heldout_split():
    import random
    from experiments.ppo_runner import validate_seed_split
    assert random.Random(-101).random() == random.Random(101).random()
    with pytest.raises(ValueError, match='overlap'):
        validate_seed_split(0, 100, 1, (101, 202))
    validate_seed_split(11, 0, 20, (101, 202, 303, 404, 505))
