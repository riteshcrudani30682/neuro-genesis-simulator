from dataclasses import replace
import math

import pytest
import torch

from brains.ppo import PPOConfig, PPOTrainer, PPOPolicy, gae
from brains.strategy import MemoryStrategyAgent
from memory.episodic import ExperienceMemory
from experiments.ppo_runner import collect_episode, evaluate
from worlds.creature.entities import WorldConfig


def config():
    return WorldConfig(width=8, height=8, population=3, max_population=8, food_count=8,
                       hazard_count=3, max_steps=6, reproduction_enabled=False)


def trainer(seed=42):
    torch.set_num_threads(1)
    return PPOTrainer(PPOConfig(hidden=16, epochs=2, minibatch=16), seed=seed)


def test_gae_terminal_does_not_bootstrap():
    adv, returns = gae([1], [.5], [100], [True], [False], gamma=.9)
    assert adv == [.5] and returns == [1]


def test_gae_truncation_bootstraps_but_cuts_trajectory():
    adv, returns = gae([1, 100], [.5, 0], [2, 0], [False, True], [True, False], gamma=.9, lam=1)
    assert adv[0] == pytest.approx(2.3)
    assert returns[0] == pytest.approx(2.8)


def test_gae_nonterminal_propagates_advantage():
    adv, _ = gae([1, 1], [0, 0], [0, 0], [False, True], [False, False], gamma=.9, lam=1)
    assert adv == pytest.approx([1.9, 1])


def test_seeded_collection_matches_and_trajectories_are_separate():
    a, b = trainer(), trainer()
    first, rows = collect_episode(a, config(), 11)
    second, _ = collect_episode(b, config(), 11)
    assert first == second
    assert len(first) == 3
    assert all(t[-1]['terminated'] or t[-1]['truncated'] for t in first)
    assert sum(len(t) for t in first) == sum(r['age'] for r in rows)
    assert all(row['policy_version'] == 0 for t in first for row in t)


def test_ppo_updates_weights_and_rejects_old_replay():
    agent = trainer()
    trajectories, _ = collect_episode(agent, config(), 11)
    before = [p.detach().clone() for p in agent.model.parameters()]
    result = agent.update(trajectories)
    assert result['update'] == 1 and result['minibatch_updates'] > 0
    assert math.isfinite(result['loss'])
    assert any(not torch.equal(p, old) for p, old in zip(agent.model.parameters(), before))
    with pytest.raises(ValueError, match='older policy'):
        agent.update(trajectories)


def test_checkpoint_resume_exact_next_update(tmp_path):
    a = trainer()
    data, _ = collect_episode(a, config(), 11)
    a.update(data)
    path = tmp_path/'ppo.pt'
    a.save(path, metadata={'seed': 42})
    b, metadata = PPOTrainer.load(path)
    assert metadata == {'seed': 42}
    x, _ = collect_episode(a, config(), 22)
    y, _ = collect_episode(b, config(), 22)
    assert x == y
    assert a.update(x) == b.update(y)
    assert all(torch.equal(v, b.model.state_dict()[k]) for k, v in a.model.state_dict().items())


def test_policy_adapter_uses_shared_model_with_independent_rng():
    t = trainer()
    policy = PPOPolicy(t, seed=4)
    agent = MemoryStrategyAgent(ExperienceMemory('a'), low_level=policy)
    from worlds.creature.environment import MultiCreatureEnvironment
    env = MultiCreatureEnvironment(config())
    state = env.reset(seed=4)[0]
    action = agent.act(state)
    assert 0 <= action <= 4 and agent.goal is not None
    assert t.updates == 0


def test_evaluation_does_not_modify_trained_weights_or_rng():
    t = trainer()
    before = {k: v.clone() for k, v in t.model.state_dict().items()}
    rng = t.generator.get_state().clone()
    report = evaluate(t, replace(config(), max_steps=2), seeds=(101, 202))
    assert len(report) == 6
    assert all(r['seeds'] == [101, 202] for r in report)
    assert torch.equal(rng, t.generator.get_state())
    assert all(torch.equal(v, before[k]) for k, v in t.model.state_dict().items())


def test_empty_or_nonfinite_rollout_rejected():
    t = trainer()
    with pytest.raises(ValueError):
        t.update([])
    data, _ = collect_episode(t, config(), 1)
    data[0][0]['reward'] = float('nan')
    with pytest.raises(ValueError):
        t.update(data)


def test_config_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        PPOConfig(gamma=1.1)
    with pytest.raises(ValueError):
        PPOConfig(clip_ratio=0)
