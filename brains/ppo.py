"""Clipped PPO with per-agent GAE, truncation bootstrapping and explicit checkpoints.

Torch is optional: importing the baseline runner never imports this module.
"""
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.distributions import Categorical

from worlds.creature.entities import Action


@dataclass(frozen=True)
class PPOConfig:
    input_dim: int = 111
    hidden: int = 64
    learning_rate: float = 3e-4
    gamma: float = .99
    gae_lambda: float = .95
    clip_ratio: float = .2
    entropy_coef: float = .01
    value_coef: float = .5
    max_grad_norm: float = .5
    epochs: int = 4
    minibatch: int = 256
    target_kl: float = .03

    def __post_init__(self):
        for name in ('input_dim', 'hidden', 'epochs', 'minibatch'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('learning_rate', 'gamma', 'gae_lambda', 'clip_ratio', 'entropy_coef',
                     'value_coef', 'max_grad_norm', 'target_kl'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.gamma > 1 or self.gae_lambda > 1 or not 0 < self.clip_ratio < 1:
            raise ValueError('invalid discount, GAE or clipping parameter')
        if min(self.learning_rate, self.max_grad_norm, self.target_kl) <= 0:
            raise ValueError('learning rate, gradient bound and KL threshold must be positive')


class ActorCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(config.input_dim, config.hidden), nn.Tanh(),
                                  nn.Linear(config.hidden, config.hidden), nn.Tanh())
        self.actor = nn.Linear(config.hidden, len(Action))
        self.critic = nn.Linear(config.hidden, 1)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, gain=.01)
        nn.init.orthogonal_(self.critic.weight, gain=1)

    def forward(self, features):
        hidden = self.body(features)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


def gae(rewards, values, next_values, terminated, truncated, gamma=.99, lam=.95):
    """One agent's contiguous trajectory. Never propagate advantage across reset/death."""
    length = len(rewards)
    if not length or any(len(x) != length for x in (values, next_values, terminated, truncated)):
        raise ValueError('GAE needs equally sized nonempty trajectory arrays')
    advantages = [0.0] * length
    carry = 0.0
    for t in reversed(range(length)):
        bootstrap = 0.0 if terminated[t] else next_values[t]
        delta = rewards[t] + gamma * bootstrap - values[t]
        continuation = not (terminated[t] or truncated[t])
        carry = delta + gamma * lam * carry * continuation
        advantages[t] = carry
    return advantages, [a + v for a, v in zip(advantages, values)]


class PPOTrainer:
    def __init__(self, config=None, *, seed=42, device='cpu'):
        self.config = config or PPOConfig()
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        # Initialization does not contaminate unrelated CPU Torch RNG users.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = ActorCritic(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, eps=1e-5)
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.updates = 0

    def sample(self, features, *, explore=True):
        values = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if values.ndim != 2 or values.shape[1] != self.config.input_dim:
            raise ValueError('PPO feature dimension mismatch')
        with torch.no_grad():
            logits, estimates = self.model(values)
            distribution = Categorical(logits=logits)
            actions = (torch.multinomial(distribution.probs.cpu(), 1, generator=self.generator).squeeze(1)
                       .to(self.device) if explore else logits.argmax(dim=1))
            log_probs = distribution.log_prob(actions)
        return actions.cpu().tolist(), log_probs.cpu().tolist(), estimates.cpu().tolist()

    def values(self, features):
        with torch.no_grad():
            return self.model(torch.tensor(features, dtype=torch.float32, device=self.device))[1].cpu().tolist()

    def update(self, trajectories):
        if any(row.get('policy_version') != self.updates for trajectory in trajectories for row in trajectory):
            raise ValueError('PPO rejects replay from an older policy version')
        features, actions, old_log_probs, advantages, returns = [], [], [], [], []
        for trajectory in trajectories:
            if not trajectory:
                continue
            adv, targets = gae(*([row[key] for row in trajectory] for key in
                                 ('reward', 'value', 'next_value', 'terminated', 'truncated')),
                               gamma=self.config.gamma, lam=self.config.gae_lambda)
            features.extend(row['features'] for row in trajectory)
            actions.extend(row['action'] for row in trajectory)
            old_log_probs.extend(row['log_prob'] for row in trajectory)
            advantages.extend(adv)
            returns.extend(targets)
        if not features:
            raise ValueError('PPO requires fresh on-policy trajectories')
        x = torch.tensor(features, dtype=torch.float32, device=self.device)
        act = torch.tensor(actions, dtype=torch.long, device=self.device)
        old = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        adv = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        target = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if not all(torch.isfinite(t).all() for t in (x, old, adv, target)):
            raise ValueError('non-finite rollout')
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        config = self.config
        losses, kls, entropies, clip_fractions = [], [], [], []
        stopped = False
        for _ in range(config.epochs):
            order = torch.randperm(len(features), generator=self.generator)
            for start in range(0, len(features), config.minibatch):
                indices = order[start:start+config.minibatch].to(self.device)
                logits, value = self.model(x[indices])
                distribution = Categorical(logits=logits)
                log_ratio = distribution.log_prob(act[indices]) - old[indices]
                ratio = log_ratio.exp()
                approx_kl = ((ratio - 1) - log_ratio).mean()
                if approx_kl.item() > config.target_kl:
                    stopped = True
                    break
                policy_loss = -torch.minimum(ratio * adv[indices],
                                             ratio.clamp(1-config.clip_ratio, 1+config.clip_ratio) * adv[indices]).mean()
                value_loss = .5 * (value - target[indices]).square().mean()
                entropy = distribution.entropy().mean()
                loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
                if not torch.isfinite(loss):
                    raise ValueError('non-finite PPO loss')
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                losses.append(loss.item())
                kls.append(approx_kl.item())
                entropies.append(entropy.item())
                clip_fractions.append(((ratio-1).abs() > config.clip_ratio).float().mean().item())
            if stopped:
                break
        if not losses:
            raise ValueError('stale rollout rejected by KL guard before any update')
        self.updates += 1
        return {'update': self.updates, 'transitions': len(features), 'minibatch_updates': len(losses),
                'loss': sum(losses)/len(losses), 'approx_kl': sum(kls)/len(kls),
                'entropy': sum(entropies)/len(entropies), 'clip_fraction': sum(clip_fractions)/len(clip_fractions),
                'early_stop_kl': stopped}

    def save(self, path, *, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Only tensors and primitive containers; load always enforces weights_only.
        data = {'schema_version': 1, 'feature_schema': 'local-goal-memory-v1',
                'config': asdict(self.config), 'updates': self.updates,
                'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                'rng': self.generator.get_state(), 'metadata': metadata or {}}
        temp = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                temp = Path(handle.name)
            torch.save(data, temp)
            temp.replace(path)
        finally:
            if temp and temp.exists():
                temp.unlink()

    @classmethod
    def load(cls, path, *, device='cpu'):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data.get('schema_version') != 1 or data.get('feature_schema') != 'local-goal-memory-v1':
            raise ValueError('PPO checkpoint schema mismatch')
        trainer = cls(PPOConfig(**data['config']), device=device)
        trainer.model.load_state_dict(data['model'])
        trainer.optimizer.load_state_dict(data['optimizer'])
        # Adam state follows the explicitly requested inference/training device.
        for state in trainer.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != 'step':
                    state[key] = value.to(trainer.device)
        trainer.generator.set_state(data['rng'])
        if type(data['updates']) is not int or data['updates'] < 0:
            raise ValueError('invalid checkpoint update count')
        trainer.updates = data['updates']
        return trainer, data['metadata']


class PPOPolicy:
    """Independent sampling state for shared frozen weights in population deployment."""
    def __init__(self, trainer, seed=0):
        self.trainer = trainer
        self.generator = torch.Generator(device='cpu').manual_seed(seed)

    def act_features(self, features, explore=True):
        with torch.no_grad():
            logits, _ = self.trainer.model(torch.tensor([features], dtype=torch.float32, device=self.trainer.device))
            action = (torch.multinomial(logits.softmax(-1).cpu(), 1, generator=self.generator).item()
                      if explore else logits.argmax(-1).item())
        return Action(action)
