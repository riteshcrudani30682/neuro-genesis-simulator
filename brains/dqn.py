"""Adapter for an existing PyTorch Q-network, including legacy QNetwork classes.

Inject a network constructed with input_dim=2+4*(2*radius+1)**2. Five outputs
support STAY; an explicit four-action map can adapt a legacy four-output net.
The adapter records real transitions; it does not claim to train a frozen net.
"""
from core.replay import ReplayBuffer
from worlds.creature.entities import Action
from .baseline import RandomAgent


class NeuralAgentAdapter(RandomAgent):
    def __init__(self, network, *, seed=0, genome=None, action_map=tuple(Action), replay_capacity=10000):
        super().__init__(genome, seed)
        if not action_map or len(set(action_map)) != len(action_map):
            raise ValueError('action_map must be nonempty and unique')
        self.action_map = tuple(Action(a) for a in action_map)
        self.network = network
        self.network.eval()
        self.replay = ReplayBuffer(replay_capacity)

    def act_batch(self, states, explore=True):
        import torch
        device = next(self.network.parameters()).device
        values = torch.tensor([s.vector() for s in states], dtype=torch.float32, device=device)
        with torch.no_grad():
            scores = self.network(values)
        if scores.shape != (len(states), len(self.action_map)):
            raise ValueError('network output does not match the explicit action map')
        actions = [self.action_map[i] for i in scores.argmax(dim=1).tolist()]
        if explore:
            actions = [self.rng.choice(self.action_map) if self.rng.random() < self.genome.exploration_tendency
                       else action for action in actions]
        return actions

    def act(self, state, explore=True):
        return self.act_batch([state], explore)[0]

    def observe(self, state, action, reward, next_state, done):
        from core.replay import Transition
        self.replay.append(Transition(state, action, reward, next_state, done))
