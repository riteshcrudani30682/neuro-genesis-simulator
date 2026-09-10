from core.contracts import StepResult
from core.replay import ReplayBuffer, Transition
from experiments.runner import run_episode


class CounterEnvironment:
    def __init__(self):
        self.value = 0

    def reset(self, seed=None):
        self.value = 0
        return self.value

    def step(self, action):
        self.value += action
        return StepResult(
            state=self.value,
            reward=float(action),
            terminated=self.value >= 3,
        )


class OneAgent:
    def __init__(self):
        self.observed = []

    def act(self, state, explore=True):
        return 1

    def observe(self, state, action, reward, next_state, done):
        self.observed.append((state, action, reward, next_state, done))


def test_step_result_done_property():
    assert not StepResult(state=0, reward=0.0).done
    assert StepResult(state=0, reward=0.0, terminated=True).done
    assert StepResult(state=0, reward=0.0, truncated=True).done


def test_replay_buffer_keeps_recent_transitions():
    buffer = ReplayBuffer(capacity=2)
    buffer.append(Transition(0, 1, 1.0, 1, False))
    buffer.append(Transition(1, 1, 1.0, 2, False))
    buffer.append(Transition(2, 1, 1.0, 3, True))
    assert len(buffer) == 2
    assert [item.state for item in buffer.sample(2)] == [1, 2] or [item.state for item in buffer.sample(2)] == [2, 1]


def test_episode_runner_preserves_transition_alignment():
    env = CounterEnvironment()
    agent = OneAgent()
    result = run_episode(env, agent, max_steps=10, seed=42)
    assert result.terminated
    assert result.steps == 3
    assert result.total_reward == 3.0
    assert result.final_state == 3
    assert agent.observed == [
        (0, 1, 1.0, 1, False),
        (1, 1, 1.0, 2, False),
        (2, 1, 1.0, 3, True),
    ]
