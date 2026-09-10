"""Generic headless episode runner shared by creature and market worlds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

from core.contracts import Agent, Environment

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass(frozen=True)
class EpisodeResult(Generic[StateT]):
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    final_state: StateT


def run_episode(
    environment: Environment[StateT, ActionT],
    agent: Agent[StateT, ActionT],
    *,
    max_steps: int = 1_000,
    seed: Optional[int] = None,
    explore: bool = True,
) -> EpisodeResult[StateT]:
    """Run one deterministic-capable episode using the shared V2 contracts."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    state = environment.reset(seed=seed)
    total_reward = 0.0
    terminated = False
    truncated = False

    for step_index in range(1, max_steps + 1):
        action = agent.act(state, explore=explore)
        result = environment.step(action)
        agent.observe(state, action, result.reward, result.state, result.done)
        total_reward += float(result.reward)
        state = result.state
        terminated = result.terminated
        truncated = result.truncated
        if result.done:
            return EpisodeResult(total_reward, step_index, terminated, truncated, state)

    return EpisodeResult(total_reward, max_steps, terminated, True, state)
