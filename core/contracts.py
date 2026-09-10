"""Minimal contracts shared by every Neuro-Genesis V2 research world."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Optional, Protocol, TypeVar

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass(frozen=True)
class StepResult(Generic[StateT]):
    """Result returned by one environment transition."""

    state: StateT
    reward: float
    terminated: bool = False
    truncated: bool = False
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class Environment(Protocol[StateT, ActionT]):
    """Small Gym-like API without requiring Gymnasium."""

    def reset(self, seed: Optional[int] = None) -> StateT:
        ...

    def step(self, action: ActionT) -> StepResult[StateT]:
        ...


class Agent(Protocol[StateT, ActionT]):
    """Decision contract for RL, evolutionary, scripted, or LLM agents."""

    def act(self, state: StateT, explore: bool = True) -> ActionT:
        ...

    def observe(
        self,
        state: StateT,
        action: ActionT,
        reward: float,
        next_state: StateT,
        done: bool,
    ) -> None:
        ...
