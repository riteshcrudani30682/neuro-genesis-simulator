"""Reusable transition and replay-buffer primitives for V2 experiments."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque, Generic, Iterable, List, TypeVar

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass(frozen=True)
class Transition(Generic[StateT, ActionT]):
    state: StateT
    action: ActionT
    reward: float
    next_state: StateT
    done: bool


class ReplayBuffer(Generic[StateT, ActionT]):
    def __init__(self, capacity: int = 10_000):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._items: Deque[Transition[StateT, ActionT]] = deque(maxlen=capacity)

    def append(self, transition: Transition[StateT, ActionT]) -> None:
        self._items.append(transition)

    def extend(self, transitions: Iterable[Transition[StateT, ActionT]]) -> None:
        self._items.extend(transitions)

    def sample(self, batch_size: int, rng: random.Random | None = None) -> List[Transition[StateT, ActionT]]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if batch_size > len(self._items):
            raise ValueError("batch_size exceeds replay buffer size")
        sampler = rng or random
        return sampler.sample(list(self._items), batch_size)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
