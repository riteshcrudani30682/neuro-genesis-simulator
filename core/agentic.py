"""Agentic Core V1: reusable observe -> specialists -> critic -> decide -> verify -> remember loop."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class Observation:
    """Normalized perception payload shared with every specialist."""

    state: Any
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Proposal:
    """One specialist's candidate action and supporting metadata."""

    agent: str
    action: Any
    confidence: float
    rationale: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class Decision:
    """Final selected action after debate/critique."""

    action: Any
    confidence: float
    rationale: str
    selected_agent: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class Verification:
    """Result of checking whether an executed decision behaved as expected."""

    ok: bool
    score: float
    details: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Experience:
    """Compact episode memory suitable for persistent storage or replay."""

    observation: Observation
    proposals: Sequence[Proposal]
    decision: Decision
    outcome: Any
    verification: Verification


class Specialist(Protocol):
    name: str

    def propose(self, observation: Observation) -> Proposal:
        ...


class Critic(Protocol):
    def review(self, observation: Observation, proposals: Sequence[Proposal]) -> Sequence[Proposal]:
        ...


class Decider(Protocol):
    def decide(self, observation: Observation, proposals: Sequence[Proposal]) -> Decision:
        ...


class Verifier(Protocol):
    def verify(self, observation: Observation, decision: Decision, outcome: Any) -> Verification:
        ...


class Memory(Protocol):
    def remember(self, experience: Experience) -> None:
        ...


class InMemoryExperienceStore:
    """Small default memory backend used by tests and local experiments."""

    def __init__(self) -> None:
        self.items: List[Experience] = []

    def remember(self, experience: Experience) -> None:
        self.items.append(experience)


class HighestConfidenceDecider:
    """Deterministic baseline decider for benchmarking smarter policies."""

    def decide(self, observation: Observation, proposals: Sequence[Proposal]) -> Decision:
        del observation
        if not proposals:
            raise ValueError("at least one proposal is required")
        winner = max(proposals, key=lambda proposal: proposal.confidence)
        return Decision(
            action=winner.action,
            confidence=winner.confidence,
            rationale=winner.rationale,
            selected_agent=winner.agent,
            metadata={"proposal_count": len(proposals)},
        )


class PassThroughCritic:
    """Baseline critic that preserves proposals unchanged."""

    def review(self, observation: Observation, proposals: Sequence[Proposal]) -> Sequence[Proposal]:
        del observation
        return tuple(proposals)


class AgenticCore:
    """Coordinates one complete autonomous reasoning-and-learning cycle.

    The executor is intentionally supplied per call. This keeps environment or
    broker side-effects outside the reasoning core and makes shadow/paper mode
    straightforward: pass a simulator executor instead of a real executor.
    """

    def __init__(
        self,
        specialists: Iterable[Specialist],
        decider: Decider,
        verifier: Verifier,
        memory: Memory,
        critic: Critic | None = None,
    ) -> None:
        self.specialists = tuple(specialists)
        if not self.specialists:
            raise ValueError("at least one specialist is required")
        self.decider = decider
        self.verifier = verifier
        self.memory = memory
        self.critic = critic or PassThroughCritic()

    def run(self, observation: Observation, executor) -> Experience:
        proposals = tuple(agent.propose(observation) for agent in self.specialists)
        reviewed = tuple(self.critic.review(observation, proposals))
        decision = self.decider.decide(observation, reviewed)
        outcome = executor(decision.action)
        verification = self.verifier.verify(observation, decision, outcome)
        experience = Experience(
            observation=observation,
            proposals=reviewed,
            decision=decision,
            outcome=outcome,
            verification=verification,
        )
        self.memory.remember(experience)
        return experience
