"""Structured, bounded multi-agent debate for Agentic Core V1.

This module deliberately depends only on the public contracts in core.agentic.
AgenticCore itself remains unchanged: StructuredDebateCritic implements the
existing Critic protocol, and DebateDecider implements the existing Decider
protocol.

The default debate is deterministic and bounded so it can serve as a baseline
for future LLM-backed specialists/critics without making cost unpredictable.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

from core.agentic import Decision, Observation, Proposal


Policy = Callable[[Observation], Proposal | tuple[Any, float, str] | Any]


@dataclass(frozen=True)
class Critique:
    """One bounded, structured challenge against a proposal."""

    critic: str
    target_agent: str
    issue_type: str
    severity: float
    confidence_delta: float
    evidence: str = ""

    VALID_ISSUES = frozenset(
        {"assumption", "contradiction", "risk", "uncertainty", "evidence"}
    )

    def __post_init__(self) -> None:
        if self.issue_type not in self.VALID_ISSUES:
            raise ValueError("unsupported critique issue_type")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("severity must be between 0 and 1")
        if not -1.0 <= self.confidence_delta <= 1.0:
            raise ValueError("confidence_delta must be between -1 and 1")


@dataclass(frozen=True)
class ProposalDebate:
    """Original/revised confidence plus critiques for one specialist."""

    agent: str
    action: Any
    original_confidence: float
    revised_confidence: float
    rationale: str
    critiques: Tuple[Critique, ...] = ()


@dataclass(frozen=True)
class DebateLog:
    """JSON-friendly audit snapshot for one debate round."""

    participants: Tuple[str, ...]
    proposals: Tuple[ProposalDebate, ...]
    disagreement_count: int
    distinct_actions: int
    critique_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "participants": list(self.participants),
            "proposals": [
                {
                    "agent": item.agent,
                    "action": item.action,
                    "original_confidence": item.original_confidence,
                    "revised_confidence": item.revised_confidence,
                    "rationale": item.rationale,
                    "critiques": [
                        {
                            "critic": critique.critic,
                            "target_agent": critique.target_agent,
                            "issue_type": critique.issue_type,
                            "severity": critique.severity,
                            "confidence_delta": critique.confidence_delta,
                            "evidence": critique.evidence,
                        }
                        for critique in item.critiques
                    ],
                }
                for item in self.proposals
            ],
            "disagreement_count": self.disagreement_count,
            "distinct_actions": self.distinct_actions,
            "critique_count": self.critique_count,
        }


class PolicySpecialist:
    """Small adapter for deterministic or LLM-backed policy functions."""

    def __init__(self, name: str, policy: Policy) -> None:
        if not name:
            raise ValueError("specialist name is required")
        self.name = name
        self.policy = policy

    def propose(self, observation: Observation) -> Proposal:
        result = self.policy(observation)
        if isinstance(result, Proposal):
            if result.agent != self.name:
                return replace(result, agent=self.name)
            return result
        if isinstance(result, tuple) and len(result) == 3:
            action, confidence, rationale = result
            return Proposal(
                agent=self.name,
                action=action,
                confidence=float(confidence),
                rationale=str(rationale),
            )
        return Proposal(
            agent=self.name,
            action=result,
            confidence=0.5,
            rationale=f"{self.name} policy proposal",
        )


class Planner(PolicySpecialist):
    def __init__(self, policy: Policy) -> None:
        super().__init__("planner", policy)


class Explorer(PolicySpecialist):
    def __init__(self, policy: Policy) -> None:
        super().__init__("explorer", policy)


class DebateReviewer:
    """Protocol-like base for bounded deterministic reviewers."""

    name = "reviewer"

    def critiques(
        self, observation: Observation, proposals: Sequence[Proposal]
    ) -> Sequence[Critique]:
        raise NotImplementedError


class SkepticCritic(DebateReviewer):
    """Challenges unsupported certainty and cross-agent contradictions."""

    name = "skeptic"

    def __init__(self, *, high_confidence: float = 0.8, penalty: float = 0.12) -> None:
        self.high_confidence = high_confidence
        self.penalty = abs(float(penalty))

    def critiques(
        self, observation: Observation, proposals: Sequence[Proposal]
    ) -> Sequence[Critique]:
        del observation
        output: List[Critique] = []
        actions = {repr(proposal.action) for proposal in proposals}
        contradictory = len(actions) > 1
        for proposal in proposals:
            rationale = proposal.rationale.strip()
            if proposal.confidence >= self.high_confidence and len(rationale) < 12:
                output.append(
                    Critique(
                        critic=self.name,
                        target_agent=proposal.agent,
                        issue_type="assumption",
                        severity=0.65,
                        confidence_delta=-self.penalty,
                        evidence="high confidence without enough explicit support",
                    )
                )
            if contradictory:
                output.append(
                    Critique(
                        critic=self.name,
                        target_agent=proposal.agent,
                        issue_type="contradiction",
                        severity=0.45,
                        confidence_delta=-(self.penalty / 2),
                        evidence="specialists proposed different actions",
                    )
                )
        return tuple(output)


class RiskCritic(DebateReviewer):
    """Applies explicit unsafe-action vetoes and optional action-cost budgets."""

    name = "risk"

    def __init__(
        self,
        *,
        unsafe_actions: Iterable[Any] = (),
        action_costs: Mapping[Any, float] | None = None,
        max_cost: float | None = None,
        unsafe_penalty: float = 1.0,
        budget_penalty: float = 0.35,
    ) -> None:
        self.unsafe_actions = tuple(unsafe_actions)
        self.action_costs = dict(action_costs or {})
        self.max_cost = max_cost
        self.unsafe_penalty = abs(float(unsafe_penalty))
        self.budget_penalty = abs(float(budget_penalty))

    def critiques(
        self, observation: Observation, proposals: Sequence[Proposal]
    ) -> Sequence[Critique]:
        del observation
        output: List[Critique] = []
        for proposal in proposals:
            if proposal.action in self.unsafe_actions:
                output.append(
                    Critique(
                        critic=self.name,
                        target_agent=proposal.agent,
                        issue_type="risk",
                        severity=1.0,
                        confidence_delta=-self.unsafe_penalty,
                        evidence="action is explicitly marked unsafe",
                    )
                )
            if self.max_cost is not None:
                cost = float(self.action_costs.get(proposal.action, 0.0))
                if cost > self.max_cost:
                    output.append(
                        Critique(
                            critic=self.name,
                            target_agent=proposal.agent,
                            issue_type="risk",
                            severity=min(1.0, cost / max(self.max_cost, 1e-9)),
                            confidence_delta=-self.budget_penalty,
                            evidence=f"action cost {cost:.4g} exceeds budget {self.max_cost:.4g}",
                        )
                    )
        return tuple(output)


class MemoryAnalyst(DebateReviewer):
    """Turns Stage-2 retrieved memories into compact supporting/contrary evidence."""

    name = "memory_analyst"

    def __init__(self, *, reward: float = 0.08, penalty: float = 0.10) -> None:
        self.reward = abs(float(reward))
        self.penalty = abs(float(penalty))

    def critiques(
        self, observation: Observation, proposals: Sequence[Proposal]
    ) -> Sequence[Critique]:
        memories = observation.context.get("retrieved_memories", ())
        if not isinstance(memories, (list, tuple)):
            return ()
        output: List[Critique] = []
        for proposal in proposals:
            supporting = 0
            contrary = 0
            for item in memories[:5]:
                if not isinstance(item, Mapping):
                    continue
                same_action = item.get("action") == proposal.action
                verified = bool(item.get("verification_ok"))
                score = float(item.get("score", 0.0) or 0.0)
                if same_action and verified and score > 0:
                    supporting += 1
                elif same_action and not verified:
                    contrary += 1
            if supporting:
                output.append(
                    Critique(
                        critic=self.name,
                        target_agent=proposal.agent,
                        issue_type="evidence",
                        severity=min(1.0, supporting / 3),
                        confidence_delta=min(0.2, self.reward * supporting),
                        evidence=f"{supporting} relevant verified memories support this action",
                    )
                )
            if contrary:
                output.append(
                    Critique(
                        critic=self.name,
                        target_agent=proposal.agent,
                        issue_type="evidence",
                        severity=min(1.0, contrary / 3),
                        confidence_delta=-min(0.25, self.penalty * contrary),
                        evidence=f"{contrary} relevant failed memories used this action",
                    )
                )
        return tuple(output)


class StructuredDebateCritic:
    """One-round bounded debate implementing core.agentic.Critic."""

    def __init__(
        self,
        reviewers: Iterable[DebateReviewer],
        *,
        max_participants: int = 8,
        max_critiques_per_proposal: int = 4,
    ) -> None:
        self.reviewers = tuple(reviewers)
        if len(self.reviewers) > max_participants:
            raise ValueError("too many debate reviewers")
        if not 1 <= max_critiques_per_proposal <= 16:
            raise ValueError("max_critiques_per_proposal must be 1..16")
        self.max_critiques_per_proposal = max_critiques_per_proposal
        self.last_log: DebateLog | None = None

    def review(
        self, observation: Observation, proposals: Sequence[Proposal]
    ) -> Sequence[Proposal]:
        if not proposals:
            self.last_log = DebateLog((), (), 0, 0, 0)
            return ()

        by_agent: Dict[str, List[Critique]] = {proposal.agent: [] for proposal in proposals}
        for reviewer in self.reviewers:
            for critique in reviewer.critiques(observation, proposals):
                if critique.target_agent not in by_agent:
                    continue
                bucket = by_agent[critique.target_agent]
                if len(bucket) < self.max_critiques_per_proposal:
                    bucket.append(critique)

        revised: List[Proposal] = []
        audit: List[ProposalDebate] = []
        for proposal in proposals:
            critiques = tuple(by_agent[proposal.agent])
            delta = sum(item.confidence_delta for item in critiques)
            confidence = max(0.0, min(1.0, proposal.confidence + delta))
            metadata = dict(proposal.metadata)
            metadata["debate"] = {
                "original_confidence": proposal.confidence,
                "confidence_delta": delta,
                "critique_count": len(critiques),
                "issues": [item.issue_type for item in critiques],
            }
            revised_proposal = replace(
                proposal,
                confidence=confidence,
                metadata=metadata,
            )
            revised.append(revised_proposal)
            audit.append(
                ProposalDebate(
                    agent=proposal.agent,
                    action=proposal.action,
                    original_confidence=proposal.confidence,
                    revised_confidence=confidence,
                    rationale=proposal.rationale,
                    critiques=critiques,
                )
            )

        distinct_actions = len({repr(item.action) for item in proposals})
        disagreement_count = max(0, distinct_actions - 1)
        participants = tuple(
            [proposal.agent for proposal in proposals]
            + [reviewer.name for reviewer in self.reviewers]
        )
        self.last_log = DebateLog(
            participants=participants,
            proposals=tuple(audit),
            disagreement_count=disagreement_count,
            distinct_actions=distinct_actions,
            critique_count=sum(len(item.critiques) for item in audit),
        )
        return tuple(revised)


class DebateDecider:
    """Deterministic final selector with auditable ranking and margin."""

    def decide(
        self, observation: Observation, proposals: Sequence[Proposal]
    ) -> Decision:
        del observation
        if not proposals:
            raise ValueError("at least one proposal is required")

        indexed = list(enumerate(proposals))
        ranked = sorted(
            indexed,
            key=lambda pair: (-pair[1].confidence, pair[0]),
        )
        winner_index, winner = ranked[0]
        runner_up = ranked[1][1].confidence if len(ranked) > 1 else 0.0
        margin = winner.confidence - runner_up
        ranking = [
            {
                "agent": proposal.agent,
                "action": proposal.action,
                "confidence": proposal.confidence,
            }
            for _, proposal in ranked
        ]
        return Decision(
            action=winner.action,
            confidence=winner.confidence,
            rationale=winner.rationale,
            selected_agent=winner.agent,
            metadata={
                "why_selected": "highest revised confidence after bounded debate",
                "winner_original_index": winner_index,
                "margin": margin,
                "ranking": ranking,
            },
        )
