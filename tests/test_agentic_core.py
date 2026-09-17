from core.agentic import (
    AgenticCore,
    HighestConfidenceDecider,
    InMemoryExperienceStore,
    Observation,
    Proposal,
    Verification,
)


class FixedSpecialist:
    def __init__(self, name, action, confidence):
        self.name = name
        self.action = action
        self.confidence = confidence

    def propose(self, observation):
        return Proposal(
            agent=self.name,
            action=self.action,
            confidence=self.confidence,
            rationale=f"{self.name} sees {observation.state}",
        )


class EqualityVerifier:
    def verify(self, observation, decision, outcome):
        return Verification(
            ok=outcome == decision.action,
            score=1.0 if outcome == decision.action else 0.0,
            details="executor echoed selected action",
        )


def test_agentic_core_selects_verifies_and_remembers():
    memory = InMemoryExperienceStore()
    core = AgenticCore(
        specialists=[
            FixedSpecialist("explorer", "LEFT", 0.55),
            FixedSpecialist("planner", "RIGHT", 0.85),
        ],
        decider=HighestConfidenceDecider(),
        verifier=EqualityVerifier(),
        memory=memory,
    )

    experience = core.run(
        Observation(state="junction", context={"energy": 0.8}),
        executor=lambda action: action,
    )

    assert experience.decision.action == "RIGHT"
    assert experience.decision.selected_agent == "planner"
    assert experience.verification.ok is True
    assert len(memory.items) == 1
    assert memory.items[0] == experience


def test_proposal_rejects_invalid_confidence():
    try:
        Proposal(agent="bad", action="WAIT", confidence=1.5)
    except ValueError as exc:
        assert "confidence" in str(exc)
    else:
        raise AssertionError("Proposal accepted confidence above 1")
