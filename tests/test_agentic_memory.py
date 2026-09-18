from core.agentic import (
    AgenticCore,
    HighestConfidenceDecider,
    Observation,
    Proposal,
    Verification,
)
from core.agentic_memory import PersistentExperienceStore


class ContextAwareSpecialist:
    name = "context-aware"

    def __init__(self):
        self.seen = []

    def propose(self, observation):
        self.seen.append(observation)
        memories = observation.context.get("retrieved_memories", [])
        if memories and memories[0]["verification_ok"]:
            action = memories[0]["action"]
            confidence = 0.95
            rationale = "reuse verified memory"
        else:
            action = "EXPLORE"
            confidence = 0.6
            rationale = "no useful memory"
        return Proposal(self.name, action, confidence, rationale)


class EchoVerifier:
    def verify(self, observation, decision, outcome):
        return Verification(
            ok=outcome == decision.action,
            score=1.0 if outcome == decision.action else 0.0,
            details="echo verification",
        )


def run_once(store, specialist, observation):
    core = AgenticCore(
        specialists=[specialist],
        decider=HighestConfidenceDecider(),
        verifier=EchoVerifier(),
        memory=store,
    )
    return core.run(observation, executor=lambda action: action)


def test_persistent_memory_survives_restart_and_retrieves_relevant_experience(tmp_path):
    path = tmp_path / "agentic_memory.json"
    first = PersistentExperienceStore(path, capacity=10, top_k=2)
    specialist = ContextAwareSpecialist()

    exp = run_once(
        first,
        specialist,
        Observation(state="red-junction", context={"energy": "low", "region": "north"}),
    )
    assert exp.decision.action == "EXPLORE"
    assert path.exists()
    assert len(first) == 1

    restored = PersistentExperienceStore(path, capacity=10, top_k=2)
    hits = restored.context(
        Observation(state="red-junction", context={"energy": "low", "region": "north"})
    )
    assert len(hits) == 1
    assert hits[0]["verification_ok"] is True
    assert hits[0]["action"] == "EXPLORE"


def test_agentic_core_injects_retrieved_memory_before_specialist_decision(tmp_path):
    path = tmp_path / "agentic_memory.json"
    store = PersistentExperienceStore(path, capacity=10, top_k=3)
    seed_specialist = ContextAwareSpecialist()
    observation = Observation(state="food-visible", context={"energy": "medium", "zone": "east"})
    run_once(store, seed_specialist, observation)

    second_specialist = ContextAwareSpecialist()
    exp = run_once(store, second_specialist, observation)

    assert "retrieved_memories" in second_specialist.seen[-1].context
    assert exp.decision.rationale == "reuse verified memory"
    assert exp.decision.confidence == 0.95


def test_memory_is_bounded_and_prefers_similar_verified_context(tmp_path):
    path = tmp_path / "agentic_memory.json"
    store = PersistentExperienceStore(path, capacity=2, top_k=1)

    for state, context in [
        ("water", {"terrain": "lake"}),
        ("forest", {"terrain": "trees"}),
        ("desert", {"terrain": "sand"}),
    ]:
        run_once(store, ContextAwareSpecialist(), Observation(state=state, context=context))

    assert len(store) == 2
    hit = store.context(Observation(state="forest", context={"terrain": "trees"}))[0]
    assert hit["context"]["terrain"] == "trees"
