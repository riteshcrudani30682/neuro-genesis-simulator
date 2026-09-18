from core.agentic import AgenticCore, InMemoryExperienceStore, Observation, Proposal, Verification
from core.debate import (
    Critique,
    DebateDecider,
    Explorer,
    MemoryAnalyst,
    Planner,
    RiskCritic,
    SkepticCritic,
    StructuredDebateCritic,
)


class EchoVerifier:
    def verify(self, observation, decision, outcome):
        del observation
        return Verification(ok=outcome == decision.action, score=1.0)


def test_critique_schema_validation():
    try:
        Critique("x", "a", "unknown", 0.5, -0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid issue_type accepted")


def test_planner_and_explorer_adapt_injected_policies():
    planner = Planner(lambda observation: ("HOLD", 0.8, f"plan:{observation.state}"))
    explorer = Explorer(lambda observation: "TRY")
    obs = Observation("state")
    assert planner.propose(obs).agent == "planner"
    assert planner.propose(obs).confidence == 0.8
    assert explorer.propose(obs).action == "TRY"


def test_skeptic_penalizes_unsupported_high_confidence():
    proposals = (Proposal("planner", "A", 0.9, "short"),)
    critic = StructuredDebateCritic([SkepticCritic()])
    reviewed = critic.review(Observation("x"), proposals)
    assert reviewed[0].confidence < 0.9
    assert critic.last_log.critique_count == 1


def test_skeptic_logs_cross_agent_disagreement():
    proposals = (
        Proposal("planner", "A", 0.7, "supported planner rationale"),
        Proposal("explorer", "B", 0.7, "supported explorer rationale"),
    )
    critic = StructuredDebateCritic([SkepticCritic()])
    critic.review(Observation("x"), proposals)
    assert critic.last_log.distinct_actions == 2
    assert critic.last_log.disagreement_count == 1


def test_risk_critic_can_veto_unsafe_action():
    proposals = (
        Proposal("planner", "UNSAFE", 0.95, "looks attractive"),
        Proposal("explorer", "SAFE", 0.55, "lower risk"),
    )
    critic = StructuredDebateCritic([RiskCritic(unsafe_actions={"UNSAFE"})])
    reviewed = critic.review(Observation("x"), proposals)
    assert reviewed[0].confidence == 0.0
    assert reviewed[1].confidence == 0.55


def test_risk_critic_penalizes_over_budget_action():
    proposals = (Proposal("planner", "EXPENSIVE", 0.8, "fast"),)
    critic = StructuredDebateCritic(
        [RiskCritic(action_costs={"EXPENSIVE": 8.0}, max_cost=5.0)]
    )
    reviewed = critic.review(Observation("x"), proposals)
    assert reviewed[0].confidence < 0.8


def test_memory_analyst_rewards_verified_matching_memory():
    observation = Observation(
        "x",
        context={
            "retrieved_memories": [
                {"action": "A", "verification_ok": True, "score": 0.9},
                {"action": "A", "verification_ok": True, "score": 0.7},
            ]
        },
    )
    proposals = (Proposal("planner", "A", 0.5, "memory backed"),)
    critic = StructuredDebateCritic([MemoryAnalyst()])
    reviewed = critic.review(observation, proposals)
    assert reviewed[0].confidence > 0.5


def test_memory_analyst_penalizes_failed_matching_memory():
    observation = Observation(
        "x",
        context={"retrieved_memories": [{"action": "A", "verification_ok": False, "score": 0.9}]},
    )
    proposals = (Proposal("planner", "A", 0.5, "repeat"),)
    critic = StructuredDebateCritic([MemoryAnalyst()])
    reviewed = critic.review(observation, proposals)
    assert reviewed[0].confidence < 0.5


def test_debate_is_bounded_per_proposal():
    class Many:
        name = "many"

        def critiques(self, observation, proposals):
            del observation
            return tuple(
                Critique("many", proposals[0].agent, "uncertainty", 0.2, -0.01, str(i))
                for i in range(20)
            )

    critic = StructuredDebateCritic([Many()], max_critiques_per_proposal=4)
    critic.review(Observation("x"), (Proposal("planner", "A", 0.8, "rationale"),))
    assert critic.last_log.critique_count == 4


def test_too_many_reviewers_are_rejected():
    class Empty:
        name = "empty"

        def critiques(self, observation, proposals):
            return ()

    try:
        StructuredDebateCritic([Empty() for _ in range(9)], max_participants=8)
    except ValueError:
        pass
    else:
        raise AssertionError("participant bound not enforced")


def test_confidence_clamps_to_zero_and_one():
    class Adjust:
        name = "adjust"

        def critiques(self, observation, proposals):
            del observation
            return (
                Critique("adjust", proposals[0].agent, "evidence", 1.0, 0.9),
                Critique("adjust", proposals[1].agent, "risk", 1.0, -0.9),
            )

    proposals = (
        Proposal("a", "A", 0.8, "a"),
        Proposal("b", "B", 0.1, "b"),
    )
    reviewed = StructuredDebateCritic([Adjust()]).review(Observation("x"), proposals)
    assert reviewed[0].confidence == 1.0
    assert reviewed[1].confidence == 0.0


def test_debate_decider_is_stable_on_ties():
    proposals = (
        Proposal("first", "A", 0.7, "first"),
        Proposal("second", "B", 0.7, "second"),
    )
    decision = DebateDecider().decide(Observation("x"), proposals)
    assert decision.selected_agent == "first"
    assert decision.metadata["margin"] == 0.0


def test_debate_log_is_json_safe_shape():
    proposals = (
        Proposal("planner", "A", 0.8, "planner support"),
        Proposal("explorer", "B", 0.6, "explorer support"),
    )
    critic = StructuredDebateCritic([SkepticCritic()])
    critic.review(Observation("x"), proposals)
    data = critic.last_log.to_dict()
    assert isinstance(data["participants"], list)
    assert isinstance(data["proposals"], list)
    assert data["distinct_actions"] == 2


def test_agentic_core_runs_with_debate_without_core_changes():
    memory = InMemoryExperienceStore()
    critic = StructuredDebateCritic(
        [
            SkepticCritic(),
            RiskCritic(unsafe_actions={"RISKY"}),
            MemoryAnalyst(),
        ]
    )
    core = AgenticCore(
        specialists=[
            Planner(lambda observation: ("RISKY", 0.9, "fast but unsafe")),
            Explorer(lambda observation: ("SAFE", 0.65, "safer alternative")),
        ],
        decider=DebateDecider(),
        verifier=EchoVerifier(),
        memory=memory,
        critic=critic,
    )
    experience = core.run(Observation("junction"), executor=lambda action: action)
    assert experience.decision.action == "SAFE"
    assert experience.verification.ok
    assert len(memory.items) == 1
    assert critic.last_log.critique_count >= 1
