"""Tests for orchestration types in llm-types package.

Tests cover:
- All dataclass to_dict()/from_dict() roundtrips
- Nested object serialization
- JSON compatibility
- Enum serialization
- datetime handling
"""

import json
import pytest
from datetime import datetime

from u_llm_sdk.types import (
    Provider,
    # Enums
    OrchestratorRole,
    WorkerRole,
    ClarityLevel,
    MessageType,
    # Task & Clarity
    Task,
    UnclearAspect,
    ClarityAssessment,
    # Brainstorm & Consensus
    BrainstormConfig,
    ParticipantInput,
    DiscussionEntry,
    DissentingView,
    ConsensusEvaluation,
    ConsensusResult,
    # Escalation
    EscalationRequest,
    EscalationResponse,
    # Inter-LLM Communication
    LLMMessage,
    # Session & State
    SessionConfig,
    OrchestratorState,
)


class TestEnums:
    """Tests for orchestration enums."""

    def test_orchestrator_role_values(self):
        assert OrchestratorRole.MASTER.value == "master"
        assert OrchestratorRole.SUB.value == "sub"

    def test_worker_role_values(self):
        assert WorkerRole.CODE_IMPLEMENTER.value == "code_implementer"
        assert WorkerRole.DEEP_ANALYZER.value == "deep_analyzer"
        assert WorkerRole.STRATEGIST.value == "strategist"

    def test_clarity_level_values(self):
        assert ClarityLevel.CLEAR.value == "clear"
        assert ClarityLevel.NEEDS_CLARIFICATION.value == "needs_clarification"
        assert ClarityLevel.AMBIGUOUS.value == "ambiguous"

    def test_message_type_categories(self):
        # Task flow messages
        assert MessageType.TASK_ASSIGNMENT.value == "task_assignment"
        assert MessageType.TASK_RESULT.value == "task_result"
        # Discussion messages
        assert MessageType.OPINION.value == "opinion"
        assert MessageType.CONSENSUS_PROPOSAL.value == "consensus_proposal"
        # Control messages
        assert MessageType.ORCHESTRATOR_SWITCH.value == "orchestrator_switch"


class TestTask:
    """Tests for Task dataclass."""

    def test_roundtrip_minimal(self):
        task = Task(task_id="t1", objective="Build X", context="ctx")
        restored = Task.from_dict(task.to_dict())
        assert restored.task_id == "t1"
        assert restored.objective == "Build X"
        assert restored.constraints == []
        assert restored.clarity_level is None

    def test_roundtrip_full(self):
        task = Task(
            task_id="t2",
            objective="Implement feature",
            context="Full context",
            constraints=["no external deps", "test coverage > 80%"],
            expected_output="Working module",
            clarity_level=0.9,
            source="orchestrator",
        )
        restored = Task.from_dict(task.to_dict())
        assert restored.constraints == ["no external deps", "test coverage > 80%"]
        assert restored.clarity_level == 0.9
        assert restored.source == "orchestrator"

    def test_json_compatible(self):
        task = Task(task_id="t3", objective="Test", context="")
        json_str = json.dumps(task.to_dict())
        parsed = json.loads(json_str)
        restored = Task.from_dict(parsed)
        assert restored.task_id == "t3"


class TestUnclearAspect:
    """Tests for UnclearAspect dataclass."""

    def test_roundtrip(self):
        aspect = UnclearAspect(
            aspect_type="scope_ambiguity",
            description="Unclear boundaries",
            clarification_needed="Define module scope",
        )
        restored = UnclearAspect.from_dict(aspect.to_dict())
        assert restored.aspect_type == "scope_ambiguity"
        assert restored.description == "Unclear boundaries"

    def test_all_aspect_types(self):
        for aspect_type in [
            "knowledge_gap",
            "scope_ambiguity",
            "objective_ambiguity",
            "constraint_missing",
            "context_insufficient",
        ]:
            aspect = UnclearAspect(
                aspect_type=aspect_type,
                description=f"Test {aspect_type}",
                clarification_needed="Need clarification",
            )
            restored = UnclearAspect.from_dict(aspect.to_dict())
            assert restored.aspect_type == aspect_type


class TestClarityAssessment:
    """Tests for ClarityAssessment dataclass."""

    def test_roundtrip_minimal(self):
        assessment = ClarityAssessment(level=ClarityLevel.CLEAR, score=0.95)
        restored = ClarityAssessment.from_dict(assessment.to_dict())
        assert restored.level == ClarityLevel.CLEAR
        assert restored.score == 0.95
        assert restored.recommendation == "execute"

    def test_roundtrip_with_unclear_aspects(self):
        aspects = [
            UnclearAspect("scope_ambiguity", "desc1", "clarify1"),
            UnclearAspect("knowledge_gap", "desc2", "clarify2"),
        ]
        assessment = ClarityAssessment(
            level=ClarityLevel.NEEDS_CLARIFICATION,
            score=0.4,
            unclear_aspects=aspects,
            self_questions=["What about edge cases?", "Which approach?"],
            recommendation="clarify",
        )
        restored = ClarityAssessment.from_dict(assessment.to_dict())
        assert len(restored.unclear_aspects) == 2
        assert restored.unclear_aspects[0].aspect_type == "scope_ambiguity"
        assert len(restored.self_questions) == 2

    def test_enum_from_string(self):
        data = {"level": "ambiguous", "score": 0.1}
        restored = ClarityAssessment.from_dict(data)
        assert restored.level == ClarityLevel.AMBIGUOUS


class TestBrainstormConfig:
    """Tests for BrainstormConfig dataclass."""

    def test_defaults(self):
        config = BrainstormConfig()
        assert config.max_rounds == 3
        assert config.consensus_method == "majority"
        assert config.consensus_threshold == 0.67
        assert config.low_agreement_threshold == 0.4
        assert config.preserve_full_discussion is True

    def test_roundtrip_custom(self):
        config = BrainstormConfig(
            max_rounds=5,
            consensus_method="unanimous",
            consensus_threshold=0.9,
        )
        restored = BrainstormConfig.from_dict(config.to_dict())
        assert restored.max_rounds == 5
        assert restored.consensus_method == "unanimous"


class TestParticipantInput:
    """Tests for ParticipantInput dataclass."""

    def test_roundtrip(self):
        participant = ParticipantInput(
            provider=Provider.CLAUDE,
            analysis="Detailed analysis...",
            position="Support approach A",
            supporting_evidence=["Evidence 1", "Evidence 2"],
            concerns=["Concern about performance"],
            proposed_approach="Use caching",
        )
        restored = ParticipantInput.from_dict(participant.to_dict())
        assert restored.provider == Provider.CLAUDE
        assert restored.position == "Support approach A"
        assert len(restored.supporting_evidence) == 2

    def test_provider_from_string(self):
        data = {
            "provider": "gemini",
            "analysis": "Test",
            "position": "Neutral",
        }
        restored = ParticipantInput.from_dict(data)
        assert restored.provider == Provider.GEMINI


class TestDiscussionEntry:
    """Tests for DiscussionEntry dataclass."""

    def test_datetime_serialization(self):
        now = datetime(2025, 12, 18, 14, 30, 0)
        entry = DiscussionEntry(
            timestamp=now,
            speaker=Provider.GEMINI,
            message_type="opinion",
            content="I think we should...",
        )
        data = entry.to_dict()
        assert data["timestamp"] == "2025-12-18T14:30:00"

        restored = DiscussionEntry.from_dict(data)
        assert restored.timestamp == now
        assert restored.speaker == Provider.GEMINI

    def test_all_message_types(self):
        for msg_type in ["opinion", "rebuttal", "support", "question", "answer"]:
            entry = DiscussionEntry(
                timestamp=datetime.now(),
                speaker=Provider.CODEX,
                message_type=msg_type,
                content=f"Test {msg_type}",
            )
            restored = DiscussionEntry.from_dict(entry.to_dict())
            assert restored.message_type == msg_type


class TestConsensusTypes:
    """Tests for consensus-related dataclasses."""

    def test_dissenting_view_roundtrip(self):
        view = DissentingView(
            provider=Provider.CODEX,
            position="Against",
            reasoning="Performance concerns",
        )
        restored = DissentingView.from_dict(view.to_dict())
        assert restored.provider == Provider.CODEX
        assert restored.reasoning == "Performance concerns"

    def test_consensus_evaluation_roundtrip(self):
        views = [DissentingView(Provider.CODEX, "Against", "Concerns")]
        evaluation = ConsensusEvaluation(
            agreement_level=0.75,
            agreement_category="high",
            majority_position="Proceed with A",
            dissenting_views=views,
            recommendation="proceed",
        )
        restored = ConsensusEvaluation.from_dict(evaluation.to_dict())
        assert restored.agreement_level == 0.75
        assert len(restored.dissenting_views) == 1

    def test_consensus_result_full(self):
        entries = [
            DiscussionEntry(
                datetime(2025, 12, 18, 14, 0, 0),
                Provider.GEMINI,
                "opinion",
                "Let's do A",
            ),
            DiscussionEntry(
                datetime(2025, 12, 18, 14, 5, 0),
                Provider.CLAUDE,
                "support",
                "Agreed",
            ),
        ]
        result = ConsensusResult(
            success=True,
            final_decision="Do A",
            vote_breakdown={"gemini": "A", "claude": "A", "codex": "B"},
            discussion_summary="Decided on A",
            full_discussion_log=entries,
            escalated_to_user=False,
        )
        restored = ConsensusResult.from_dict(result.to_dict())
        assert restored.success is True
        assert len(restored.full_discussion_log) == 2
        assert restored.vote_breakdown["codex"] == "B"


class TestEscalationTypes:
    """Tests for escalation-related dataclasses."""

    def test_escalation_request_roundtrip(self):
        task = Task("t1", "Build", "ctx")
        assessment = ClarityAssessment(ClarityLevel.AMBIGUOUS, 0.2)
        request = EscalationRequest(
            source_worker=Provider.CLAUDE,
            original_task=task,
            clarity_assessment=assessment,
            specific_questions=["What's the scope?"],
            request_type="scope_definition",
        )
        restored = EscalationRequest.from_dict(request.to_dict())
        assert restored.source_worker == Provider.CLAUDE
        assert restored.original_task.task_id == "t1"
        assert restored.clarity_assessment.level == ClarityLevel.AMBIGUOUS

    def test_escalation_response_with_refined_task(self):
        refined = Task("t1-refined", "Build X specifically", "new ctx")
        response = EscalationResponse(
            clarifications={"scope": "Module X only"},
            refined_task=refined,
            additional_context="Focus on performance",
            permission_granted=True,
            guidance="Use existing patterns",
        )
        restored = EscalationResponse.from_dict(response.to_dict())
        assert restored.refined_task.task_id == "t1-refined"
        assert restored.clarifications["scope"] == "Module X only"

    def test_escalation_response_without_refined_task(self):
        response = EscalationResponse(
            clarifications={"answer": "Yes"},
        )
        restored = EscalationResponse.from_dict(response.to_dict())
        assert restored.refined_task is None


class TestLLMMessage:
    """Tests for LLMMessage dataclass."""

    def test_broadcast_target(self):
        msg = LLMMessage(
            message_id="m1",
            timestamp=datetime(2025, 12, 18, 15, 0, 0),
            source=Provider.GEMINI,
            target="broadcast",
            message_type=MessageType.STATUS_UPDATE,
            payload={"status": "ready"},
        )
        data = msg.to_dict()
        assert data["target"] == "broadcast"

        restored = LLMMessage.from_dict(data)
        assert restored.target == "broadcast"

    def test_provider_target(self):
        msg = LLMMessage(
            message_id="m2",
            timestamp=datetime(2025, 12, 18, 15, 0, 0),
            source=Provider.GEMINI,
            target=Provider.CLAUDE,
            message_type=MessageType.TASK_ASSIGNMENT,
            payload={"task_id": "t1"},
            requires_response=True,
        )
        restored = LLMMessage.from_dict(msg.to_dict())
        assert restored.target == Provider.CLAUDE
        assert restored.requires_response is True

    def test_all_message_types(self):
        for msg_type in MessageType:
            msg = LLMMessage(
                message_id="test",
                timestamp=datetime.now(),
                source=Provider.GEMINI,
                target="broadcast",
                message_type=msg_type,
            )
            restored = LLMMessage.from_dict(msg.to_dict())
            assert restored.message_type == msg_type


class TestSessionConfig:
    """Tests for SessionConfig dataclass."""

    def test_defaults(self):
        config = SessionConfig(
            session_id="s1",
            orchestrator_provider=Provider.GEMINI,
        )
        assert config.brainstorm_config.max_rounds == 3
        assert config.max_consensus_rounds == 3
        assert config.preserve_full_records is True

    def test_roundtrip_custom(self):
        brainstorm = BrainstormConfig(max_rounds=5, consensus_threshold=0.8)
        config = SessionConfig(
            session_id="s2",
            orchestrator_provider=Provider.CLAUDE,
            brainstorm_config=brainstorm,
            max_consensus_rounds=5,
        )
        restored = SessionConfig.from_dict(config.to_dict())
        assert restored.orchestrator_provider == Provider.CLAUDE
        assert restored.brainstorm_config.max_rounds == 5


class TestOrchestratorState:
    """Tests for OrchestratorState dataclass."""

    def test_minimal(self):
        state = OrchestratorState(
            session_id="s1",
            current_provider=Provider.GEMINI,
            session_context="Initial context",
        )
        restored = OrchestratorState.from_dict(state.to_dict())
        assert restored.session_id == "s1"
        assert restored.active_tasks == []

    def test_full_state(self):
        task = Task("t1", "Build", "ctx")
        assessment = ClarityAssessment(ClarityLevel.AMBIGUOUS, 0.3)
        escalation = EscalationRequest(Provider.CLAUDE, task, assessment)
        result = ConsensusResult(True, "Do A", {}, "Summary")

        state = OrchestratorState(
            session_id="s2",
            current_provider=Provider.GEMINI,
            session_context="Complex context",
            active_tasks=[task],
            pending_escalations=[escalation],
            consensus_history=[result],
        )
        restored = OrchestratorState.from_dict(state.to_dict())
        assert len(restored.active_tasks) == 1
        assert len(restored.pending_escalations) == 1
        assert len(restored.consensus_history) == 1
        assert restored.active_tasks[0].task_id == "t1"


class TestJSONCompatibility:
    """Tests for full JSON roundtrip compatibility."""

    def test_complex_state_json_roundtrip(self):
        """Test that a complex state can be serialized to JSON and back."""
        task = Task(
            task_id="t1",
            objective="Build feature",
            context="Full context",
            constraints=["constraint1"],
            clarity_level=0.8,
        )
        aspect = UnclearAspect("scope_ambiguity", "desc", "clarify")
        assessment = ClarityAssessment(
            ClarityLevel.NEEDS_CLARIFICATION,
            0.5,
            [aspect],
            ["question?"],
            "clarify",
        )
        escalation = EscalationRequest(Provider.CLAUDE, task, assessment)
        entry = DiscussionEntry(
            datetime(2025, 12, 18, 14, 0, 0),
            Provider.GEMINI,
            "opinion",
            "content",
        )
        result = ConsensusResult(True, "decision", {"a": "b"}, "summary", [entry])

        state = OrchestratorState(
            session_id="complex",
            current_provider=Provider.GEMINI,
            session_context="ctx",
            active_tasks=[task],
            pending_escalations=[escalation],
            consensus_history=[result],
        )

        # Full JSON roundtrip
        json_str = json.dumps(state.to_dict())
        parsed = json.loads(json_str)
        restored = OrchestratorState.from_dict(parsed)

        assert restored.session_id == "complex"
        assert restored.active_tasks[0].task_id == "t1"
        assert restored.pending_escalations[0].clarity_assessment.level == ClarityLevel.NEEDS_CLARIFICATION
        assert restored.consensus_history[0].full_discussion_log[0].speaker == Provider.GEMINI
