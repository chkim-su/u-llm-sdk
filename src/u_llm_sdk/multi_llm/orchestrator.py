"""Orchestrator implementations for Multi-LLM coordination.

This module provides orchestrator classes that manage multi-LLM workflows.
Each orchestrator has provider-specific characteristics and routing logic.

Architecture:
- MasterOrchestrator: Protocol defining the orchestrator interface
- GeminiOrchestrator: Default orchestrator (human communication, session management)
- ClaudeOrchestrator: Code-focused sub-orchestrator (precise implementation)
- CodexOrchestrator: Analysis sub-orchestrator (deep thinking, debugging)
- OrchestratorFactory: Creates orchestrators based on provider type

Flexibility Levels:
- Level A: Runtime replacement (switch during execution)
- Level B: Session-fixed with delegation (default)
- Level C: Task-unit switching (per-task orchestrator)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Protocol

from u_llm_sdk.types import (
    BrainstormConfig,
    ClarityLevel,
    ConsensusResult,
    EscalationRequest,
    EscalationResponse,
    LLMMessage,
    LLMResult,
    MessageType,
    OrchestratorRole,
    OrchestratorState,
    Provider,
    SessionConfig,
    Task,
    WorkerRole,
)

from .brainstorm import BrainstormModule
from .clarity import ClarityGate
from .consensus import ConsensusLoop
from .escalation import EscalationProtocol
from .utils import extract_json

if TYPE_CHECKING:
    from u_llm_sdk.llm.providers.base import BaseProvider

logger = logging.getLogger(__name__)


# =============================================================================
# SPECIALIZATION KNOWLEDGE (Injected into orchestrator prompts)
# =============================================================================

SPECIALIZATION_KNOWLEDGE = """
## Available Specialized Workers

### Claude (Code Specialist)
- **Strengths**: Code writing, precise mechanical work, detailed implementation
- **Weaknesses**:
  - Cannot see the "big picture" (limited scope view)
  - Difficulty maintaining long-term memory
  - May distort or misunderstand original intent
- **Suitable tasks**: Code implementation, refactoring, bug fixes, test writing
- **Delegation ability**: Can parallelize via built-in Task tool (clear tasks only)
- **Cautions**:
  - Encourage upward queries when uncertain
  - Do not delegate decisions requiring broad perspective
  - Must provide clear scope

### Codex (Deep Thinker)
- **Strengths**: Academic knowledge, mathematical theory, forward-thinking debugging, risk analysis
- **Weaknesses**:
  - May dive too deep
  - Can overdo even when not wrong
- **Suitable tasks**: Architecture verification, theoretical analysis, debugging strategy, risk prediction
- **Cautions**:
  - Must clearly limit work scope
  - Results need filtering (remove excessive parts)

### Gemini (Human Interface) - Self
- **Strengths**: Understanding human needs, spatial cognition, multi-dimensional problem understanding, large context
- **Weaknesses**: Very weak at actual coding, prone to errors and conflicts
- **Suitable tasks**: Requirements analysis, overall session management, understanding user intent
- **Cautions**:
  - Code implementation must be delegated to Claude
  - Can provide coding knowledge/tips, but implementation must be delegated

## Routing Guidelines

Select appropriate Worker based on task characteristics:

1. Code-related + clear scope → Claude
2. Theory/analysis + potential risks → Codex (scope limitation required)
3. User intent understanding + overall coordination → Gemini (self)
4. Unclear tasks → Request clarification first, then route
5. Design/strategy needed → Gemini provides, Claude implements

## Collaboration Pattern

Gemini knowledge + Claude implementation:
- Gemini: Coding knowledge, design philosophy, architecture suggestions
- Claude: Technical review/modification of Gemini's suggestions, then implementation
"""


# =============================================================================
# ORCHESTRATOR PROTOCOL
# =============================================================================


class LLMProvider(Protocol):
    """Protocol for LLM provider used by orchestrators."""

    async def run(
        self,
        prompt: str,
        *,
        session_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> LLMResult:
        ...


class MasterOrchestrator(Protocol):
    """Protocol defining the orchestrator interface.

    Orchestrators manage multi-LLM workflows, handling:
    - Session management
    - Task routing to specialized workers
    - Consensus facilitation
    - Result aggregation

    Example:
        >>> orchestrator = GeminiOrchestrator(providers, config)
        >>> result = await orchestrator.process_request("Build auth system")
        >>> if result.needs_brainstorm:
        ...     consensus = await orchestrator.facilitate_brainstorm(result.topic)
    """

    @property
    def role(self) -> OrchestratorRole:
        """The orchestrator's role (master/worker/reviewer)."""
        ...

    @property
    def provider(self) -> Provider:
        """The underlying LLM provider."""
        ...

    async def process_request(
        self,
        request: str,
        *,
        context: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> OrchestratorResponse:
        """Process a user request and determine next actions.

        Args:
            request: The user's request or task description
            context: Optional context from previous interactions
            session_id: Optional session ID for continuity

        Returns:
            OrchestratorResponse with routing decisions and actions
        """
        ...

    async def route_task(
        self,
        task: Task,
        *,
        session_id: Optional[str] = None,
    ) -> TaskRouting:
        """Route a task to the appropriate worker.

        Args:
            task: The task to route
            session_id: Optional session ID

        Returns:
            TaskRouting with target worker and instructions
        """
        ...

    async def handle_escalation(
        self,
        request: EscalationRequest,
        *,
        session_id: Optional[str] = None,
    ) -> EscalationResponse:
        """Handle an escalation from a worker.

        Args:
            request: The escalation request from a worker
            session_id: Optional session ID

        Returns:
            EscalationResponse with clarifications or refined task
        """
        ...

    async def facilitate_brainstorm(
        self,
        topic: str,
        *,
        initial_context: Optional[str] = None,
    ) -> ConsensusResult:
        """Facilitate a brainstorming session among all providers.

        Args:
            topic: The topic for brainstorming
            initial_context: Optional context to seed the discussion

        Returns:
            ConsensusResult with final decision and discussion log
        """
        ...

    async def aggregate_results(
        self,
        results: list[WorkerResult],
        *,
        session_id: Optional[str] = None,
    ) -> AggregatedResult:
        """Aggregate results from multiple workers.

        Args:
            results: List of results from workers
            session_id: Optional session ID

        Returns:
            AggregatedResult with combined output
        """
        ...


# =============================================================================
# DATA CLASSES
# =============================================================================


class TaskType(Enum):
    """Classification of task types for routing."""

    SIMPLE = "simple"  # Direct execution, no discussion needed
    COMPLEX = "complex"  # Requires brainstorming
    CODE = "code"  # Code implementation task
    ANALYSIS = "analysis"  # Deep analysis task
    DESIGN = "design"  # Design/strategy task
    UNCLEAR = "unclear"  # Needs clarification


class OrchestratorResponse:
    """Response from orchestrator after processing a request."""

    def __init__(
        self,
        task_type: TaskType,
        tasks: list[Task],
        needs_brainstorm: bool = False,
        brainstorm_topic: Optional[str] = None,
        needs_clarification: bool = False,
        clarification_questions: Optional[list[str]] = None,
        direct_response: Optional[str] = None,
    ):
        self.task_type = task_type
        self.tasks = tasks
        self.needs_brainstorm = needs_brainstorm
        self.brainstorm_topic = brainstorm_topic
        self.needs_clarification = needs_clarification
        self.clarification_questions = clarification_questions or []
        self.direct_response = direct_response

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "task_type": self.task_type.value,
            "tasks": [t.to_dict() for t in self.tasks],
            "needs_brainstorm": self.needs_brainstorm,
            "brainstorm_topic": self.brainstorm_topic,
            "needs_clarification": self.needs_clarification,
            "clarification_questions": self.clarification_questions,
            "direct_response": self.direct_response,
        }


class TaskRouting:
    """Routing decision for a task."""

    def __init__(
        self,
        target_worker: Provider,
        worker_role: WorkerRole,
        task: Task,
        instructions: str,
        delegation_allowed: bool = True,
        scope_limitations: Optional[list[str]] = None,
    ):
        self.target_worker = target_worker
        self.worker_role = worker_role
        self.task = task
        self.instructions = instructions
        self.delegation_allowed = delegation_allowed
        self.scope_limitations = scope_limitations or []

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "target_worker": self.target_worker.value,
            "worker_role": self.worker_role.value,
            "task": self.task.to_dict(),
            "instructions": self.instructions,
            "delegation_allowed": self.delegation_allowed,
            "scope_limitations": self.scope_limitations,
        }


class WorkerResult:
    """Result from a worker execution."""

    def __init__(
        self,
        worker: Provider,
        task_id: str,
        success: bool,
        output: str,
        artifacts: Optional[list[str]] = None,
        errors: Optional[list[str]] = None,
    ):
        self.worker = worker
        self.task_id = task_id
        self.success = success
        self.output = output
        self.artifacts = artifacts or []
        self.errors = errors or []

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "worker": self.worker.value,
            "task_id": self.task_id,
            "success": self.success,
            "output": self.output,
            "artifacts": self.artifacts,
            "errors": self.errors,
        }


class AggregatedResult:
    """Aggregated result from multiple workers."""

    def __init__(
        self,
        success: bool,
        summary: str,
        worker_results: list[WorkerResult],
        final_output: str,
        recommendations: Optional[list[str]] = None,
    ):
        self.success = success
        self.summary = summary
        self.worker_results = worker_results
        self.final_output = final_output
        self.recommendations = recommendations or []

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "summary": self.summary,
            "worker_results": [r.to_dict() for r in self.worker_results],
            "final_output": self.final_output,
            "recommendations": self.recommendations,
        }


# =============================================================================
# BASE ORCHESTRATOR
# =============================================================================


class BaseOrchestrator(ABC):
    """Base class for orchestrator implementations.

    Provides common functionality for all orchestrators including:
    - State management
    - Message logging
    - Component initialization (ClarityGate, ConsensusLoop, etc.)
    """

    ORCHESTRATOR_ROLE: OrchestratorRole
    PROVIDER: Provider

    def __init__(
        self,
        providers: dict[Provider, LLMProvider],
        config: Optional[SessionConfig] = None,
    ):
        """Initialize the orchestrator.

        Args:
            providers: Dict mapping Provider enum to LLM provider instances
            config: Optional session configuration
        """
        self.providers = providers

        # Create default config if not provided
        if config is None:
            import uuid

            config = SessionConfig(
                session_id=str(uuid.uuid4()),
                orchestrator_provider=self.PROVIDER,
            )
        self.config = config

        self._state = OrchestratorState(
            session_id=config.session_id,
            current_provider=self.PROVIDER,
            session_context="",
            active_tasks=[],
            pending_escalations=[],
            consensus_history=[],
        )

        # Additional internal state (not in OrchestratorState)
        self._message_history: list[LLMMessage] = []

        # Initialize components
        self._clarity_gate: Optional[ClarityGate] = None
        self._escalation_protocol: Optional[EscalationProtocol] = None
        self._consensus_loop: Optional[ConsensusLoop] = None
        self._brainstorm_module: Optional[BrainstormModule] = None

        self._init_components()

    def _init_components(self) -> None:
        """Initialize orchestration components."""
        # ClarityGate uses the primary provider for this orchestrator
        if self.PROVIDER in self.providers:
            self._clarity_gate = ClarityGate(self.providers[self.PROVIDER])

        # EscalationProtocol uses this orchestrator as handler
        if self.PROVIDER in self.providers:
            self._escalation_protocol = EscalationProtocol(
                self.providers[self.PROVIDER]
            )

        # ConsensusLoop uses brainstorm_config from session config
        brainstorm_cfg = self.config.brainstorm_config
        self._consensus_loop = ConsensusLoop(self.providers, brainstorm_cfg)
        self._brainstorm_module = BrainstormModule(self.providers, brainstorm_cfg)

    @property
    def role(self) -> OrchestratorRole:
        """The orchestrator's role."""
        return self.ORCHESTRATOR_ROLE

    @property
    def provider(self) -> Provider:
        """The underlying LLM provider."""
        return self.PROVIDER

    @property
    def state(self) -> OrchestratorState:
        """Current orchestrator state."""
        return self._state

    @property
    def message_history(self) -> list[LLMMessage]:
        """Message history for this orchestrator."""
        return self._message_history

    def _log_message(
        self,
        content: str,
        message_type: MessageType = MessageType.TASK_RESULT,
        target: Provider | str = "broadcast",
    ) -> None:
        """Log a message to history."""
        import uuid

        msg = LLMMessage(
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            source=self.PROVIDER,
            target=target,
            message_type=message_type,
            payload={"content": content},
            context_reference=self._state.session_id,
            requires_response=False,
        )
        self._message_history.append(msg)

    def _get_own_provider(self) -> LLMProvider:
        """Get the LLM provider for this orchestrator."""
        if self.PROVIDER not in self.providers:
            raise ValueError(f"Provider {self.PROVIDER} not available")
        return self.providers[self.PROVIDER]

    @abstractmethod
    async def process_request(
        self,
        request: str,
        *,
        context: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> OrchestratorResponse:
        """Process a user request."""
        ...

    @abstractmethod
    async def route_task(
        self,
        task: Task,
        *,
        session_id: Optional[str] = None,
    ) -> TaskRouting:
        """Route a task to appropriate worker."""
        ...

    async def handle_escalation(
        self,
        request: EscalationRequest,
        *,
        session_id: Optional[str] = None,
    ) -> EscalationResponse:
        """Handle escalation from worker."""
        if self._escalation_protocol is None:
            raise ValueError("EscalationProtocol not initialized")
        return await self._escalation_protocol.escalate(request, session_id=session_id)

    async def facilitate_brainstorm(
        self,
        topic: str,
        *,
        initial_context: Optional[str] = None,
    ) -> ConsensusResult:
        """Facilitate brainstorming session."""
        if self._brainstorm_module is None:
            raise ValueError("BrainstormModule not initialized")
        result = await self._brainstorm_module.run_session(
            topic, context=initial_context
        )
        return result.consensus

    async def aggregate_results(
        self,
        results: list[WorkerResult],
        *,
        session_id: Optional[str] = None,
    ) -> AggregatedResult:
        """Aggregate results from workers."""
        # Default implementation - subclasses can override
        all_success = all(r.success for r in results)
        outputs = [r.output for r in results if r.output]

        return AggregatedResult(
            success=all_success,
            summary=f"Aggregated {len(results)} worker results",
            worker_results=results,
            final_output="\n\n---\n\n".join(outputs),
            recommendations=[],
        )


# =============================================================================
# GEMINI ORCHESTRATOR (Default Master)
# =============================================================================

GEMINI_SYSTEM_PROMPT = f"""You are the Master Orchestrator managing a multi-LLM system.

{SPECIALIZATION_KNOWLEDGE}

Your responsibilities:
1. Understand user intent and requests
2. Classify tasks (simple/complex/code/analysis/design/unclear)
3. Route tasks to appropriate specialized workers
4. Facilitate brainstorming when needed
5. Aggregate results and communicate with users

Remember:
- You handle session management and human communication
- Delegate code implementation to Claude
- Delegate deep analysis to Codex
- Provide design/strategy guidance, but don't implement code yourself

Respond with a JSON object containing your analysis and decisions.
"""


class GeminiOrchestrator(BaseOrchestrator):
    """Gemini-based master orchestrator.

    Gemini excels at:
    - Human communication and intent understanding
    - Session management (large context)
    - Strategic thinking and design
    - Multi-dimensional problem analysis

    Gemini should NOT:
    - Write code directly (delegate to Claude)
    - Perform mechanical repetitive tasks
    """

    ORCHESTRATOR_ROLE = OrchestratorRole.MASTER
    PROVIDER = Provider.GEMINI

    async def process_request(
        self,
        request: str,
        *,
        context: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> OrchestratorResponse:
        """Process user request and determine actions.

        Gemini analyzes the request, classifies it, and decides
        whether to route directly, initiate brainstorming, or
        request clarification.
        """
        self._state.session_id = session_id
        self._log_message(request, MessageType.TASK_ASSIGNMENT)

        prompt = self._build_process_prompt(request, context)

        try:
            llm = self._get_own_provider()
            result = await llm.run(prompt, session_id=session_id)
            response = self._parse_process_response(result.text)
            self._log_message(result.text, MessageType.TASK_RESULT)
            return response

        except Exception as e:
            logger.error(f"Failed to process request: {e}")
            # Fail-open: return simple task classification
            return OrchestratorResponse(
                task_type=TaskType.SIMPLE,
                tasks=[
                    Task(
                        task_id="fallback-1",
                        objective=request,
                        context=context or "",
                        constraints=[],
                        expected_output=None,
                        clarity_level=0.5,
                        source="gemini_fallback",
                    )
                ],
            )

    def _build_process_prompt(self, request: str, context: Optional[str]) -> str:
        """Build prompt for request processing."""
        parts = [
            GEMINI_SYSTEM_PROMPT,
            "",
            "---",
            "",
            "USER REQUEST:",
            request,
        ]

        if context:
            parts.extend(["", "CONTEXT:", context])

        parts.extend(
            [
                "",
                "Please analyze this request and respond with JSON:",
                "{",
                '  "task_type": "simple|complex|code|analysis|design|unclear",',
                '  "needs_brainstorm": true/false,',
                '  "brainstorm_topic": "topic if brainstorm needed",',
                '  "needs_clarification": true/false,',
                '  "clarification_questions": ["question1", ...],',
                '  "tasks": [',
                "    {",
                '      "task_id": "unique-id",',
                '      "objective": "what to accomplish",',
                '      "context": "relevant context",',
                '      "constraints": ["constraint1", ...],',
                '      "expected_output": "expected result"',
                "    }",
                "  ],",
                '  "direct_response": "response if simple query"',
                "}",
            ]
        )

        return "\n".join(parts)

    def _parse_process_response(self, text: str) -> OrchestratorResponse:
        """Parse Gemini's response into OrchestratorResponse."""
        import json

        try:
            data = json.loads(extract_json(text))

            task_type = TaskType(data.get("task_type", "simple"))

            tasks = []
            for t in data.get("tasks", []):
                tasks.append(
                    Task(
                        task_id=t.get("task_id", "auto"),
                        objective=t.get("objective", ""),
                        context=t.get("context", ""),
                        constraints=t.get("constraints", []),
                        expected_output=t.get("expected_output"),
                        clarity_level=0.8,
                        source="gemini_orchestrator",
                    )
                )

            return OrchestratorResponse(
                task_type=task_type,
                tasks=tasks,
                needs_brainstorm=data.get("needs_brainstorm", False),
                brainstorm_topic=data.get("brainstorm_topic"),
                needs_clarification=data.get("needs_clarification", False),
                clarification_questions=data.get("clarification_questions", []),
                direct_response=data.get("direct_response"),
            )

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse process response: {e}")
            return OrchestratorResponse(
                task_type=TaskType.UNCLEAR,
                tasks=[],
                needs_clarification=True,
                clarification_questions=["Could you please clarify your request?"],
            )

    async def route_task(
        self,
        task: Task,
        *,
        session_id: Optional[str] = None,
    ) -> TaskRouting:
        """Route task to appropriate worker based on characteristics."""
        prompt = self._build_routing_prompt(task)

        try:
            llm = self._get_own_provider()
            result = await llm.run(prompt, session_id=session_id)
            return self._parse_routing_response(result.text, task)

        except Exception as e:
            logger.error(f"Failed to route task: {e}")
            # Default: route to Claude for code, Codex for analysis
            if any(
                kw in task.objective.lower()
                for kw in ["implement", "code", "write", "fix", "refactor"]
            ):
                return TaskRouting(
                    target_worker=Provider.CLAUDE,
                    worker_role=WorkerRole.CODE_IMPLEMENTER,
                    task=task,
                    instructions="Implement as specified. Escalate if unclear.",
                    delegation_allowed=True,
                )
            else:
                return TaskRouting(
                    target_worker=Provider.CODEX,
                    worker_role=WorkerRole.DEEP_ANALYZER,
                    task=task,
                    instructions="Analyze within defined scope. Filter excessive details.",
                    delegation_allowed=False,
                    scope_limitations=["Focus on the specific question"],
                )

    def _build_routing_prompt(self, task: Task) -> str:
        """Build prompt for task routing decision."""
        return f"""Based on the specialization knowledge, route this task to the appropriate worker.

TASK:
- Objective: {task.objective}
- Context: {task.context}
- Constraints: {', '.join(task.constraints) if task.constraints else 'None'}

Respond with JSON:
{{
  "target_worker": "claude|codex|gemini",
  "worker_role": "implementer|analyzer|reviewer|coordinator",
  "instructions": "specific instructions for the worker",
  "delegation_allowed": true/false,
  "scope_limitations": ["limitation1", ...]
}}"""

    def _parse_routing_response(self, text: str, task: Task) -> TaskRouting:
        """Parse routing response."""
        import json

        try:
            data = json.loads(extract_json(text))

            worker_map = {
                "claude": Provider.CLAUDE,
                "codex": Provider.CODEX,
                "gemini": Provider.GEMINI,
            }
            role_map = {
                "implementer": WorkerRole.CODE_IMPLEMENTER,
                "analyzer": WorkerRole.DEEP_ANALYZER,
                "reviewer": WorkerRole.DEEP_ANALYZER,
                "coordinator": WorkerRole.STRATEGIST,
            }

            return TaskRouting(
                target_worker=worker_map.get(
                    data.get("target_worker", "claude"), Provider.CLAUDE
                ),
                worker_role=role_map.get(
                    data.get("worker_role", "implementer"), WorkerRole.CODE_IMPLEMENTER
                ),
                task=task,
                instructions=data.get("instructions", "Execute task as specified"),
                delegation_allowed=data.get("delegation_allowed", True),
                scope_limitations=data.get("scope_limitations", []),
            )

        except (json.JSONDecodeError, ValueError):
            # Default to Claude implementer
            return TaskRouting(
                target_worker=Provider.CLAUDE,
                worker_role=WorkerRole.CODE_IMPLEMENTER,
                task=task,
                instructions="Execute task. Escalate uncertainties.",
                delegation_allowed=True,
            )


# =============================================================================
# CLAUDE ORCHESTRATOR (Code-Focused Sub-Orchestrator)
# =============================================================================

CLAUDE_SYSTEM_PROMPT = """You are a Code Specialist sub-orchestrator.

Your strengths:
- Precise code implementation
- Detailed technical work
- Mechanical and systematic execution
- Bug fixing and refactoring

Your limitations:
- Cannot see the "big picture" easily
- May lose track of original intent
- Should escalate when uncertain

Your responsibilities:
1. Assess task clarity before execution (ClarityGate)
2. Escalate to master orchestrator when unclear
3. Execute clear code tasks precisely
4. Delegate sub-tasks only when clearly defined

CRITICAL: Always verify task clarity. Escalate uncertainties to master.
"""


class ClaudeOrchestrator(BaseOrchestrator):
    """Claude-based code-focused sub-orchestrator.

    Claude excels at:
    - Code implementation
    - Precise mechanical work
    - Detailed technical tasks
    - Systematic execution

    Claude should:
    - Always assess clarity before execution
    - Escalate when tasks are unclear
    - Only delegate clearly defined sub-tasks
    """

    ORCHESTRATOR_ROLE = OrchestratorRole.SUB
    PROVIDER = Provider.CLAUDE

    async def process_request(
        self,
        request: str,
        *,
        context: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> OrchestratorResponse:
        """Process request with clarity-first approach.

        Claude always checks clarity before proceeding.
        Unclear tasks are flagged for escalation.
        """
        self._state.session_id = session_id
        self._log_message(request, MessageType.TASK_ASSIGNMENT)

        # Create task for clarity assessment
        task = Task(
            task_id="clarity-check",
            objective=request,
            context=context or "",
            constraints=[],
            expected_output=None,
            clarity_level=0.0,
            source="claude_orchestrator",
        )

        # Check clarity first (Claude's required behavior)
        if self._clarity_gate:
            assessment = await self._clarity_gate.assess(task)

            if assessment.level == ClarityLevel.AMBIGUOUS:
                return OrchestratorResponse(
                    task_type=TaskType.UNCLEAR,
                    tasks=[task],
                    needs_clarification=True,
                    clarification_questions=assessment.self_questions,
                )

            if assessment.recommendation == "escalate":
                return OrchestratorResponse(
                    task_type=TaskType.UNCLEAR,
                    tasks=[task],
                    needs_clarification=True,
                    clarification_questions=assessment.self_questions
                    or ["Please clarify the task requirements"],
                )

        # Task is clear enough - process it
        prompt = self._build_process_prompt(request, context)

        try:
            llm = self._get_own_provider()
            result = await llm.run(prompt, session_id=session_id)
            response = self._parse_process_response(result.text, request)
            self._log_message(result.text, MessageType.TASK_RESULT)
            return response

        except Exception as e:
            logger.error(f"Failed to process request: {e}")
            return OrchestratorResponse(
                task_type=TaskType.CODE,
                tasks=[task],
            )

    def _build_process_prompt(self, request: str, context: Optional[str]) -> str:
        """Build prompt for code task processing."""
        parts = [
            CLAUDE_SYSTEM_PROMPT,
            "",
            "---",
            "",
            "TASK:",
            request,
        ]

        if context:
            parts.extend(["", "CONTEXT:", context])

        parts.extend(
            [
                "",
                "Analyze this code task and respond with JSON:",
                "{",
                '  "task_type": "code",',
                '  "subtasks": [',
                "    {",
                '      "task_id": "subtask-1",',
                '      "objective": "specific subtask",',
                '      "can_delegate": true/false',
                "    }",
                "  ],",
                '  "implementation_approach": "brief approach description"',
                "}",
            ]
        )

        return "\n".join(parts)

    def _parse_process_response(
        self, text: str, original_request: str
    ) -> OrchestratorResponse:
        """Parse Claude's response."""
        import json

        try:
            data = json.loads(extract_json(text))

            tasks = []
            for st in data.get("subtasks", []):
                tasks.append(
                    Task(
                        task_id=st.get("task_id", "auto"),
                        objective=st.get("objective", ""),
                        context=data.get("implementation_approach", ""),
                        constraints=[],
                        expected_output=None,
                        clarity_level=0.9 if st.get("can_delegate", False) else 0.7,
                        source="claude_orchestrator",
                    )
                )

            if not tasks:
                tasks = [
                    Task(
                        task_id="main-task",
                        objective=original_request,
                        context="",
                        constraints=[],
                        expected_output=None,
                        clarity_level=0.8,
                        source="claude_orchestrator",
                    )
                ]

            return OrchestratorResponse(
                task_type=TaskType.CODE,
                tasks=tasks,
            )

        except (json.JSONDecodeError, ValueError):
            return OrchestratorResponse(
                task_type=TaskType.CODE,
                tasks=[
                    Task(
                        task_id="main-task",
                        objective=original_request,
                        context="",
                        constraints=[],
                        expected_output=None,
                        clarity_level=0.7,
                        source="claude_orchestrator",
                    )
                ],
            )

    async def route_task(
        self,
        task: Task,
        *,
        session_id: Optional[str] = None,
    ) -> TaskRouting:
        """Route task - Claude routes code tasks to itself or escalates."""
        # Check if task is clear enough for execution
        if self._clarity_gate:
            assessment = await self._clarity_gate.assess(task)

            if assessment.level == ClarityLevel.AMBIGUOUS:
                # Escalate unclear tasks
                return TaskRouting(
                    target_worker=Provider.GEMINI,
                    worker_role=WorkerRole.STRATEGIST,
                    task=task,
                    instructions="Task needs clarification before execution",
                    delegation_allowed=False,
                    scope_limitations=assessment.self_questions,
                )

        # Clear code task - execute it
        return TaskRouting(
            target_worker=Provider.CLAUDE,
            worker_role=WorkerRole.CODE_IMPLEMENTER,
            task=task,
            instructions="Execute code task. Escalate if uncertainties arise.",
            delegation_allowed=task.clarity_level >= 0.8,
        )


# =============================================================================
# CODEX ORCHESTRATOR (Analysis Sub-Orchestrator)
# =============================================================================

CODEX_SYSTEM_PROMPT = """You are a Deep Analysis sub-orchestrator.

Your strengths:
- Academic and theoretical knowledge
- Mathematical reasoning
- Forward-thinking analysis
- Debugging and risk assessment
- Large context handling

Your limitations:
- May dive too deep into analysis
- Can over-elaborate even when unnecessary
- Results may need filtering

Your responsibilities:
1. Provide deep analysis within defined scope
2. Identify potential risks and edge cases
3. Suggest debugging strategies
4. Stay within scope limitations

CRITICAL: Stay focused on the defined scope. Avoid excessive elaboration.
"""


class CodexOrchestrator(BaseOrchestrator):
    """Codex-based analysis sub-orchestrator.

    Codex excels at:
    - Deep theoretical analysis
    - Risk assessment
    - Debugging strategies
    - Academic knowledge application

    Codex should:
    - Stay within defined scope
    - Filter excessive analysis
    - Focus on actionable insights
    """

    ORCHESTRATOR_ROLE = OrchestratorRole.SUB
    PROVIDER = Provider.CODEX

    async def process_request(
        self,
        request: str,
        *,
        context: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> OrchestratorResponse:
        """Process analysis request with scope awareness.

        Codex provides deep analysis but stays within scope.
        """
        self._state.session_id = session_id
        self._log_message(request, MessageType.TASK_ASSIGNMENT)

        prompt = self._build_process_prompt(request, context)

        try:
            llm = self._get_own_provider()
            result = await llm.run(prompt, session_id=session_id)
            response = self._parse_process_response(result.text, request)
            self._log_message(result.text, MessageType.TASK_RESULT)
            return response

        except Exception as e:
            logger.error(f"Failed to process request: {e}")
            return OrchestratorResponse(
                task_type=TaskType.ANALYSIS,
                tasks=[
                    Task(
                        task_id="analysis-fallback",
                        objective=request,
                        context=context or "",
                        constraints=["Focus on essential analysis only"],
                        expected_output=None,
                        clarity_level=0.6,
                        source="codex_fallback",
                    )
                ],
            )

    def _build_process_prompt(self, request: str, context: Optional[str]) -> str:
        """Build prompt for analysis task processing."""
        parts = [
            CODEX_SYSTEM_PROMPT,
            "",
            "---",
            "",
            "ANALYSIS REQUEST:",
            request,
        ]

        if context:
            parts.extend(["", "CONTEXT:", context])

        parts.extend(
            [
                "",
                "Provide focused analysis and respond with JSON:",
                "{",
                '  "task_type": "analysis",',
                '  "analysis_areas": [',
                "    {",
                '      "area": "specific analysis area",',
                '      "priority": "high|medium|low",',
                '      "scope_limit": "boundary for this analysis"',
                "    }",
                "  ],",
                '  "risks_identified": ["risk1", ...],',
                '  "recommendations": ["recommendation1", ...]',
                "}",
            ]
        )

        return "\n".join(parts)

    def _parse_process_response(
        self, text: str, original_request: str
    ) -> OrchestratorResponse:
        """Parse Codex's response."""
        import json

        try:
            data = json.loads(extract_json(text))

            tasks = []
            for area in data.get("analysis_areas", []):
                tasks.append(
                    Task(
                        task_id=f"analysis-{area.get('area', 'unknown')[:20]}",
                        objective=area.get("area", ""),
                        context=area.get("scope_limit", ""),
                        constraints=[area.get("scope_limit", "Stay focused")],
                        expected_output=None,
                        clarity_level=0.7,
                        source="codex_orchestrator",
                    )
                )

            if not tasks:
                tasks = [
                    Task(
                        task_id="main-analysis",
                        objective=original_request,
                        context="",
                        constraints=["Focus on essential analysis"],
                        expected_output=None,
                        clarity_level=0.7,
                        source="codex_orchestrator",
                    )
                ]

            return OrchestratorResponse(
                task_type=TaskType.ANALYSIS,
                tasks=tasks,
                direct_response="\n".join(
                    [
                        "Risks: " + ", ".join(data.get("risks_identified", [])),
                        "Recommendations: "
                        + ", ".join(data.get("recommendations", [])),
                    ]
                ),
            )

        except (json.JSONDecodeError, ValueError):
            return OrchestratorResponse(
                task_type=TaskType.ANALYSIS,
                tasks=[
                    Task(
                        task_id="main-analysis",
                        objective=original_request,
                        context="",
                        constraints=["Focus on essential analysis"],
                        expected_output=None,
                        clarity_level=0.7,
                        source="codex_orchestrator",
                    )
                ],
            )

    async def route_task(
        self,
        task: Task,
        *,
        session_id: Optional[str] = None,
    ) -> TaskRouting:
        """Route task - Codex handles analysis or routes to appropriate worker."""
        objective_lower = task.objective.lower()

        # Analysis tasks stay with Codex
        if any(
            kw in objective_lower
            for kw in ["analyze", "debug", "risk", "verify", "review", "assess"]
        ):
            return TaskRouting(
                target_worker=Provider.CODEX,
                worker_role=WorkerRole.DEEP_ANALYZER,
                task=task,
                instructions="Analyze within scope. Filter excessive details.",
                delegation_allowed=False,
                scope_limitations=task.constraints or ["Stay within defined scope"],
            )

        # Code tasks go to Claude
        if any(
            kw in objective_lower
            for kw in ["implement", "code", "write", "fix", "refactor"]
        ):
            return TaskRouting(
                target_worker=Provider.CLAUDE,
                worker_role=WorkerRole.CODE_IMPLEMENTER,
                task=task,
                instructions="Implement based on analysis findings.",
                delegation_allowed=True,
            )

        # Default: keep for analysis
        return TaskRouting(
            target_worker=Provider.CODEX,
            worker_role=WorkerRole.DEEP_ANALYZER,
            task=task,
            instructions="Provide focused analysis.",
            delegation_allowed=False,
            scope_limitations=["Essential analysis only"],
        )


# =============================================================================
# ORCHESTRATOR FACTORY
# =============================================================================


class OrchestratorFactory:
    """Factory for creating orchestrators based on provider type.

    Example:
        >>> providers = {
        ...     Provider.GEMINI: gemini_provider,
        ...     Provider.CLAUDE: claude_provider,
        ...     Provider.CODEX: codex_provider,
        ... }
        >>> factory = OrchestratorFactory(providers)
        >>> master = factory.create_master()  # Returns GeminiOrchestrator
        >>> code_sub = factory.create_sub(Provider.CLAUDE)  # Returns ClaudeOrchestrator
    """

    def __init__(
        self,
        providers: dict[Provider, LLMProvider],
        default_config: Optional[SessionConfig] = None,
    ):
        """Initialize factory.

        Args:
            providers: Dict mapping Provider enum to LLM provider instances
            default_config: Optional default session configuration
        """
        self.providers = providers
        self.default_config = default_config  # None is valid, orchestrator will create its own

    def create_master(
        self,
        provider: Provider = Provider.GEMINI,
        config: Optional[SessionConfig] = None,
    ) -> BaseOrchestrator:
        """Create a master orchestrator.

        Args:
            provider: Provider to use as master (default: Gemini)
            config: Optional session configuration

        Returns:
            Orchestrator instance configured as master
        """
        cfg = config or self.default_config

        orchestrator_map = {
            Provider.GEMINI: GeminiOrchestrator,
            Provider.CLAUDE: ClaudeOrchestrator,
            Provider.CODEX: CodexOrchestrator,
        }

        orchestrator_class = orchestrator_map.get(provider, GeminiOrchestrator)
        return orchestrator_class(self.providers, cfg)

    def create_sub(
        self,
        provider: Provider,
        config: Optional[SessionConfig] = None,
    ) -> BaseOrchestrator:
        """Create a sub-orchestrator.

        Args:
            provider: Provider for the sub-orchestrator
            config: Optional session configuration

        Returns:
            Orchestrator instance configured as sub-orchestrator
        """
        cfg = config or self.default_config

        orchestrator_map = {
            Provider.GEMINI: GeminiOrchestrator,
            Provider.CLAUDE: ClaudeOrchestrator,
            Provider.CODEX: CodexOrchestrator,
        }

        if provider not in orchestrator_map:
            raise ValueError(f"No orchestrator available for provider: {provider}")

        return orchestrator_map[provider](self.providers, cfg)

    def create_all(
        self,
        config: Optional[SessionConfig] = None,
    ) -> dict[Provider, BaseOrchestrator]:
        """Create all available orchestrators.

        Args:
            config: Optional session configuration

        Returns:
            Dict mapping Provider to orchestrator instance
        """
        result = {}
        for provider in self.providers.keys():
            try:
                result[provider] = self.create_sub(provider, config)
            except ValueError:
                logger.warning(f"Could not create orchestrator for {provider}")
        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def create_default_orchestrator(
    providers: dict[Provider, LLMProvider],
    config: Optional[SessionConfig] = None,
) -> GeminiOrchestrator:
    """Create the default Gemini-based master orchestrator.

    Args:
        providers: Dict mapping Provider enum to LLM provider instances
        config: Optional session configuration

    Returns:
        GeminiOrchestrator configured as master
    """
    return GeminiOrchestrator(providers, config)


def get_specialization_knowledge() -> str:
    """Get the specialization knowledge for prompt injection.

    Returns:
        SPECIALIZATION_KNOWLEDGE string for system prompts
    """
    return SPECIALIZATION_KNOWLEDGE
