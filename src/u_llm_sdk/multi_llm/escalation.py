"""EscalationProtocol - Upward communication for unclear tasks.

When a worker (typically Claude) encounters an unclear task, it should
escalate to the orchestrator (typically Gemini) for clarification.
This module provides the protocol and implementation for this
upward communication channel.

Escalation Types:
- clarification: Need more details about the task
- scope_definition: Need to define task boundaries
- permission: Need approval for certain actions
- guidance: Need strategic guidance on approach

The orchestrator can respond by:
- Providing direct answers
- Delegating to Codex for deep analysis
- Escalating to the user for input
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional, Protocol, Awaitable

from u_llm_sdk.types import (
    ClarityAssessment,
    EscalationRequest,
    EscalationResponse,
    Provider,
    Task,
    LLMResult,
)

from .utils import extract_json

if TYPE_CHECKING:
    from u_llm_sdk.llm.providers.base import BaseProvider

logger = logging.getLogger(__name__)


# System prompt for orchestrator to handle escalations
ESCALATION_HANDLER_PROMPT = """You are handling an escalation from a worker LLM.

The worker has indicated they need help with a task. Review their request and provide guidance.

For each specific question, provide a clear answer.
If the task needs refinement, provide a refined version.
If you need more context from the user, indicate what questions to ask.

Respond in this JSON format:
{
  "clarifications": {
    "question1": "answer1",
    "question2": "answer2"
  },
  "refined_task": {
    "task_id": "same as original",
    "objective": "refined objective if needed",
    "context": "enhanced context",
    "constraints": ["list of constraints"],
    "expected_output": "clarified expected output"
  },
  "additional_context": "Any additional context that might help",
  "permission_granted": true,
  "guidance": "Strategic guidance for the worker",
  "needs_user_input": false,
  "user_questions": []
}"""


class OrchestratorProvider(Protocol):
    """Protocol for orchestrator that handles escalations."""

    async def run(
        self,
        prompt: str,
        *,
        session_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> LLMResult: ...


# Type for user query callback
UserQueryCallback = Callable[[list[str]], Awaitable[dict[str, str]]]


class EscalationProtocol:
    """Handles escalation from workers to orchestrator.

    When a worker cannot proceed due to unclear task requirements,
    it creates an EscalationRequest and sends it through this protocol.
    The orchestrator processes the request and returns an EscalationResponse.

    Example:
        >>> protocol = EscalationProtocol(gemini_provider)
        >>> request = EscalationRequest(
        ...     source_worker=Provider.CLAUDE,
        ...     original_task=task,
        ...     clarity_assessment=assessment,
        ...     specific_questions=["What is the target API version?"],
        ...     request_type="clarification",
        ... )
        >>> response = await protocol.escalate(request)
        >>> if response.refined_task:
        ...     # Use refined task
        ... elif response.clarifications:
        ...     # Use provided clarifications

    Attributes:
        orchestrator: LLM provider acting as orchestrator
        user_query_callback: Optional callback for user queries
    """

    def __init__(
        self,
        orchestrator: OrchestratorProvider,
        *,
        user_query_callback: Optional[UserQueryCallback] = None,
    ):
        """Initialize EscalationProtocol.

        Args:
            orchestrator: LLM provider that handles escalations
            user_query_callback: Optional callback for querying the user
        """
        self.orchestrator = orchestrator
        self.user_query_callback = user_query_callback

    async def escalate(
        self,
        request: EscalationRequest,
        *,
        session_id: Optional[str] = None,
    ) -> EscalationResponse:
        """Process an escalation request.

        Sends the request to the orchestrator for processing.
        If the orchestrator determines user input is needed and
        a callback is provided, it will query the user.

        Args:
            request: The escalation request from a worker
            session_id: Optional session ID for context

        Returns:
            EscalationResponse with clarifications and/or refined task
        """
        prompt = self._build_escalation_prompt(request)

        try:
            result = await self.orchestrator.run(prompt, session_id=session_id)
            response = self._parse_response(result.text, request.original_task)

            # Check if user input is needed
            if self._needs_user_input(response) and self.user_query_callback:
                user_answers = await self.user_query_callback(
                    response.get("user_questions", [])
                )
                # Re-process with user answers
                response = await self._refine_with_user_input(
                    request, response, user_answers, session_id
                )

            return self._create_response(response, request.original_task)

        except Exception as e:
            logger.error(f"Escalation failed: {e}")
            return EscalationResponse(
                clarifications={},
                refined_task=None,
                additional_context=f"Escalation processing failed: {e}",
                permission_granted=True,  # Fail-open
                guidance="Please proceed with best judgment or ask user directly",
            )

    def _build_escalation_prompt(self, request: EscalationRequest) -> str:
        """Build prompt for orchestrator to handle escalation."""
        task = request.original_task
        assessment = request.clarity_assessment

        prompt_parts = [
            ESCALATION_HANDLER_PROMPT,
            "",
            "---",
            "",
            "ESCALATION REQUEST:",
            f"Source Worker: {request.source_worker.value}",
            f"Request Type: {request.request_type}",
            "",
            "ORIGINAL TASK:",
            f"  Task ID: {task.task_id}",
            f"  Objective: {task.objective}",
            f"  Context: {task.context}",
            f"  Constraints: {', '.join(task.constraints) if task.constraints else 'None'}",
            f"  Expected Output: {task.expected_output or 'Not specified'}",
            "",
            "CLARITY ASSESSMENT:",
            f"  Level: {assessment.level.value}",
            f"  Score: {assessment.score:.2f}",
            "",
        ]

        if assessment.unclear_aspects:
            prompt_parts.append("UNCLEAR ASPECTS:")
            for asp in assessment.unclear_aspects:
                prompt_parts.append(f"  - {asp.aspect_type}: {asp.description}")
                prompt_parts.append(f"    Needs: {asp.clarification_needed}")
            prompt_parts.append("")

        if request.specific_questions:
            prompt_parts.append("SPECIFIC QUESTIONS FROM WORKER:")
            for q in request.specific_questions:
                prompt_parts.append(f"  - {q}")
            prompt_parts.append("")

        return "\n".join(prompt_parts)

    def _parse_response(self, response_text: str, original_task: Task) -> dict:
        """Parse orchestrator response."""
        import json

        try:
            # Extract JSON from response
            json_str = extract_json(response_text)
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse escalation response: {e}")
            return {
                "clarifications": {},
                "additional_context": response_text,
                "permission_granted": True,
                "guidance": "Could not parse structured response, raw text included",
            }

    def _needs_user_input(self, response: dict) -> bool:
        """Check if response indicates user input is needed."""
        return response.get("needs_user_input", False) and response.get("user_questions")

    async def _refine_with_user_input(
        self,
        original_request: EscalationRequest,
        initial_response: dict,
        user_answers: dict[str, str],
        session_id: Optional[str],
    ) -> dict:
        """Refine response with user input."""
        # Build refinement prompt
        prompt = f"""Based on the user's answers, please refine your response.

User's answers to questions:
{self._format_user_answers(user_answers)}

Previous guidance:
{initial_response.get('guidance', '')}

Please provide updated JSON response with refined task and clarifications."""

        try:
            result = await self.orchestrator.run(prompt, session_id=session_id)
            return self._parse_response(result.text, original_request.original_task)
        except Exception as e:
            logger.warning(f"Failed to refine with user input: {e}")
            # Add user answers to initial response
            initial_response["clarifications"].update(user_answers)
            return initial_response

    def _format_user_answers(self, answers: dict[str, str]) -> str:
        """Format user answers for prompt."""
        lines = []
        for q, a in answers.items():
            lines.append(f"Q: {q}")
            lines.append(f"A: {a}")
            lines.append("")
        return "\n".join(lines)

    def _create_response(self, data: dict, original_task: Task) -> EscalationResponse:
        """Create EscalationResponse from parsed data."""
        refined_task = None
        if data.get("refined_task"):
            rt = data["refined_task"]
            refined_task = Task(
                task_id=rt.get("task_id", original_task.task_id),
                objective=rt.get("objective", original_task.objective),
                context=rt.get("context", original_task.context),
                constraints=rt.get("constraints", original_task.constraints),
                expected_output=rt.get("expected_output", original_task.expected_output),
                clarity_level=0.9,  # Refined task should be clearer
                source="orchestrator_refinement",
            )

        return EscalationResponse(
            clarifications=data.get("clarifications", {}),
            refined_task=refined_task,
            additional_context=data.get("additional_context", ""),
            permission_granted=data.get("permission_granted", True),
            guidance=data.get("guidance", ""),
        )


# Convenience function for quick escalation
async def escalate_to_orchestrator(
    task: Task,
    assessment: ClarityAssessment,
    orchestrator: OrchestratorProvider,
    questions: Optional[list[str]] = None,
    request_type: str = "clarification",
) -> EscalationResponse:
    """Quick function to escalate a task to the orchestrator.

    Args:
        task: Original task that needs clarification
        assessment: Clarity assessment that triggered escalation
        orchestrator: LLM provider acting as orchestrator
        questions: Specific questions to ask (optional)
        request_type: Type of escalation request

    Returns:
        EscalationResponse with clarifications
    """
    request = EscalationRequest(
        source_worker=Provider.CLAUDE,  # Default to Claude as worker
        original_task=task,
        clarity_assessment=assessment,
        specific_questions=questions or assessment.self_questions,
        request_type=request_type,
    )

    protocol = EscalationProtocol(orchestrator)
    return await protocol.escalate(request)
