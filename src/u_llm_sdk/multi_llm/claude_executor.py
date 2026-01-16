"""Claude Code Executor for SEMI_AUTONOMOUS Delegation.

This module provides the executor for running Claude Code in autonomous
delegation mode. It:
- Builds delegation prompts with boundaries
- Runs Claude CLI with stream-json output
- Monitors boundary violations in real-time
- Parses events and extracts results
- Integrates with claude-only-sdk for session/template management

Usage:
    >>> from u_llm_sdk.types import ClaudeCodeDelegation, BoundaryConstraints
    >>> from u_llm_sdk.multi_llm import ClaudeCodeExecutor
    >>>
    >>> executor = ClaudeCodeExecutor(claude_provider)
    >>> outcome = await executor.execute(delegation)
    >>>
    >>> # With session template
    >>> from claude_only_sdk.session import SessionTemplate
    >>> outcome = await executor.execute(
    ...     delegation,
    ...     session_template=SessionTemplate.SECURITY_ANALYST,
    ... )
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Callable, Optional, Union

from u_llm_sdk.types import (
    BoundaryConstraints,
    ClaudeCodeDelegation,
    DelegationOutcome,
    DelegationPhase,
)

from .boundary_validation import (
    BoundaryValidator,
    BoundaryViolationError,
)

# Import claude-only-sdk utilities (no circular dependency - these don't import U-llm-sdk)
try:
    from claude_only_sdk.session import SessionManager, SessionTemplate, get_template_prompt
    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False
    SessionManager = None  # type: ignore
    SessionTemplate = None  # type: ignore
    get_template_prompt = None  # type: ignore

if TYPE_CHECKING:
    from ..llm.providers import ClaudeProvider

logger = logging.getLogger(__name__)


# =============================================================================
# Event Types
# =============================================================================


@dataclass
class StreamEvent:
    """Parsed stream-json event from Claude CLI.

    Attributes:
        event_type: Type of event (system, assistant, user, result)
        subtype: Subtype for system events (init, hook_response)
        content: Event content
        tool_use: Tool use information if applicable
        tool_result: Tool result information if applicable
        session_id: Session ID if present
        model: Model name if present
        parent_tool_use_id: Parent tool use ID for subagent events
        raw: Raw event dictionary
    """

    event_type: str
    subtype: str = ""
    content: str = ""
    tool_use: Optional[dict] = None
    tool_result: Optional[dict] = None
    session_id: Optional[str] = None
    model: Optional[str] = None
    parent_tool_use_id: Optional[str] = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> StreamEvent:
        """Parse event from stream-json dictionary."""
        event_type = data.get("type", "unknown")
        subtype = data.get("subtype", "")

        session_id = None
        if event_type == "system" and "session_id" in data:
            session_id = data["session_id"]

        tool_use = None
        message = data.get("message", {})
        content = message.get("content", []) if isinstance(message, dict) else []

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    tool_use = {
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "input": item.get("input", {}),
                    }
                    break

        tool_result = data.get("tool_use_result")

        text_content = ""
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_content = item.get("text", "")
                    break

        return cls(
            event_type=event_type,
            subtype=subtype,
            content=text_content,
            tool_use=tool_use,
            tool_result=tool_result,
            session_id=session_id,
            model=message.get("model") if isinstance(message, dict) else None,
            parent_tool_use_id=data.get("parent_tool_use_id"),
            raw=data,
        )


# =============================================================================
# Prompt Builder
# =============================================================================


def build_delegation_prompt(delegation: ClaudeCodeDelegation) -> str:
    """Build the delegation prompt with boundaries and context."""
    sections = []

    sections.append("# Autonomous Implementation Task")
    sections.append("")

    sections.append("## Objective")
    sections.append(delegation.objective)
    sections.append("")

    if delegation.design_context:
        sections.append("## Design Context")
        sections.append(delegation.design_context)
        sections.append("")

    b = delegation.boundaries
    sections.append("## LOCKED Boundaries (Must Not Exceed)")
    sections.append(f"- **Budget**: ${b.max_budget_usd:.2f} maximum")
    sections.append(f"- **Timeout**: {b.max_timeout_seconds // 60} minutes maximum")
    sections.append(f"- **Max Files**: {b.max_files_modified} files maximum")

    if b.file_scope:
        sections.append(f"- **Allowed Files**: {', '.join(b.file_scope)}")
    else:
        sections.append("- **Allowed Files**: All files in scope")

    if b.forbidden_paths:
        sections.append(f"- **Forbidden Paths**: {', '.join(b.forbidden_paths)}")

    sections.append(f"- **Tests Required**: {'Yes' if b.require_tests else 'No'}")
    sections.append(f"- **Typecheck Required**: {'Yes' if b.require_typecheck else 'No'}")
    sections.append(f"- **Shell Commands**: {'Allowed' if b.allow_shell_commands else 'Not Allowed'}")
    sections.append(f"- **Web Access**: {'Allowed' if b.allow_web_access else 'Not Allowed'}")
    sections.append("")

    o = delegation.options
    if o.suggested_approach or o.suggested_plugins or o.suggested_tools:
        sections.append("## Suggested Options (May Adjust)")

        if o.suggested_approach:
            sections.append(f"- **Approach**: {o.suggested_approach}")

        if o.suggested_plugins:
            sections.append(f"- **Plugins**: {', '.join(o.suggested_plugins)}")

        if o.suggested_tools:
            sections.append(f"- **Tools**: {', '.join(o.suggested_tools)}")

        if o.code_style_hints:
            hints = ", ".join(f"{k}: {v}" for k, v in o.code_style_hints.items())
            sections.append(f"- **Style**: {hints}")

        sections.append("")

    sections.append("## Instructions")
    sections.append("""Execute the implementation autonomously. You have full authority to:
- Create, modify, and delete files within the allowed scope
- Run commands and tests
- Make implementation decisions
- Spawn subagents for exploration or specialized tasks

Complete the task independently. Report results when done.""")

    return "\n".join(sections)


# =============================================================================
# Claude Code Executor
# =============================================================================


class ClaudeCodeExecutor:
    """Executes Claude Code in autonomous delegation mode.

    This executor runs Claude CLI with stream-json output, monitors for
    boundary violations in real-time, and captures all events for audit.

    Integrates with claude-only-sdk for:
    - Session templates (pre-defined personas like SECURITY_ANALYST)
    - System prompt injection via session files
    """

    FILE_EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
    SHELL_TOOLS = {"Bash"}
    WEB_TOOLS = {"WebFetch", "WebSearch"}

    def __init__(
        self,
        provider: ClaudeProvider,
        default_boundaries: Optional[BoundaryConstraints] = None,
        *,
        cwd: Optional[Union[str, Path]] = None,
    ):
        """Initialize executor.

        Args:
            provider: Claude provider instance
            default_boundaries: Default boundary constraints
            cwd: Working directory (for session management)
        """
        self.provider = provider
        self.default_boundaries = default_boundaries or BoundaryConstraints()
        self.cwd = Path(cwd) if cwd else Path.cwd()
        self._session_manager: Optional[SessionManager] = None

    def _get_session_manager(self) -> Optional[SessionManager]:
        """Get or create session manager (lazy initialization)."""
        if not CLAUDE_SDK_AVAILABLE:
            return None
        if self._session_manager is None:
            self._session_manager = SessionManager(self.cwd)
        return self._session_manager

    def _create_template_session(
        self,
        template: SessionTemplate,
        variables: Optional[dict[str, str]] = None,
    ) -> Optional[str]:
        """Create a session with the given template.

        Args:
            template: SessionTemplate enum value
            variables: Template variables for substitution

        Returns:
            Session ID or None if claude-only-sdk not available
        """
        manager = self._get_session_manager()
        if manager is None or get_template_prompt is None:
            logger.warning("claude-only-sdk not available, skipping template session")
            return None

        system_prompt = get_template_prompt(template, variables=variables)
        return manager.create_from_system_prompt(system_prompt)

    async def execute(
        self,
        delegation: ClaudeCodeDelegation,
        *,
        on_event: Optional[Callable[[dict], None]] = None,
        session_template: Optional[SessionTemplate] = None,
        template_variables: Optional[dict[str, str]] = None,
        system_prompt: Optional[str] = None,
    ) -> DelegationOutcome:
        """Execute the delegation and return outcome.

        Args:
            delegation: Delegation configuration with objective and boundaries
            on_event: Optional callback for each stream event
            session_template: Optional SessionTemplate for pre-seeded context
                             (e.g., SessionTemplate.SECURITY_ANALYST)
            template_variables: Variables for template substitution
            system_prompt: Optional raw system prompt to inject (takes precedence
                          over session_template if both provided). This is typically
                          generated by SystemPromptGenerator based on task type.

        Returns:
            DelegationOutcome with results and metrics
        """
        if not delegation.delegation_id:
            delegation.delegation_id = f"del-{uuid.uuid4().hex[:8]}"

        validator = BoundaryValidator(delegation.boundaries)
        prompt = build_delegation_prompt(delegation)

        # Create session with system prompt injection (priority order):
        # 1. Raw system_prompt (custom, generated by SystemPromptGenerator)
        # 2. session_template (pre-defined enum)
        # 3. delegation.session_id (fallback)
        effective_session_id = delegation.session_id

        if system_prompt is not None and CLAUDE_SDK_AVAILABLE:
            # Use raw system_prompt (takes precedence over session_template)
            manager = self._get_session_manager()
            if manager is not None:
                custom_session = manager.create_from_system_prompt(system_prompt)
                if custom_session:
                    effective_session_id = custom_session
                    logger.info(
                        f"Using custom system prompt session: {custom_session[:16]}..."
                    )
        elif session_template is not None and CLAUDE_SDK_AVAILABLE:
            # Use pre-defined template
            template_session = self._create_template_session(
                session_template,
                variables=template_variables,
            )
            if template_session:
                effective_session_id = template_session
                logger.info(
                    f"Using template session: {session_template.value} -> {template_session}"
                )

        events: list[dict] = []
        session_id: Optional[str] = None
        text_content: list[str] = []
        files_modified: set[str] = set()
        commands_run: list[str] = []
        total_cost: float = 0.0
        total_turns: int = 0
        start_time = time.time()

        try:
            async for raw_event in self._stream_execution(
                prompt,
                delegation.boundaries,
                effective_session_id,
            ):
                events.append(raw_event)
                total_turns += 1

                if on_event:
                    try:
                        on_event(raw_event)
                    except Exception as e:
                        logger.warning(f"Event callback error: {e}")

                event = StreamEvent.from_dict(raw_event)

                if event.session_id:
                    session_id = event.session_id

                if event.content:
                    text_content.append(event.content)

                if event.tool_use:
                    tool_name = event.tool_use.get("name", "")
                    tool_input = event.tool_use.get("input", {})

                    if tool_name in self.FILE_EDIT_TOOLS:
                        file_path = tool_input.get("file_path", "")
                        if file_path:
                            validator.validate_file_operation(file_path)
                            files_modified.add(file_path)

                    elif tool_name in self.SHELL_TOOLS:
                        command = tool_input.get("command", "")
                        validator.validate_shell_command(command)
                        commands_run.append(command)

                    elif tool_name in self.WEB_TOOLS:
                        url = tool_input.get("url", "")
                        validator.validate_web_access(url)

                if event.event_type == "result":
                    cost = raw_event.get("total_cost_usd", 0.0)
                    if cost > 0:
                        total_cost = cost
                        validator.record_cost(cost)

            duration_ms = int((time.time() - start_time) * 1000)

            tests_passed = None
            typecheck_passed = None

            if delegation.boundaries.require_tests and files_modified:
                tests_passed = await self._run_tests(delegation.cwd)

            if delegation.boundaries.require_typecheck and files_modified:
                typecheck_passed = await self._run_typecheck(delegation.cwd)

            summary = "\n".join(text_content[-3:]) if text_content else "No output"

            return DelegationOutcome(
                delegation_id=delegation.delegation_id,
                phase=DelegationPhase.COMPLETED,
                success=True,
                summary=summary,
                files_modified=list(files_modified),
                commands_run=commands_run,
                tests_passed=tests_passed,
                typecheck_passed=typecheck_passed,
                budget_used_usd=total_cost,
                duration_ms=duration_ms,
                total_turns=total_turns,
                session_id=session_id,
                raw_events=events,
            )

        except BoundaryViolationError as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.warning(f"Delegation {delegation.delegation_id} boundary violation: {e}")

            return DelegationOutcome.failed(
                delegation_id=delegation.delegation_id,
                error=str(e),
                phase=DelegationPhase.FAILED,
                violations=[str(e)],
            )

        except asyncio.TimeoutError:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.warning(f"Delegation {delegation.delegation_id} timed out")

            return DelegationOutcome.failed(
                delegation_id=delegation.delegation_id,
                error=f"Timeout after {delegation.boundaries.max_timeout_seconds} seconds",
                violations=[f"max_timeout_seconds: {delegation.boundaries.max_timeout_seconds}"],
            )

        except Exception as e:
            logger.exception(f"Delegation {delegation.delegation_id} failed: {e}")

            return DelegationOutcome.failed(
                delegation_id=delegation.delegation_id,
                error=str(e),
            )

    async def _stream_execution(
        self,
        prompt: str,
        boundaries: BoundaryConstraints,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[dict]:
        """Stream execution events from Claude CLI."""
        timeout = boundaries.max_timeout_seconds

        async for event in self.provider.stream(
            prompt,
            session_id=session_id,
            timeout=timeout,
        ):
            yield event

    async def _run_tests(self, cwd: str) -> bool:
        """Run tests in the working directory.

        Note: Uses asyncio.create_subprocess_exec which is the safe
        equivalent of Node.js execFile (no shell injection risk).
        """
        if not cwd:
            return True

        try:
            proc = await asyncio.create_subprocess_exec(
                "pytest", "--tb=short", "-q",
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=120)
            return proc.returncode == 0
        except Exception as e:
            logger.warning(f"Test execution failed: {e}")
            return False

    async def _run_typecheck(self, cwd: str) -> bool:
        """Run type checking in the working directory.

        Note: Uses asyncio.create_subprocess_exec which is the safe
        equivalent of Node.js execFile (no shell injection risk).
        """
        if not cwd:
            return True

        try:
            proc = await asyncio.create_subprocess_exec(
                "mypy", ".", "--ignore-missing-imports",
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=120)
            return proc.returncode == 0
        except Exception as e:
            logger.warning(f"Typecheck execution failed: {e}")
            return False
