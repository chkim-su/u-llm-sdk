"""Base Provider Abstract Class.

All provider implementations must inherit from BaseProvider.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

from u_llm_sdk.types import (
    CommandRun,
    FileChange,
    LLMResult,
    PreActionContext,
    Provider,
    ProviderNotFoundError,
    ResultType,
)

from u_llm_sdk.config import LLMConfig
from u_llm_sdk.core.discovery import get_cli_path
from u_llm_sdk.llm.providers.services import ApiKeyResolver, HookManager

if TYPE_CHECKING:
    from u_llm_sdk.llm.providers.hooks import InterventionHook

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """Abstract base class for all LLM providers.

    All providers must implement:
    - run(): Execute a prompt and return LLMResult
    - stream(): Stream execution results
    - is_available(): Check if provider CLI is installed

    Providers may optionally implement:
    - list_sessions(): List available sessions
    - get_session_info(): Get session details
    """

    # Provider identifier (set by subclasses)
    PROVIDER: Provider
    CLI_NAME: str  # CLI binary name (e.g., "claude", "codex", "gemini")

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        verify_cli: bool = True,
        intervention_hook: Optional["InterventionHook"] = None,
        validate_features: bool = True,
    ):
        """Initialize provider.

        Args:
            config: Configuration (uses defaults if None)
            verify_cli: Whether to verify CLI exists (default: True)
            intervention_hook: Optional PGSDK intervention hook for context injection.
                When provided, the provider will call on_pre_action before LLM requests
                and on_post_action after receiving results. This enables the TimeCorrector
                to inject learned patterns and collect feedback.
            validate_features: Whether to validate feature support (default: True).
                Logs warnings for unsupported features. Results are stored in
                self._validation_results for programmatic access.

        Raises:
            ProviderNotFoundError: If verify_cli=True and CLI not found
            ValueError: If cwd is invalid
        """
        self.config = config or LLMConfig(provider=self.PROVIDER)
        self._session_id: Optional[str] = None
        self._cli_path: Optional[str] = None
        self._validation_results: list = []  # FeatureValidationResult list

        # Initialize services
        intervention = intervention_hook or self.config.intervention_hook
        self._hook_manager = HookManager(intervention, self.provider_name)
        self._api_key_resolver = ApiKeyResolver(self.PROVIDER, self.config)

        # Validate cwd if specified
        if self.config.cwd:
            self._validate_cwd(self.config.cwd)

        # Validate feature support and log warnings
        if validate_features:
            self._validation_results = self.config.validate_for_provider(
                strict=False, log_warnings=True
            )

        if verify_cli:
            self._cli_path = self.get_cli_path()
            if not self._cli_path:
                raise ProviderNotFoundError(self.PROVIDER.value, self.CLI_NAME)

    def _validate_cwd(self, cwd: str) -> None:
        """Validate working directory path.

        Args:
            cwd: Working directory path

        Raises:
            ValueError: If cwd is invalid or suspicious
        """
        cwd_path = Path(cwd).resolve()

        # Must exist
        if not cwd_path.exists():
            raise ValueError(f"Working directory does not exist: {cwd}")

        # Must be a directory
        if not cwd_path.is_dir():
            raise ValueError(f"Working directory is not a directory: {cwd}")

        # Block obvious path traversal attempts
        cwd_str = str(cwd_path)
        suspicious_patterns = ["/etc/", "/var/log/", "/root/", "/.ssh/"]
        for pattern in suspicious_patterns:
            if pattern in cwd_str:
                logger.warning(
                    f"Working directory contains sensitive path pattern ({pattern}): {cwd}"
                )

    @classmethod
    def is_available(cls) -> bool:
        """Check if provider CLI is installed and accessible.

        Automatically searches common installation paths.

        Returns:
            True if CLI is available
        """
        return cls.get_cli_path() is not None

    @classmethod
    def get_cli_path(cls) -> Optional[str]:
        """Get full path to CLI binary.

        Searches PATH and common installation directories.
        Results are cached and persisted to disk for performance.

        Returns:
            Full path to CLI or None if not found
        """
        return get_cli_path(cls.PROVIDER)

    @property
    def cli_executable(self) -> str:
        """Get the CLI executable path to use.

        Returns:
            Full path to CLI if discovered, otherwise CLI_NAME
        """
        return self._cli_path or self.CLI_NAME

    # ========================================================================
    # Abstract Methods (must be implemented by subclasses)
    # ========================================================================

    @abstractmethod
    def _build_args(
        self,
        prompt: str,
        session_id: Optional[str] = None,
    ) -> list[str]:
        """Build CLI arguments for this provider.

        Args:
            prompt: Prompt to execute
            session_id: Session ID to resume

        Returns:
            List of CLI arguments
        """
        pass

    @abstractmethod
    def _parse_output(
        self,
        stdout: str,
        stderr: str,
        success: bool,
        duration_ms: int,
        **kwargs: Any,
    ) -> LLMResult:
        """Parse CLI output into LLMResult.

        Args:
            stdout: Standard output from CLI
            stderr: Standard error from CLI
            success: Whether CLI exited successfully
            duration_ms: Execution duration in milliseconds
            **kwargs: Additional provider-specific arguments

        Returns:
            LLMResult with parsed content
        """
        pass

    # ========================================================================
    # Template Method: run()
    # ========================================================================

    async def run(
        self,
        prompt: str,
        *,
        session_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> LLMResult:
        """Execute a prompt and return the result.

        This is a Template Method that handles common execution logic.
        Subclasses implement _build_args() and _parse_output() for specifics.

        Args:
            prompt: Prompt to execute
            session_id: Session ID to resume (overrides config)
            timeout: Timeout in seconds (overrides config)

        Returns:
            LLMResult with unified schema
        """
        start_time = time.monotonic()

        # Pre-action hook: Check for context injection (via HookManager)
        model: Optional[str] = None
        try:
            model = self.config.get_model()
        except Exception:
            pass
        pre_action_context = await self._hook_manager.call_pre_action(
            prompt=prompt,
            model=model,
            session_id=self._get_effective_session_id(session_id),
        )
        effective_prompt = prompt
        if pre_action_context:
            effective_prompt = self._apply_injection(prompt, pre_action_context)

        args = self._build_args(effective_prompt, session_id)
        effective_timeout = self._get_effective_timeout(timeout)

        try:
            # NOTE: asyncio.create_subprocess_exec is shell-injection safe (no shell)
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.config.cwd,
                env=self._api_key_resolver.get_env_with_api_key(),
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=effective_timeout,
                )
                stdout = stdout_bytes.decode("utf-8", errors="replace")
                stderr = stderr_bytes.decode("utf-8", errors="replace")
                exit_code = process.returncode or 0
                timed_out = False
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                stdout = ""
                stderr = ""
                exit_code = -1
                timed_out = True

        except Exception as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            result = LLMResult(
                success=False,
                result_type=ResultType.ERROR,
                provider=self.provider_name,
                model=self.config.get_model(require_explicit=False),
                text="",
                summary="Execution failed",
                error=str(e),
                duration_ms=duration_ms,
            )
            await self._hook_manager.call_post_action(result, pre_action_context)
            return result

        duration_ms = int((time.monotonic() - start_time) * 1000)

        if timed_out:
            result = LLMResult(
                success=False,
                result_type=ResultType.ERROR,
                provider=self.provider_name,
                model=self.config.get_model(require_explicit=False),
                text="",
                summary="Timeout",
                error=f"Timeout after {effective_timeout}s",
                duration_ms=duration_ms,
            )
            await self._hook_manager.call_post_action(result, pre_action_context)
            return result

        result = self._parse_output(
            stdout,
            stderr,
            exit_code == 0,
            duration_ms,
            session_id=self._get_effective_session_id(session_id),
        )

        # Post-action hook: Record feedback
        await self._hook_manager.call_post_action(result, pre_action_context)

        return result

    # ========================================================================
    # Template Method: stream()
    # ========================================================================

    async def stream(
        self,
        prompt: str,
        *,
        session_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream execution results.

        This is a Template Method that handles common streaming logic.
        Subclasses implement _build_args() for CLI argument construction.

        Args:
            prompt: Prompt to execute
            session_id: Session ID to resume
            timeout: Timeout in seconds (overrides config)

        Yields:
            Event dictionaries from CLI output
        """
        import json

        args = self._build_args(prompt, session_id)
        effective_timeout = self._get_effective_timeout(timeout)
        start_time = time.monotonic()

        # NOTE: asyncio.create_subprocess_exec is shell-injection safe (no shell)
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.config.cwd,
            env=self._api_key_resolver.get_env_with_api_key(),
        )

        try:
            while True:
                # Check timeout
                if effective_timeout:
                    elapsed = time.monotonic() - start_time
                    if elapsed > effective_timeout:
                        process.kill()
                        break

                # Read line with timeout
                try:
                    line = await asyncio.wait_for(
                        process.stdout.readline(),
                        timeout=min(1.0, effective_timeout or 1.0),
                    )
                except asyncio.TimeoutError:
                    continue

                if not line:
                    break

                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue

                try:
                    data = json.loads(line_str)

                    # Extract session ID if present
                    if "session_id" in data:
                        self._session_id = data["session_id"]

                    yield data
                except json.JSONDecodeError:
                    # Non-JSON line, yield as raw text
                    yield {"type": "text", "content": line_str}

        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    def resume(self, session_id: str) -> "BaseProvider":
        """Set session ID for conversation continuity.

        Args:
            session_id: Session ID from previous execution

        Returns:
            Self for method chaining
        """
        self._session_id = session_id
        return self

    async def parallel_run(
        self,
        prompts: list[str],
        *,
        timeout: Optional[float] = None,
    ) -> list[LLMResult]:
        """Execute multiple prompts in parallel.

        Args:
            prompts: List of prompts to execute
            timeout: Timeout per execution

        Returns:
            List of LLMResults in same order as prompts
        """
        import asyncio

        tasks = [self.run(prompt, timeout=timeout) for prompt in prompts]
        return await asyncio.gather(*tasks)

    @property
    def session_id(self) -> Optional[str]:
        """Current session ID from last execution."""
        return self._session_id

    @property
    def provider_name(self) -> str:
        """Provider name string."""
        return self.PROVIDER.value

    @property
    def validation_results(self) -> list:
        """Get feature validation results from initialization.

        Returns:
            List of FeatureValidationResult objects for any unsupported
            or caveated features. Empty list if all features are supported.

        This is useful for LLM agents to programmatically check what
        features may not work as expected.

        Example:
            >>> provider = CodexProvider(config)
            >>> for result in provider.validation_results:
            ...     if not result.supported:
            ...         print(f"Feature {result.feature.value} not supported")
            ...         print(f"Suggestion: {result.suggestion}")
        """
        return self._validation_results

    def _build_base_args(self) -> list[str]:
        """Build common CLI arguments.

        Subclasses should extend this with provider-specific args.
        Uses full CLI path if discovered.
        """
        args = [self.cli_executable]

        # Model
        try:
            model = self.config.get_model()
            if model:
                args.extend(["-m", model])
        except Exception:
            # ModelNotSpecifiedError - let subclass handle
            pass

        return args

    def _get_effective_timeout(self, timeout: Optional[float]) -> Optional[float]:
        """Get effective timeout value."""
        return timeout if timeout is not None else self.config.timeout

    def _get_effective_session_id(self, session_id: Optional[str]) -> Optional[str]:
        """Get effective session ID."""
        if session_id is not None:
            return session_id
        if self._session_id is not None:
            return self._session_id
        return self.config.session_id

    # ========================================================================
    # Result Helpers
    # ========================================================================

    def _determine_result_type(
        self,
        text_parts: list[str],
        files_modified: list[FileChange],
        commands_run: list[CommandRun],
        success: bool,
    ) -> ResultType:
        """Determine the result type based on provider outputs."""
        if not success:
            return ResultType.ERROR

        has_text = any(text_parts)
        has_files = bool(files_modified)
        has_commands = bool(commands_run)

        if has_files and has_text:
            return ResultType.MIXED
        if has_files:
            return ResultType.FILE_EDIT
        if has_commands:
            return ResultType.COMMAND
        return ResultType.TEXT

    def _build_summary(
        self,
        result_type: ResultType,
        text_parts: list[str],
        files_modified: list[FileChange],
        commands_run: list[CommandRun],
    ) -> str:
        """Build a short summary string for the LLMResult."""
        parts: list[str] = []

        if files_modified:
            file_names = [f.path.split("/")[-1] for f in files_modified]
            parts.append(
                f"{len(files_modified)} files modified: {', '.join(file_names[:3])}"
            )
            if len(file_names) > 3:
                parts[-1] += f" (+{len(file_names) - 3} more)"

        if commands_run:
            parts.append(f"{len(commands_run)} commands executed")

        if text_parts and not parts:
            # Text-only response - truncate for summary
            full_text = " ".join(text_parts)
            if len(full_text) > 100:
                parts.append(full_text[:100] + "...")
            else:
                parts.append(full_text)

        return "; ".join(parts) if parts else "Completed"

    # ========================================================================
    # Intervention Hook Support (TimeCorrector / PGSDK Integration)
    # ========================================================================

    @property
    def intervention_hook(self) -> Optional["InterventionHook"]:
        """Get the configured intervention hook."""
        return self._hook_manager._hook if self._hook_manager else None

    def _apply_injection(
        self,
        prompt: str,
        context: PreActionContext,
    ) -> str:
        """Apply pre-action context injection to the prompt.

        Args:
            prompt: Original prompt
            context: PreActionContext with injection content

        Returns:
            Modified prompt with injection applied
        """
        if not context or not context.context_text:
            return prompt

        position = context.injection_position

        if position == "prepend":
            # Add context before the prompt
            return f"{context.context_text}\n\n---\n\n{prompt}"
        elif position == "append":
            # Add context after the prompt
            return f"{prompt}\n\n---\n\n{context.context_text}"
        elif position == "system":
            # For system position, the caller should handle it differently
            # (e.g., via system_prompt config). Just prepend for now.
            return f"[System Context]\n{context.context_text}\n[/System Context]\n\n{prompt}"
        else:
            # Default to prepend
            return f"{context.context_text}\n\n{prompt}"


__all__ = ["BaseProvider"]
