"""Evidence Requirement Execution Pipeline.

Executes tests, typecheck, and lint for WorkOrder compliance verification.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .contracts import EvidenceRequirement


@dataclass
class CommandExecutionResult:
    """Result of executing a single command.

    Attributes:
        command: The command that was executed
        success: Whether the command succeeded (exit code 0)
        exit_code: Process exit code
        stdout: Standard output
        stderr: Standard error
        duration_ms: Execution duration in milliseconds
    """

    command: str
    success: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0


@dataclass
class EvidenceExecutionResult:
    """Result of executing all evidence requirements.

    Attributes:
        test_results: Command -> pass/fail for each test
        test_logs: Command -> combined stdout/stderr
        typecheck_passed: Whether typecheck passed
        typecheck_log: Typecheck output
        lint_passed: Whether lint passed
        lint_log: Lint output
        total_duration_ms: Total execution time
    """

    test_results: dict[str, bool] = field(default_factory=dict)
    test_logs: dict[str, str] = field(default_factory=dict)
    typecheck_passed: bool = True  # Default True if not required
    typecheck_log: str = ""
    lint_passed: bool = True  # Default True if not required
    lint_log: str = ""
    total_duration_ms: int = 0


class EvidenceExecutor:
    """Executes evidence requirements (tests, typecheck, lint).

    This implements the P1 EvidenceRequirement execution pipeline.
    Each WorkOrder specifies what evidence must be collected:
    - tests: List of test commands to run (e.g., ["pytest", "npm test"])
    - typecheck: Whether to run typecheck (auto-detected command)
    - lint: Whether to run lint (auto-detected command)

    All commands are executed in the WorkOrder's working directory
    (worktree path if using worktree isolation).

    Example:
        >>> executor = EvidenceExecutor()
        >>> result = await executor.execute(
        ...     evidence_required=wo.evidence_required,
        ...     cwd="/path/to/worktree",
        ... )
        >>> print(result.test_results)
        {"pytest": True, "npm test": False}
    """

    # Auto-detected typecheck commands by project type
    TYPECHECK_COMMANDS = {
        "python": ["mypy .", "pyright"],
        "typescript": ["tsc --noEmit", "npx tsc --noEmit"],
        "javascript": [],  # No typecheck for JS
    }

    # Auto-detected lint commands by project type
    LINT_COMMANDS = {
        "python": ["ruff check .", "flake8 .", "pylint ."],
        "typescript": ["eslint .", "npx eslint ."],
        "javascript": ["eslint .", "npx eslint ."],
    }

    def __init__(self, timeout: float = 120.0):
        """Initialize evidence executor.

        Args:
            timeout: Default timeout for each command (seconds)
        """
        self.timeout = timeout

    async def execute(
        self,
        evidence_required: "EvidenceRequirement",
        cwd: str,
        project_type: Optional[str] = None,
    ) -> EvidenceExecutionResult:
        """Execute all evidence requirements.

        Args:
            evidence_required: Evidence requirements from WorkOrder
            cwd: Working directory (worktree path if using isolation)
            project_type: Project type for auto-detection (python/typescript/javascript)

        Returns:
            EvidenceExecutionResult with all command results
        """
        result = EvidenceExecutionResult()
        start_time = time.time()

        # Auto-detect project type if not specified
        if project_type is None:
            project_type = self._detect_project_type(cwd)

        # Execute test commands
        for test_cmd in evidence_required.tests:
            cmd_result = await self._run_command(test_cmd, cwd)
            result.test_results[test_cmd] = cmd_result.success
            result.test_logs[test_cmd] = self._format_log(cmd_result)

        # Execute typecheck if required
        if evidence_required.typecheck:
            typecheck_result = await self._run_typecheck(cwd, project_type)
            result.typecheck_passed = typecheck_result.success
            result.typecheck_log = self._format_log(typecheck_result)

        # Execute lint if required
        if evidence_required.lint:
            lint_result = await self._run_lint(cwd, project_type)
            result.lint_passed = lint_result.success
            result.lint_log = self._format_log(lint_result)

        result.total_duration_ms = int((time.time() - start_time) * 1000)
        return result

    async def _run_command(
        self,
        command: str,
        cwd: str,
        timeout: Optional[float] = None,
    ) -> CommandExecutionResult:
        """Run a single command and capture output.

        Args:
            command: Command to execute (shell string)
            cwd: Working directory
            timeout: Command timeout (default: self.timeout)

        Returns:
            CommandExecutionResult with output and status
        """
        timeout = timeout or self.timeout
        start_time = time.time()

        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            proc = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    command,
                    shell=True,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                ),
            )

            duration_ms = int((time.time() - start_time) * 1000)

            return CommandExecutionResult(
                command=command,
                success=proc.returncode == 0,
                exit_code=proc.returncode,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                duration_ms=duration_ms,
            )

        except subprocess.TimeoutExpired:
            return CommandExecutionResult(
                command=command,
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                duration_ms=int(timeout * 1000),
            )
        except Exception as e:
            return CommandExecutionResult(
                command=command,
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Failed to execute command: {e}",
                duration_ms=0,
            )

    async def _run_typecheck(
        self,
        cwd: str,
        project_type: str,
    ) -> CommandExecutionResult:
        """Run typecheck for the detected project type.

        Tries commands in order until one succeeds or all fail.
        """
        commands = self.TYPECHECK_COMMANDS.get(project_type, [])

        if not commands:
            # No typecheck available for this project type
            return CommandExecutionResult(
                command="(no typecheck)",
                success=True,
                exit_code=0,
                stdout="No typecheck command for this project type",
            )

        # Try each command until one works
        for cmd in commands:
            result = await self._run_command(cmd, cwd, timeout=180.0)
            if result.exit_code != 127:  # 127 = command not found
                return result

        # All commands not found
        return CommandExecutionResult(
            command="typecheck",
            success=True,  # Pass if no typecheck tool available
            exit_code=0,
            stdout="No typecheck tool found (mypy, pyright, tsc)",
        )

    async def _run_lint(
        self,
        cwd: str,
        project_type: str,
    ) -> CommandExecutionResult:
        """Run lint for the detected project type.

        Tries commands in order until one succeeds or all fail.
        """
        commands = self.LINT_COMMANDS.get(project_type, [])

        if not commands:
            return CommandExecutionResult(
                command="(no lint)",
                success=True,
                exit_code=0,
                stdout="No lint command for this project type",
            )

        # Try each command until one works
        for cmd in commands:
            result = await self._run_command(cmd, cwd, timeout=120.0)
            if result.exit_code != 127:  # 127 = command not found
                return result

        # All commands not found
        return CommandExecutionResult(
            command="lint",
            success=True,  # Pass if no lint tool available
            exit_code=0,
            stdout="No lint tool found (ruff, eslint, flake8)",
        )

    def _detect_project_type(self, cwd: str) -> str:
        """Auto-detect project type from files in cwd.

        Returns:
            Project type: "python", "typescript", "javascript", or "unknown"
        """
        cwd_path = Path(cwd)

        # Check for Python
        if (
            (cwd_path / "pyproject.toml").exists()
            or (cwd_path / "setup.py").exists()
            or (cwd_path / "requirements.txt").exists()
        ):
            return "python"

        # Check for TypeScript
        if (cwd_path / "tsconfig.json").exists():
            return "typescript"

        # Check for JavaScript/Node
        if (cwd_path / "package.json").exists():
            return "javascript"

        # Fallback: check file extensions
        py_files = list(cwd_path.glob("**/*.py"))[:10]
        ts_files = list(cwd_path.glob("**/*.ts"))[:10]
        js_files = list(cwd_path.glob("**/*.js"))[:10]

        if len(py_files) > len(ts_files) and len(py_files) > len(js_files):
            return "python"
        elif len(ts_files) > len(js_files):
            return "typescript"
        elif js_files:
            return "javascript"

        return "unknown"

    def _format_log(self, result: CommandExecutionResult) -> str:
        """Format command result as log string.

        Args:
            result: Command execution result

        Returns:
            Formatted log string
        """
        lines = [
            f"$ {result.command}",
            f"Exit code: {result.exit_code}",
            f"Duration: {result.duration_ms}ms",
        ]

        if result.stdout:
            lines.append(f"\n--- stdout ---\n{result.stdout[:5000]}")
        if result.stderr:
            lines.append(f"\n--- stderr ---\n{result.stderr[:5000]}")

        return "\n".join(lines)


# Global evidence executor instance
_evidence_executor: Optional[EvidenceExecutor] = None


def get_evidence_executor() -> EvidenceExecutor:
    """Get global evidence executor instance."""
    global _evidence_executor
    if _evidence_executor is None:
        _evidence_executor = EvidenceExecutor()
    return _evidence_executor


__all__ = [
    "CommandExecutionResult",
    "EvidenceExecutionResult",
    "EvidenceExecutor",
    "get_evidence_executor",
]
