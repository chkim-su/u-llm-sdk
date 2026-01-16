"""Verification System for Result Validation.

Provides multiple verification strategies:
- LLM-based verification (natural language criteria)
- Callback-based verification (programmatic checks)
- File existence verification
- Contract compliance verification (WorkOrder-based)

Verifiers answer the question: "Did the execution actually succeed?"
This goes beyond result.success to check actual outcomes.
"""

import asyncio
import fnmatch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from u_llm_sdk.types import LLMResult

from .contracts import ExpectedDelta, WorkOrder

if TYPE_CHECKING:
    from u_llm_sdk.types import ModelTier, Provider


@dataclass
class VerificationResult:
    """Result of verification.

    Attributes:
        passed: Whether verification passed
        message: Human-readable message explaining result
        details: Additional details (optional)
        checks: Individual check results (for multi-check verifiers)
    """

    passed: bool
    message: str
    details: Optional[dict] = None
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "checks": self.checks,
        }


class Verifier(ABC):
    """Base class for result verification."""

    @abstractmethod
    async def verify(
        self,
        result: LLMResult,
        context: Optional[Any] = None,
    ) -> VerificationResult:
        """Verify execution result.

        Args:
            result: LLMResult to verify
            context: Optional context (TaskContext, WorkOrder, etc.)

        Returns:
            VerificationResult with pass/fail and message
        """
        pass


class LLMVerifier(Verifier):
    """Verify using LLM with natural language criteria.

    Uses another LLM call to verify the result against
    specified criteria. Good for semantic verification.

    Example:
        >>> verifier = LLMVerifier(
        ...     criteria="Check if modified code parses without syntax errors",
        ...     provider=Provider.CLAUDE,
        ... )
        >>> result = await verifier.verify(llm_result)
    """

    def __init__(
        self,
        criteria: str,
        provider: Optional["Provider"] = None,
        model: Optional[str] = None,
    ):
        """Initialize LLM verifier.

        Args:
            criteria: Natural language verification criteria
            provider: Provider to use for verification
            model: Model to use for verification
        """
        self.criteria = criteria
        self.provider = provider
        self.model = model

    async def verify(
        self,
        result: LLMResult,
        context: Optional[Any] = None,
    ) -> VerificationResult:
        """Verify result using LLM."""
        from u_llm_sdk.types import Provider

        from u_llm_sdk.core.utils import quick_run

        provider = self.provider or Provider.CLAUDE

        # Extract files modified paths
        files_modified = []
        if result.files_modified:
            files_modified = [f.path for f in result.files_modified]

        # Extract commands run
        commands_run = []
        if result.commands_run:
            commands_run = [c.command for c in result.commands_run]

        # Build verification prompt
        prompt = f"""Review the previous action and determine if it meets the criteria.

## Action Summary
{result.summary}

## Result Type
{result.result_type}

## Modified Files
{files_modified}

## Executed Commands
{commands_run}

## Text Output (if any)
{result.text[:500] if result.text else "(none)"}

## Verification Criteria
{self.criteria}

## Instructions
Answer with EXACTLY one of:
- "PASS: <reason>" if criteria is met
- "FAIL: <reason>" if criteria is not met

Your response:"""

        verify_result = await quick_run(
            prompt,
            provider=provider,
            model=self.model,
            timeout=60.0,
        )

        # Parse response
        response_text = verify_result.text.upper()
        passed = (
            "PASS" in response_text
            and "FAIL" not in response_text.split("PASS")[0]
        )

        return VerificationResult(
            passed=passed,
            message=verify_result.text,
            details={"criteria": self.criteria},
        )


class CallbackVerifier(Verifier):
    """Verify using a Python callback function.

    Good for programmatic checks that can be expressed in code.

    Example:
        >>> verifier = CallbackVerifier(
        ...     lambda r: len(r.files_modified) > 0,
        ...     "At least one file should be modified",
        ... )
    """

    def __init__(
        self,
        callback: Callable[[LLMResult], bool],
        description: str = "Callback verification",
    ):
        """Initialize callback verifier.

        Args:
            callback: Function that takes LLMResult and returns bool
            description: Human-readable description of what's being checked
        """
        self.callback = callback
        self.description = description

    async def verify(
        self,
        result: LLMResult,
        context: Optional[Any] = None,
    ) -> VerificationResult:
        """Verify result using callback."""
        try:
            passed = self.callback(result)
            return VerificationResult(
                passed=passed,
                message=f"{self.description}: {'PASS' if passed else 'FAIL'}",
            )
        except Exception as e:
            return VerificationResult(
                passed=False,
                message=f"Verification error: {e}",
                details={"exception": str(e)},
            )


class FileExistsVerifier(Verifier):
    """Verify that expected files exist.

    Checks that all specified files exist in the filesystem.
    Useful for verifying file creation tasks.

    Example:
        >>> verifier = FileExistsVerifier(
        ...     expected_files=["src/new_module.py", "tests/test_new_module.py"],
        ...     base_path="/project",
        ... )
    """

    def __init__(
        self,
        expected_files: list[str],
        base_path: Optional[str] = None,
    ):
        """Initialize file existence verifier.

        Args:
            expected_files: List of file paths that should exist
            base_path: Base directory for relative paths
        """
        self.expected_files = expected_files
        self.base_path = Path(base_path) if base_path else Path.cwd()

    async def verify(
        self,
        result: LLMResult,
        context: Optional[Any] = None,
    ) -> VerificationResult:
        """Verify that expected files exist."""
        missing = []
        existing = []

        for file_path in self.expected_files:
            full_path = self.base_path / file_path
            if full_path.exists():
                existing.append(file_path)
            else:
                missing.append(file_path)

        passed = len(missing) == 0

        return VerificationResult(
            passed=passed,
            message=f"Files: {len(existing)}/{len(self.expected_files)} exist"
            + (f", missing: {missing}" if missing else ""),
            details={"existing": existing, "missing": missing},
            checks={f: f not in missing for f in self.expected_files},
        )


class FileSetComplianceVerifier(Verifier):
    """Verify that modifications are within allowed file_set.

    Critical for WorkOrder compliance - ensures Editor
    didn't modify files outside their assigned scope.

    Example:
        >>> verifier = FileSetComplianceVerifier(
        ...     allowed_patterns=["src/payments/**", "tests/payments/**"],
        ... )
    """

    def __init__(
        self,
        allowed_patterns: list[str],
        base_path: Optional[str] = None,
    ):
        """Initialize file set compliance verifier.

        Args:
            allowed_patterns: Glob patterns for allowed files
            base_path: Base directory for pattern matching
        """
        self.allowed_patterns = allowed_patterns
        self.base_path = base_path

    async def verify(
        self,
        result: LLMResult,
        context: Optional[Any] = None,
    ) -> VerificationResult:
        """Verify all modifications are within allowed file_set."""
        violations = []
        compliant = []

        files_modified = result.files_modified or []
        for fc in files_modified:
            file_path = fc.path
            is_allowed = self._matches_any_pattern(file_path)

            if is_allowed:
                compliant.append(file_path)
            else:
                violations.append(file_path)

        passed = len(violations) == 0

        return VerificationResult(
            passed=passed,
            message=f"File set compliance: {len(compliant)}/{len(files_modified)} within scope"
            + (f", violations: {violations}" if violations else ""),
            details={
                "compliant": compliant,
                "violations": violations,
                "allowed_patterns": self.allowed_patterns,
            },
        )

    def _matches_any_pattern(self, file_path: str) -> bool:
        """Check if file matches any allowed pattern."""
        for pattern in self.allowed_patterns:
            if fnmatch.fnmatch(file_path, pattern):
                return True
            # Also try with ** expansion
            if "**" in pattern:
                # Simple ** handling
                base_pattern = pattern.replace("**", "*")
                if fnmatch.fnmatch(file_path, base_pattern):
                    return True
        return False


class WorkOrderComplianceVerifier(Verifier):
    """Verify compliance with WorkOrder contract.

    Comprehensive verification that checks:
    - File set compliance
    - Expected delta compliance
    - Constraint compliance

    This is the main verifier for the Supervisor role.
    """

    def __init__(
        self,
        work_order: WorkOrder,
        repo_path: Optional[str] = None,
        base_ref: Optional[str] = None,
    ):
        """Initialize WorkOrder compliance verifier.

        Args:
            work_order: WorkOrder to verify against
            repo_path: Path to repository root (required for delta compliance)
            base_ref: Git ref for "before" state (default: HEAD~1)
        """
        self.work_order = work_order
        self.repo_path = repo_path
        self.base_ref = base_ref or "HEAD~1"

    async def verify(
        self,
        result: LLMResult,
        context: Optional[Any] = None,
    ) -> VerificationResult:
        """Verify compliance with WorkOrder."""
        checks = {}
        violations = []

        # Check 1: File set compliance
        file_set_verifier = FileSetComplianceVerifier(
            allowed_patterns=self.work_order.file_set + self.work_order.create_files,
        )
        file_set_result = await file_set_verifier.verify(result)
        checks["file_set_compliance"] = file_set_result.passed
        if not file_set_result.passed:
            violations.append(f"File set violation: {file_set_result.message}")

        # Check 2: Expected delta compliance (if specified)
        if self.work_order.expected_delta:
            delta_result = await self._check_delta_compliance(result)
            checks["delta_compliance"] = delta_result.passed
            if not delta_result.passed:
                violations.append(f"Delta violation: {delta_result.message}")
        else:
            checks["delta_compliance"] = True

        # Check 3: File count constraints
        if self.work_order.expected_delta:
            delta = self.work_order.expected_delta

            if delta.max_files_modified is not None:
                files_modified = result.files_modified or []
                modified_count = len(files_modified)
                checks["max_files_modified"] = (
                    modified_count <= delta.max_files_modified
                )
                if not checks["max_files_modified"]:
                    violations.append(
                        f"Too many files modified: {modified_count} > {delta.max_files_modified}"
                    )

        # Overall result
        passed = all(checks.values())

        return VerificationResult(
            passed=passed,
            message="WorkOrder compliance: " + ("PASS" if passed else "FAIL"),
            details={
                "work_order_id": self.work_order.id,
                "violations": violations,
            },
            checks=checks,
        )

    async def _check_delta_compliance(self, result: LLMResult) -> VerificationResult:
        """Check expected delta compliance using DeltaAnalyzer.

        Fail-closed: If analysis fails, returns failure (not pass).
        """
        expected = self.work_order.expected_delta
        if expected is None:
            return VerificationResult(
                passed=True,
                message="Delta compliance: no expected_delta constraint",
            )

        # Fail-closed: repo_path required for delta analysis
        if not self.repo_path:
            return VerificationResult(
                passed=False,
                message="Delta compliance FAIL: repo_path not provided (fail-closed)",
                details={"error": "Cannot analyze delta without repo_path"},
            )

        try:
            # Import here to avoid circular dependency
            from .delta_compliance import DeltaAnalyzer

            analyzer = DeltaAnalyzer(self.repo_path)

            # Get before/after snapshots
            before = analyzer.analyze_snapshot(self.base_ref)
            after = analyzer.analyze_current()

            # Check compliance
            compliance = analyzer.check_compliance(before, after, expected)

            if compliance.compliant:
                return VerificationResult(
                    passed=True,
                    message="Delta compliance: OK",
                    details={
                        "new_exports": list(compliance.new_exports),
                        "modified_symbols": list(compliance.modified_symbols),
                    },
                )
            else:
                violation_msgs = [v.description for v in compliance.violations]
                return VerificationResult(
                    passed=False,
                    message=f"Delta compliance FAIL: {len(compliance.violations)} violation(s)",
                    details={
                        "violations": [v.to_dict() for v in compliance.violations],
                        "violation_messages": violation_msgs,
                    },
                )

        except Exception as e:
            # Fail-closed: analyzer error = failure
            return VerificationResult(
                passed=False,
                message=f"Delta compliance FAIL: analyzer error - {type(e).__name__}: {e}",
                details={"error": str(e), "error_type": type(e).__name__},
            )


class CompositeVerifier(Verifier):
    """Combine multiple verifiers with AND/OR logic.

    Example:
        >>> verifier = CompositeVerifier(
        ...     verifiers=[file_verifier, callback_verifier],
        ...     mode="all",  # All must pass
        ... )
    """

    def __init__(
        self,
        verifiers: list[Verifier],
        mode: str = "all",  # "all" or "any"
    ):
        """Initialize composite verifier.

        Args:
            verifiers: List of verifiers to combine
            mode: "all" (AND) or "any" (OR)
        """
        self.verifiers = verifiers
        self.mode = mode

    async def verify(
        self,
        result: LLMResult,
        context: Optional[Any] = None,
    ) -> VerificationResult:
        """Run all verifiers and combine results."""
        results = await asyncio.gather(
            *[v.verify(result, context) for v in self.verifiers]
        )

        checks = {}
        messages = []

        for i, vr in enumerate(results):
            checks[f"verifier_{i}"] = vr.passed
            messages.append(vr.message)

        if self.mode == "all":
            passed = all(vr.passed for vr in results)
        else:  # "any"
            passed = any(vr.passed for vr in results)

        return VerificationResult(
            passed=passed,
            message=f"Composite ({self.mode}): " + "; ".join(messages),
            details={"individual_results": [vr.to_dict() for vr in results]},
            checks=checks,
        )


def normalize_verifier(
    verifier: Union[str, Verifier, Callable[[LLMResult], bool]],
    provider: Optional["Provider"] = None,
) -> Verifier:
    """Normalize various verifier inputs to Verifier instance.

    Args:
        verifier: String (LLM criteria), Verifier, or callback
        provider: Provider to use for LLM verification

    Returns:
        Verifier instance
    """
    if isinstance(verifier, Verifier):
        return verifier
    elif isinstance(verifier, str):
        return LLMVerifier(criteria=verifier, provider=provider)
    elif callable(verifier):
        return CallbackVerifier(callback=verifier)
    else:
        raise ValueError(f"Cannot convert {type(verifier)} to Verifier")


__all__ = [
    "VerificationResult",
    "Verifier",
    "LLMVerifier",
    "CallbackVerifier",
    "FileExistsVerifier",
    "FileSetComplianceVerifier",
    "WorkOrderComplianceVerifier",
    "CompositeVerifier",
    "normalize_verifier",
]
