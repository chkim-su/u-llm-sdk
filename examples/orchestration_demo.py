"""Orchestration Demo - Contract-Based Parallel Execution.

This example demonstrates the orchestration framework for parallel code editing.
"""

import asyncio
import tempfile
from pathlib import Path

from u_llm_sdk.types import Provider


def example_work_order_creation():
    """Example: Creating WorkOrder contracts."""
    from u_llm_sdk.llm.orchestration import (
        EvidenceRequirement,
        ExpectedDelta,
        WorkOrder,
    )

    print("=== WorkOrder Creation Example ===")

    # Simple WorkOrder
    wo_simple = WorkOrder(
        id="WO-001",
        objective="Add input validation to user form",
        file_set=["src/forms/*.py"],
    )
    print(f"Simple WorkOrder: {wo_simple.id} - {wo_simple.objective}")

    # Complex WorkOrder with constraints
    wo_complex = WorkOrder(
        id="WO-002",
        objective="Implement user authentication endpoint",
        file_set=["src/auth/**/*.py", "src/api/routes.py"],
        create_files=["src/auth/jwt.py"],
        constraints=[
            "Use only stdlib for JWT",
            "No external HTTP calls",
            "Must include unit tests",
        ],
        resource_locks=["api:auth", "db:users"],
        expected_delta=ExpectedDelta(
            max_files_modified=5,
            forbid_new_deps=["requests", "httpx"],
            forbid_new_public_exports=False,
        ),
        evidence_required=EvidenceRequirement(
            tests=["pytest tests/auth/"],
            typecheck=True,
            lint=True,
        ),
        priority=1,
    )

    print(f"\nComplex WorkOrder:")
    print(f"  ID: {wo_complex.id}")
    print(f"  Objective: {wo_complex.objective}")
    print(f"  File set: {wo_complex.file_set}")
    print(f"  Create files: {wo_complex.create_files}")
    print(f"  Constraints: {wo_complex.constraints}")
    print(f"  Resource locks: {wo_complex.resource_locks}")
    print(f"  Priority: {wo_complex.priority}")

    # Convert to prompt
    print(f"\n  Prompt preview:\n{wo_complex.to_prompt()[:300]}...")


def example_evidence_collection():
    """Example: Creating and checking Evidence."""
    from u_llm_sdk.llm.orchestration import Evidence

    print("\n=== Evidence Collection Example ===")

    # Successful evidence
    evidence_success = Evidence(
        work_order_id="WO-001",
        branch="wo/WO-001",
        diff="--- a/src/auth.py\n+++ b/src/auth.py\n...",
        files_modified=["src/auth.py"],
        files_created=["src/auth/jwt.py"],
        test_results={"pytest tests/auth/": True},
        typecheck_passed=True,
        lint_passed=True,
    )

    print("Successful Evidence:")
    print(f"  Files modified: {evidence_success.files_modified}")
    print(f"  Files created: {evidence_success.files_created}")
    print(f"  Tests passed: {all(evidence_success.test_results.values())}")
    print(f"  Typecheck: {evidence_success.typecheck_passed}")
    print(f"  Lint: {evidence_success.lint_passed}")

    # Evidence with violations
    evidence_violation = Evidence(
        work_order_id="WO-002",
        branch="wo/WO-002",
        files_modified=["src/auth.py", "src/unrelated.py"],
        file_set_violations=["src/unrelated.py"],
        delta_violations=["Modified file outside allowed scope"],
        test_results={"pytest": False},
        typecheck_passed=False,
        lint_passed=True,
    )

    print("\nEvidence with Violations:")
    print(f"  File set violations: {evidence_violation.file_set_violations}")
    print(f"  Delta violations: {evidence_violation.delta_violations}")
    print(f"  Tests passed: {all(evidence_violation.test_results.values())}")


def example_roles():
    """Example: Pre-defined roles in orchestration."""
    from u_llm_sdk.llm.orchestration import ROLES

    print("\n=== Pre-defined Roles Example ===")

    for role_name, role_spec in list(ROLES.items())[:5]:
        print(f"\n{role_name}:")
        print(f"  Name: {role_spec.name}")
        print(f"  Purpose: {role_spec.purpose[:80]}...")
        print(f"  Side effects: {role_spec.side_effects}")
        print(f"  Capabilities: {role_spec.capabilities}")


def example_execution_states():
    """Example: Understanding execution states."""
    from u_llm_sdk.llm.orchestration import ExecutionState, WorkOrderState

    print("\n=== Execution States Example ===")

    print("Orchestrator States (in order):")
    for state in ExecutionState:
        print(f"  - {state.value}")

    print("\nWorkOrder States:")
    for state in WorkOrderState:
        print(f"  - {state.value}")


def example_lock_manager():
    """Example: Resource locking."""
    from u_llm_sdk.llm.orchestration import InMemoryLockManager

    print("\n=== Lock Manager Example ===")

    async def demo_locks():
        manager = InMemoryLockManager()

        # Acquire locks
        result1 = await manager.acquire(
            ["api:auth", "db:users"],
            holder_id="WO-001",
            timeout=5.0,
        )
        print(f"WO-001 acquired locks: {result1.success}")
        print(f"  Acquired: {result1.acquired_locks}")

        # Try to acquire conflicting lock
        result2 = await manager.acquire(
            ["api:auth"],  # Already held by WO-001
            holder_id="WO-002",
            timeout=1.0,
        )
        print(f"\nWO-002 tried to acquire 'api:auth': {result2.success}")
        print(f"  Conflict with: {result2.conflicts}")

        # Release locks
        await manager.release(["api:auth", "db:users"], holder_id="WO-001")
        print(f"\nWO-001 released locks")

        # Now WO-002 can acquire
        result3 = await manager.acquire(
            ["api:auth"],
            holder_id="WO-002",
            timeout=1.0,
        )
        print(f"WO-002 acquired 'api:auth' after release: {result3.success}")

    asyncio.run(demo_locks())


def example_git_integration():
    """Example: Git integration utilities."""
    from u_llm_sdk.llm.orchestration import (
        work_order_branch_name,
        worktree_path_for_work_order,
    )

    print("\n=== Git Integration Example ===")

    wo_id = "WO-001"
    objective = "Implement user authentication"

    branch_name = work_order_branch_name(wo_id, objective)
    print(f"Generated branch name: {branch_name}")

    with tempfile.TemporaryDirectory() as repo_path:
        worktree_path = worktree_path_for_work_order(repo_path, wo_id)
        print(f"Worktree path: {worktree_path}")


def example_merge_executor_config():
    """Example: MergeExecutor configuration."""
    from u_llm_sdk.llm.orchestration import MergeExecutorConfig

    print("\n=== MergeExecutor Config Example ===")

    # Default config
    default_config = MergeExecutorConfig()
    print("Default Configuration:")
    print(f"  Base branch: {default_config.base_branch}")
    print(f"  Integration branch: {default_config.integration_branch}")
    print(f"  Max parallel: {default_config.max_parallel}")
    print(f"  Require all pass: {default_config.require_all_pass}")

    # Custom config
    custom_config = MergeExecutorConfig(
        base_branch="develop",
        integration_branch="llm/feature-auth",
        auto_push=False,
        create_pr=True,
        max_parallel=5,
        require_tests=True,
        require_typecheck=True,
        require_lint=True,
        planner_provider=Provider.CLAUDE,
        editor_provider=Provider.GEMINI,
    )

    print("\nCustom Configuration:")
    print(f"  Base branch: {custom_config.base_branch}")
    print(f"  Integration branch: {custom_config.integration_branch}")
    print(f"  Create PR: {custom_config.create_pr}")
    print(f"  Planner: {custom_config.planner_provider}")
    print(f"  Editor: {custom_config.editor_provider}")


def main():
    """Run all examples."""
    print("U-llm-sdk Orchestration Demo")
    print("=" * 50)

    example_work_order_creation()
    example_evidence_collection()
    example_roles()
    example_execution_states()
    example_lock_manager()
    example_git_integration()
    example_merge_executor_config()

    print("\n" + "=" * 50)
    print("Note: Full orchestration (StateMachineOrchestrator, MergeExecutor)")
    print("requires a git repository and actual LLM CLI tools.")
    print("See the documentation for complete usage examples.")


if __name__ == "__main__":
    main()
