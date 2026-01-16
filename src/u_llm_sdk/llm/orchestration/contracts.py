"""Contract-based Execution Framework.

Defines the core contracts for the agent orchestration system:
- RoleSpec: Agent role definition (capabilities + artifacts + side effects)
- WorkOrder: Contract document for Editors (file_set, locks, expected_delta)
- Evidence: Reproducible proof bundle attached to PRs
- ReviewReport: Supervisor's verdict with per-clause decisions

Design Philosophy: "Schema-driven, Scoped, Auditable"
- Schema-driven: Structured input/output (not string prompts)
- Scoped: cwd is a safety boundary (not just convenience)
- Auditable: Every execution has a verifiable checkpoint
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional


# =============================================================================
# Side Effects & Capabilities
# =============================================================================


class SideEffect(Enum):
    """Side effect level for agent roles."""

    NONE = "none"  # Pure analysis, no I/O
    READ = "read"  # Read-only access
    WRITE = "write"  # Modification allowed (most dangerous)


class Capability(Enum):
    """Allowed capabilities for agents."""

    WEB_SEARCH = "web_search"
    CODE_SEARCH = "code_search"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    COMMAND_EXEC = "command_exec"
    GIT_BRANCH = "git_branch"
    GIT_COMMIT = "git_commit"
    GIT_MERGE = "git_merge"  # Only Supervisor


# =============================================================================
# RoleSpec - Agent Role Definition
# =============================================================================


@dataclass
class RoleSpec:
    """Agent role specification.

    Defines what an agent CAN do and what it MUST NOT do.
    Each agent is defined by: capabilities + artifacts + side_effects.

    Attributes:
        name: Human-readable role name
        purpose: What this role does (and doesn't do)
        capabilities: Set of allowed capabilities
        side_effects: Level of side effects allowed
        artifacts_produced: List of artifact types this role produces
        forbidden: Explicit list of forbidden actions
    """

    name: str
    purpose: str
    capabilities: set[Capability]
    side_effects: SideEffect
    artifacts_produced: list[str]
    forbidden: list[str] = field(default_factory=list)

    def can(self, cap: Capability) -> bool:
        """Check if this role has a capability."""
        return cap in self.capabilities

    def allows_write(self) -> bool:
        """Check if this role can modify files."""
        return self.side_effects == SideEffect.WRITE

    def to_system_prompt(self) -> str:
        """Convert to system prompt for LLM."""
        lines = [
            f"# Role: {self.name}",
            f"\n## Purpose\n{self.purpose}",
            f"\n## Allowed Capabilities\n"
            + ", ".join(c.value for c in self.capabilities),
            f"\n## Side Effects: {self.side_effects.value}",
            f"\n## Expected Outputs\n" + ", ".join(self.artifacts_produced),
        ]
        if self.forbidden:
            lines.append(
                f"\n## FORBIDDEN (violation = immediate failure)\n"
                + "\n".join(f"- {f}" for f in self.forbidden)
            )
        return "\n".join(lines)


# Pre-defined roles
ROLES: dict[str, RoleSpec] = {
    "web_searcher": RoleSpec(
        name="Web Searcher",
        purpose="외부 정보 수집 및 근거 정리. 결론을 내리거나 설계를 확정하지 않음. "
        "산출물은 '출처가 달린 근거 묶음'이어야 함.",
        capabilities={Capability.WEB_SEARCH},
        side_effects=SideEffect.READ,
        artifacts_produced=["SourcedClaims"],
        forbidden=["코드 수정", "설계 확정", "결론 도출"],
    ),
    "code_searcher": RoleSpec(
        name="Code Searcher",
        purpose="저장소 내부 구조를 '지도'로 만듦. 무엇을 고칠지 제안하지 않고, "
        "'어디에 무엇이 있는지'와 '변경 영향이 어디로 퍼지는지'를 보여줌.",
        capabilities={Capability.CODE_SEARCH, Capability.FILE_READ},
        side_effects=SideEffect.READ,
        artifacts_produced=[
            "RepoMap",
            "DependencyGraph",
            "EntryPoints",
            "PublicSurfaceCandidates",
        ],
        forbidden=["수정 제안", "코드 변경", "설계 결정"],
    ),
    "tech_researcher": RoleSpec(
        name="Tech Researcher",
        purpose="기술 결정을 위한 대안 비교 및 트레이드오프 정리. "
        "구현을 직접 하지 않으며, 선택 기준과 위험, 비용을 명시.",
        capabilities={Capability.WEB_SEARCH, Capability.FILE_READ},
        side_effects=SideEffect.NONE,
        artifacts_produced=["ADR"],  # Architecture Decision Record
        forbidden=["구현", "코드 수정", "최종 결정"],
    ),
    "science_thinker": RoleSpec(
        name="Science-Engineering Thinker",
        purpose="논리적/수학적/시스템적 리스크를 사전에 찾아냄. "
        "의미 충돌 가능성이 높은 경계를 찾아 테스트 전략과 불변조건을 제시.",
        capabilities={Capability.FILE_READ, Capability.CODE_SEARCH},
        side_effects=SideEffect.NONE,
        artifacts_produced=["RiskNotes", "FailureModes", "TestsToAdd", "Invariants"],
        forbidden=["코드 수정", "구현"],
    ),
    "planner": RoleSpec(
        name="Task Planner",
        purpose="목표를 서브태스크로 분해하고, 병렬 실행 가능한 단위를 계산하며, "
        "각 Editor에게 겹치지 않는 file_set과 예상 변경 계약(expected_delta)을 부여.",
        capabilities={Capability.CODE_SEARCH, Capability.FILE_READ},
        side_effects=SideEffect.NONE,
        artifacts_produced=["ExecutionPlan", "WorkOrder"],
        forbidden=["코드 수정", "직접 실행", "범위 확장"],
    ),
    "editor": RoleSpec(
        name="Editor",
        purpose="WorkOrder를 '정확히 그대로' 수행. 창의적으로 범위를 넓히지 않음. "
        "file_set 바깥 수정은 금지. 결과는 PR로 제출.",
        capabilities={
            Capability.FILE_READ,
            Capability.FILE_WRITE,
            Capability.COMMAND_EXEC,
            Capability.GIT_BRANCH,
            Capability.GIT_COMMIT,
        },
        side_effects=SideEffect.WRITE,
        artifacts_produced=["PR", "Evidence", "Checkpoint"],
        forbidden=[
            "file_set 외부 수정",
            "새 public export 추가 (명시된 경우 제외)",
            "새 외부 의존성 추가",
            "범위 확장",
            "추가 개선 (별도 Proposal만 가능)",
        ],
    ),
    "supervisor": RoleSpec(
        name="Editing Supervisor",
        purpose="지시서 대비 변경이 정확한가 판정. 계약 위반 또는 예상 밖 변화 시 반려. "
        "통과 시 merge 수행.",
        capabilities={
            Capability.FILE_READ,
            Capability.CODE_SEARCH,
            Capability.GIT_MERGE,
        },
        side_effects=SideEffect.READ,  # merge is special privilege
        artifacts_produced=["ReviewReport"],
        forbidden=["코드 직접 수정", "계약 변경"],
    ),
    "aggregator": RoleSpec(
        name="Aggregator",
        purpose="모든 서브태스크 결과를 종합하여 최종 보고서 작성.",
        capabilities={Capability.FILE_READ},
        side_effects=SideEffect.NONE,
        artifacts_produced=["FinalReport", "Summary"],
        forbidden=["코드 수정"],
    ),
}


# =============================================================================
# WorkOrder - Contract for Editors
# =============================================================================


@dataclass
class ExpectedDelta:
    """Expected change scope (contract).

    Defines what changes are ALLOWED and what are FORBIDDEN.
    Supervisor uses this to validate PR changes.

    Attributes:
        forbid_new_deps: Forbidden new dependency edges
        allow_symbol_changes: Symbols allowed to change
        forbid_new_public_exports: Whether new public exports are forbidden
        max_files_modified: Maximum files that can be modified
        max_files_created: Maximum files that can be created
    """

    # Dependency constraints
    forbid_new_deps: list[dict] = field(
        default_factory=list
    )  # {"from": glob, "to": glob}

    # Public surface constraints
    allow_symbol_changes: list[str] = field(default_factory=list)
    forbid_new_public_exports: bool = True

    # File count constraints
    max_files_modified: Optional[int] = None
    max_files_created: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "forbid_new_deps": self.forbid_new_deps,
            "allow_symbol_changes": self.allow_symbol_changes,
            "forbid_new_public_exports": self.forbid_new_public_exports,
            "max_files_modified": self.max_files_modified,
            "max_files_created": self.max_files_created,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExpectedDelta":
        return cls(
            forbid_new_deps=data.get("forbid_new_deps", []),
            allow_symbol_changes=data.get("allow_symbol_changes", []),
            forbid_new_public_exports=data.get("forbid_new_public_exports", True),
            max_files_modified=data.get("max_files_modified"),
            max_files_created=data.get("max_files_created"),
        )


@dataclass
class EvidenceRequirement:
    """Evidence requirements for WorkOrder.

    Specifies what evidence must be collected and submitted with PR.
    """

    diff: bool = True
    commits_pattern: str = ""  # "message must include {id}"
    tests: list[str] = field(default_factory=list)  # ["pnpm test", "pnpm lint"]
    typecheck: bool = True
    lint: bool = True

    def to_dict(self) -> dict:
        return {
            "diff": self.diff,
            "commits_pattern": self.commits_pattern,
            "tests": self.tests,
            "typecheck": self.typecheck,
            "lint": self.lint,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvidenceRequirement":
        return cls(
            diff=data.get("diff", True),
            commits_pattern=data.get("commits_pattern", ""),
            tests=data.get("tests", []),
            typecheck=data.get("typecheck", True),
            lint=data.get("lint", True),
        )


@dataclass
class WorkOrder:
    """Contract document for Editor.

    The key to parallel editing is ensuring file_set disjoint.
    WorkOrder is not just an instruction but a CONTRACT with:
    - Allowed scope (file_set)
    - Expected changes (expected_delta)
    - Resource locks (resource_locks)

    Attributes:
        id: Unique identifier (e.g., "WO-001")
        objective: What needs to be done
        assignee_role: Role to execute (usually "editor")
        branch: Git branch name for isolation
        file_set: Glob patterns for allowed files (MUST be disjoint)
        create_files: Files that may be created
        constraints: Explicit constraints
        resource_locks: Logical resources that require exclusive access
        expected_delta: Expected change bounds
        evidence_required: Evidence requirements
        priority: Execution priority (lower = higher)
        dependencies: Other WorkOrder IDs that must complete first
    """

    id: str
    objective: str
    assignee_role: str = "editor"

    # Branch isolation
    branch: str = ""

    # File scope (disjoint guarantee is KEY)
    file_set: list[str] = field(default_factory=list)  # glob patterns
    create_files: list[str] = field(default_factory=list)  # allowed new files

    # Constraints
    constraints: list[str] = field(default_factory=list)

    # Resource locks (semantic conflict prevention)
    resource_locks: list[str] = field(default_factory=list)

    # Expected change contract
    expected_delta: Optional[ExpectedDelta] = None

    # Evidence requirements
    evidence_required: EvidenceRequirement = field(default_factory=EvidenceRequirement)

    # DAG structure
    priority: int = 5
    dependencies: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.branch:
            self.branch = f"edit/{self.id}"
        if not self.constraints:
            self.constraints = [
                "file_set 외부 수정 금지",
                "새 public export 추가 금지 (명시된 경우 제외)",
                "새 외부 의존성 추가 금지",
            ]

    def to_prompt(self) -> str:
        """Convert to prompt for Editor."""
        lines = [
            f"# WorkOrder: {self.id}",
            f"\n## Objective\n{self.objective}",
            f"\n## Branch\n{self.branch}",
            f"\n## Allowed Files (ONLY modify these)\n"
            + "\n".join(f"- `{f}`" for f in self.file_set),
        ]

        if self.create_files:
            lines.append(
                f"\n## May Create\n"
                + "\n".join(f"- `{f}`" for f in self.create_files)
            )

        lines.append(
            f"\n## Constraints (MUST follow)\n"
            + "\n".join(f"- {c}" for c in self.constraints)
        )

        if self.resource_locks:
            lines.append(f"\n## Resource Locks\n{', '.join(self.resource_locks)}")

        if self.expected_delta:
            lines.append(
                f"\n## Expected Delta\n```json\n"
                f"{json.dumps(self.expected_delta.to_dict(), indent=2)}\n```"
            )

        lines.append(
            f"\n## Required Evidence\n"
            f"- Diff: {self.evidence_required.diff}\n"
            f"- Tests: {self.evidence_required.tests}\n"
            f"- Typecheck: {self.evidence_required.typecheck}"
        )

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "objective": self.objective,
            "assignee_role": self.assignee_role,
            "branch": self.branch,
            "file_set": self.file_set,
            "create_files": self.create_files,
            "constraints": self.constraints,
            "resource_locks": self.resource_locks,
            "expected_delta": self.expected_delta.to_dict()
            if self.expected_delta
            else None,
            "evidence_required": self.evidence_required.to_dict(),
            "priority": self.priority,
            "dependencies": self.dependencies,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkOrder":
        expected_delta = None
        if data.get("expected_delta"):
            expected_delta = ExpectedDelta.from_dict(data["expected_delta"])

        evidence_req = EvidenceRequirement()
        if data.get("evidence_required"):
            evidence_req = EvidenceRequirement.from_dict(data["evidence_required"])

        return cls(
            id=data["id"],
            objective=data["objective"],
            assignee_role=data.get("assignee_role", "editor"),
            branch=data.get("branch", ""),
            file_set=data.get("file_set", []),
            create_files=data.get("create_files", []),
            constraints=data.get("constraints", []),
            resource_locks=data.get("resource_locks", []),
            expected_delta=expected_delta,
            evidence_required=evidence_req,
            priority=data.get("priority", 5),
            dependencies=data.get("dependencies", []),
        )


# =============================================================================
# Evidence - Reproducible Proof Bundle
# =============================================================================


@dataclass
class Evidence:
    """Reproducible evidence bundle attached to PR.

    Evidence is "reproducible facts", not "claims".
    Minimum: git diff, commits, CI logs, changed files.

    Attributes:
        work_order_id: Associated WorkOrder ID
        branch: Git branch name
        diff: Git diff content
        commits: List of commits with sha, message, files
        files_modified: List of modified file paths
        files_created: List of created file paths
        test_results: Test command -> pass/fail
        typecheck_passed: Whether typecheck passed
        lint_passed: Whether lint passed
        file_set_violations: Files modified outside allowed file_set
        delta_violations: Contract violations found
    """

    work_order_id: str
    branch: str

    # Git evidence
    diff: str = ""
    commits: list[dict] = field(default_factory=list)  # {"sha", "message", "files"}
    files_modified: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)

    # CI evidence
    test_results: dict[str, bool] = field(default_factory=dict)  # {"pnpm test": True}
    test_logs: dict[str, str] = field(
        default_factory=dict
    )  # {"pnpm test": "...output..."}
    typecheck_passed: bool = False
    typecheck_log: str = ""  # Typecheck command output
    lint_passed: bool = False
    lint_log: str = ""  # Lint command output

    # Contract compliance
    file_set_violations: list[str] = field(default_factory=list)
    delta_violations: list[str] = field(default_factory=list)

    # Merge results
    merged_commit: Optional[str] = None  # Commit hash after merge
    merge_conflicts: list[str] = field(default_factory=list)  # Files with conflicts

    # Context building (MED-07: structured error field)
    context_build_error: Optional[str] = None  # Error if context pack build failed

    # Timing
    duration_ms: int = 0

    def is_valid(self) -> bool:
        """Check if evidence is valid (no contract violations)."""
        return (
            not self.file_set_violations
            and not self.delta_violations
            and not self.merge_conflicts
            and self.typecheck_passed
            and self.lint_passed
            and all(self.test_results.values())
        )

    def to_checkpoint(self) -> str:
        """Convert to human-readable checkpoint."""
        status = "✓ PASS" if self.is_valid() else "✗ FAIL"
        lines = [
            f"[{status}] WorkOrder: {self.work_order_id}",
            f"Branch: {self.branch}",
            f"Files Modified: {len(self.files_modified)}",
            f"Files Created: {len(self.files_created)}",
        ]

        if self.test_results:
            passed = sum(self.test_results.values())
            total = len(self.test_results)
            lines.append(f"Tests: {passed}/{total} passed")

        lines.append(f"Typecheck: {'✓' if self.typecheck_passed else '✗'}")
        lines.append(f"Lint: {'✓' if self.lint_passed else '✗'}")

        if self.file_set_violations:
            lines.append(f"⚠️ File Set Violations: {self.file_set_violations}")

        if self.delta_violations:
            lines.append(f"⚠️ Delta Violations: {self.delta_violations}")

        if self.merged_commit:
            lines.append(f"Merged Commit: {self.merged_commit[:8]}")

        if self.merge_conflicts:
            lines.append(f"⚠️ Merge Conflicts: {self.merge_conflicts}")

        if self.duration_ms:
            lines.append(f"Duration: {self.duration_ms}ms")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "work_order_id": self.work_order_id,
            "branch": self.branch,
            "diff": self.diff[:1000] + "..." if len(self.diff) > 1000 else self.diff,
            "commits": self.commits,
            "files_modified": self.files_modified,
            "files_created": self.files_created,
            "files_deleted": self.files_deleted,
            "test_results": self.test_results,
            "test_logs": {
                k: v[:500] + "..." if len(v) > 500 else v
                for k, v in self.test_logs.items()
            },
            "typecheck_passed": self.typecheck_passed,
            "typecheck_log": self.typecheck_log[:500] + "..."
            if len(self.typecheck_log) > 500
            else self.typecheck_log,
            "lint_passed": self.lint_passed,
            "lint_log": self.lint_log[:500] + "..."
            if len(self.lint_log) > 500
            else self.lint_log,
            "file_set_violations": self.file_set_violations,
            "delta_violations": self.delta_violations,
            "merged_commit": self.merged_commit,
            "merge_conflicts": self.merge_conflicts,
            "duration_ms": self.duration_ms,
            "is_valid": self.is_valid(),
        }


# =============================================================================
# ReviewReport - Supervisor's Verdict
# =============================================================================


@dataclass
class ReviewReport:
    """Supervisor's review report.

    Contains per-clause verdict and rejection reasons.

    Attributes:
        work_order_id: Reviewed WorkOrder ID
        verdict: "approved" or "rejected"
        checks: Per-clause check results
        rejection_reasons: List of rejection reasons (per clause)
        additional_tests_required: Tests required for conditional approval
        recommendations: Optional recommendations for improvement
    """

    work_order_id: str
    verdict: Literal["approved", "rejected"]

    # Per-clause checks
    checks: dict[str, bool] = field(default_factory=dict)
    # Expected keys:
    # - "file_set_compliance": All changes within file_set
    # - "delta_compliance": Changes within expected_delta
    # - "ci_passed": All CI checks passed
    # - "commits_valid": Commit messages follow pattern
    # - "no_scope_creep": No unauthorized additions

    # Rejection details
    rejection_reasons: list[str] = field(default_factory=list)

    # Conditional approval
    additional_tests_required: list[str] = field(default_factory=list)

    # Recommendations (optional)
    recommendations: list[str] = field(default_factory=list)

    def to_summary(self) -> str:
        """Convert to human-readable summary."""
        if self.verdict == "approved":
            msg = f"✓ APPROVED: {self.work_order_id}"
            if self.additional_tests_required:
                msg += (
                    f"\n  Additional tests required: {self.additional_tests_required}"
                )
            return msg
        else:
            return f"✗ REJECTED: {self.work_order_id}\n" + "\n".join(
                f"  - {r}" for r in self.rejection_reasons
            )

    def to_dict(self) -> dict:
        return {
            "work_order_id": self.work_order_id,
            "verdict": self.verdict,
            "checks": self.checks,
            "rejection_reasons": self.rejection_reasons,
            "additional_tests_required": self.additional_tests_required,
            "recommendations": self.recommendations,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewReport":
        return cls(
            work_order_id=data["work_order_id"],
            verdict=data.get("verdict", "rejected"),
            checks=data.get("checks", {}),
            rejection_reasons=data.get("rejection_reasons", []),
            additional_tests_required=data.get("additional_tests_required", []),
            recommendations=data.get("recommendations", []),
        )


# =============================================================================
# ExecutionPlan - Overall Plan from Planner
# =============================================================================


@dataclass
class ExecutionPlan:
    """Overall execution plan containing all WorkOrders.

    Attributes:
        id: Plan identifier
        objective: Original objective
        work_orders: List of WorkOrders
        layers: DAG layers for parallel execution
        metadata: Additional plan metadata
    """

    id: str
    objective: str
    work_orders: list[WorkOrder]
    layers: list[list[str]] = field(default_factory=list)  # [[wo_id, ...], ...]
    metadata: dict = field(default_factory=dict)

    def get_work_order(self, wo_id: str) -> Optional[WorkOrder]:
        """Get WorkOrder by ID."""
        for wo in self.work_orders:
            if wo.id == wo_id:
                return wo
        return None

    def to_mermaid(self) -> str:
        """Generate Mermaid diagram for visualization."""
        lines = ["graph TD"]

        for wo in self.work_orders:
            node_label = f"{wo.id}[{wo.id}: {wo.objective[:30]}...]"
            lines.append(f"    {node_label}")

            for dep in wo.dependencies:
                lines.append(f"    {dep} --> {wo.id}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "objective": self.objective,
            "work_orders": [wo.to_dict() for wo in self.work_orders],
            "layers": self.layers,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionPlan":
        return cls(
            id=data["id"],
            objective=data["objective"],
            work_orders=[WorkOrder.from_dict(wo) for wo in data.get("work_orders", [])],
            layers=data.get("layers", []),
            metadata=data.get("metadata", {}),
        )


__all__ = [
    # Enums
    "SideEffect",
    "Capability",
    # Role definitions
    "RoleSpec",
    "ROLES",
    # WorkOrder components
    "ExpectedDelta",
    "EvidenceRequirement",
    "WorkOrder",
    # Evidence and Review
    "Evidence",
    "ReviewReport",
    # Plan
    "ExecutionPlan",
]
