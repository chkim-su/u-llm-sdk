"""Pipeline module for workflow execution.

This module provides a declarative workflow system for orchestrating
LLM-based pipelines using U-llm-sdk components.

Key Components:
    - WorkflowTemplate: YAML-based workflow definition
    - PipelineExecutor: Executes workflow phases
    - PhaseType: Built-in phase types (brainstorm, clarity_gate, delegation, review)

Usage:
    >>> from u_llm_sdk.pipeline import PipelineExecutor, WorkflowTemplate
    >>>
    >>> # Load pre-defined template
    >>> template = PipelineExecutor.get_template("semi_autonomous")
    >>>
    >>> # Or load custom template
    >>> template = WorkflowTemplate.from_yaml("my_workflow.yaml")
    >>>
    >>> # Execute
    >>> executor = PipelineExecutor(providers)
    >>> result = await executor.run(
    ...     template,
    ...     request="Implement user authentication",
    ...     cwd="/project",
    ... )
    >>>
    >>> if result.needs_input:
    ...     print(f"Questions: {result.pending_questions}")
    >>> else:
    ...     print(f"Success: {result.success}")

Template Format (YAML):
    name: semi_autonomous
    description: Semi-autonomous implementation pipeline
    version: "1.0.0"

    variables:
      default_timeout: 600

    boundaries:
      max_budget_usd: 2.0
      require_tests: true

    phases:
      - name: design
        type: brainstorm
        config:
          topic_template: "Design approach for: {request}"

      - name: clarity
        type: clarity_gate
        on_failure: ask_user

      - name: implementation
        type: delegation
        config:
          session_template: code_reviewer

      - name: review
        type: review
        optional: true
"""

from .types import (
    PhaseType,
    PhaseResult,
    PhaseConfig,
    WorkflowTemplate,
    PhaseOutput,
    PipelineResult,
)

from .executor import (
    PipelineExecutor,
    TEMPLATES_DIR,
)

from .chronicle_adapter import PipelineChronicleAdapter
from .report import PipelineReportWriter, PipelineReport
from .evidence_gate import (
    EvidenceGate,
    EvidenceStage,
    EvidenceResult,
    StagePolicy,
    TASK_POLICIES,
    get_delegation_evidence_context,
)

# Scribe re-exports for pipeline users
from ..scribe import (
    ScribeStore,
    ScribeType,
    ScribeStatus,
    Clearance,
    ScribeItem,
    PHASE_SECTION_MAP,
    PHASE_CLEARANCE_MAP,
    PHASE_UPDATE_MAP,
    should_escalate,
)

__all__ = [
    # Types
    "PhaseType",
    "PhaseResult",
    "PhaseConfig",
    "WorkflowTemplate",
    "PhaseOutput",
    "PipelineResult",
    # Chronicle
    "PipelineChronicleAdapter",
    # Report
    "PipelineReportWriter",
    "PipelineReport",
    # Evidence Gate
    "EvidenceGate",
    "EvidenceStage",
    "EvidenceResult",
    "StagePolicy",
    "TASK_POLICIES",
    "get_delegation_evidence_context",
    # Scribe (editable knowledge state)
    "ScribeStore",
    "ScribeType",
    "ScribeStatus",
    "Clearance",
    "ScribeItem",
    "PHASE_SECTION_MAP",
    "PHASE_CLEARANCE_MAP",
    "PHASE_UPDATE_MAP",
    "should_escalate",
    # Executor
    "PipelineExecutor",
    "TEMPLATES_DIR",
]
