"""Task classification for dynamic pipeline configuration.

Classifies user requests into task types to enable:
1. System prompt template selection
2. Plugin chain selection
3. Phase configuration adaptation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Primary task type classification."""

    # Feature development - new functionality
    NEW_FEATURE = "new_feature"

    # Refactoring - restructure without changing behavior
    REFACTORING = "refactoring"

    # Project creation - greenfield scaffolding
    PROJECT_CREATION = "project_creation"

    # Integration - external module integration or porting
    INTEGRATION = "integration"

    # Bug fix - addressing specific issues
    BUG_FIX = "bug_fix"

    # Documentation - docs, comments, README
    DOCUMENTATION = "documentation"

    # Configuration - settings, env, build config
    CONFIGURATION = "configuration"

    # Testing - test creation or modification
    TESTING = "testing"

    # Unknown - fallback to general approach
    UNKNOWN = "unknown"


@dataclass
class TaskClassification:
    """Result of task classification."""

    task_type: TaskType
    confidence: float  # 0.0 to 1.0
    subtypes: list[str] = field(default_factory=list)

    # Derived configurations
    requires_codebase_analysis: bool = True
    requires_external_analysis: bool = False
    requires_scaffolding: bool = False
    behavior_preservation_critical: bool = False

    # Recommended configurations
    recommended_template: str = "general"
    recommended_plugins: list[str] = field(default_factory=list)
    recommended_review_focus: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Set derived configurations based on task type."""
        self._configure_for_task_type()

    def _configure_for_task_type(self):
        """Configure derived settings based on task type."""
        configs = {
            TaskType.NEW_FEATURE: {
                "requires_codebase_analysis": True,
                "recommended_template": "feature_developer",
                "recommended_plugins": ["feature-dev:feature-dev"],
                "recommended_review_focus": ["functionality", "integration", "tests"],
            },
            TaskType.REFACTORING: {
                "requires_codebase_analysis": True,
                "behavior_preservation_critical": True,
                "recommended_template": "refactoring_expert",
                "recommended_plugins": ["code-explorer", "refactor-assistant"],
                "recommended_review_focus": ["behavior_preservation", "code_quality", "tests"],
            },
            TaskType.PROJECT_CREATION: {
                "requires_codebase_analysis": False,
                "requires_scaffolding": True,
                "recommended_template": "project_architect",
                "recommended_plugins": ["project-scaffolder"],
                "recommended_review_focus": ["structure", "conventions", "dependencies"],
            },
            TaskType.INTEGRATION: {
                "requires_codebase_analysis": True,
                "requires_external_analysis": True,
                "recommended_template": "integration_specialist",
                "recommended_plugins": ["code-explorer", "api-analyzer"],
                "recommended_review_focus": ["compatibility", "api_surface", "error_handling"],
            },
            TaskType.BUG_FIX: {
                "requires_codebase_analysis": True,
                "behavior_preservation_critical": True,
                "recommended_template": "bug_hunter",
                "recommended_plugins": ["debugger-assistant"],
                "recommended_review_focus": ["root_cause", "regression", "tests"],
            },
            TaskType.DOCUMENTATION: {
                "requires_codebase_analysis": True,
                "recommended_template": "documentation_writer",
                "recommended_plugins": [],
                "recommended_review_focus": ["accuracy", "completeness", "clarity"],
            },
            TaskType.TESTING: {
                "requires_codebase_analysis": True,
                "recommended_template": "test_engineer",
                "recommended_plugins": ["test-generator"],
                "recommended_review_focus": ["coverage", "edge_cases", "maintainability"],
            },
            TaskType.CONFIGURATION: {
                "requires_codebase_analysis": True,
                "recommended_template": "devops_engineer",
                "recommended_plugins": [],
                "recommended_review_focus": ["security", "compatibility", "documentation"],
            },
        }

        config = configs.get(self.task_type, {})
        for key, value in config.items():
            setattr(self, key, value)


# Keyword patterns for classification
TASK_PATTERNS = {
    TaskType.NEW_FEATURE: [
        "implement", "add", "create", "build", "develop",
        "new feature", "새 기능", "추가", "구현",
        "functionality", "capability",
    ],
    TaskType.REFACTORING: [
        "refactor", "restructure", "reorganize", "clean up",
        "리팩토링", "재구조화", "정리",
        "improve code", "simplify", "extract", "inline",
        "move", "rename", "split", "merge",
    ],
    TaskType.PROJECT_CREATION: [
        "create project", "new project", "scaffold", "bootstrap",
        "프로젝트 생성", "새 프로젝트", "초기화",
        "from scratch", "greenfield", "initialize",
    ],
    TaskType.INTEGRATION: [
        "integrate", "port", "migrate", "import",
        "통합", "이식", "마이그레이션",
        "external", "library", "module", "api",
        "connect", "adapt", "wrapper",
    ],
    TaskType.BUG_FIX: [
        "fix", "bug", "error", "issue", "problem",
        "버그", "수정", "오류", "문제",
        "broken", "doesn't work", "failing",
    ],
    TaskType.DOCUMENTATION: [
        "document", "docs", "readme", "comment",
        "문서", "주석", "설명",
        "explain", "describe", "annotate",
    ],
    TaskType.TESTING: [
        "test", "spec", "coverage", "unit test",
        "테스트", "검증",
        "e2e", "integration test", "mock",
    ],
    TaskType.CONFIGURATION: [
        "config", "setting", "environment", "env",
        "설정", "환경",
        "build", "deploy", "ci/cd", "docker",
    ],
}


def classify_task(request: str) -> TaskClassification:
    """Classify a task request into a task type.

    Uses keyword matching and heuristics. In production,
    this could be enhanced with LLM-based classification.

    Args:
        request: User's task request

    Returns:
        TaskClassification with type and configurations
    """
    request_lower = request.lower()

    # Score each task type
    scores: dict[TaskType, float] = {}

    for task_type, patterns in TASK_PATTERNS.items():
        score = 0.0
        for pattern in patterns:
            if pattern.lower() in request_lower:
                # Longer patterns are more specific
                score += len(pattern) / 10.0

        scores[task_type] = score

    # Find best match
    if not scores or max(scores.values()) == 0:
        return TaskClassification(
            task_type=TaskType.UNKNOWN,
            confidence=0.0,
        )

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # Normalize confidence (cap at 1.0)
    confidence = min(1.0, best_score / 3.0)

    # Detect subtypes
    subtypes = []
    for task_type, score in scores.items():
        if task_type != best_type and score > 0:
            subtypes.append(task_type.value)

    return TaskClassification(
        task_type=best_type,
        confidence=confidence,
        subtypes=subtypes,
    )


def classify_task_with_llm(
    request: str,
    llm_provider: Any,
) -> TaskClassification:
    """Classify task using LLM for higher accuracy.

    Args:
        request: User's task request
        llm_provider: LLM provider instance

    Returns:
        TaskClassification with type and configurations
    """
    # TODO: Implement LLM-based classification
    # For now, fall back to keyword-based
    return classify_task(request)


__all__ = [
    "TaskType",
    "TaskClassification",
    "classify_task",
    "classify_task_with_llm",
]
