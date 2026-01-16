"""System prompt generator for dynamic Claude configuration.

Combines task-type templates with codebase context to generate
appropriate system prompts for Claude execution phases.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Template directory
TEMPLATES_DIR = Path(__file__).parent.parent / "prompts" / "task_templates"


@dataclass
class TaskConfig:
    """Configuration derived from task classification."""

    task_type: str
    template_name: str
    plugins: list[str] = field(default_factory=list)
    review_focus: list[str] = field(default_factory=list)
    requires_codebase_analysis: bool = True
    behavior_critical: bool = False


# Task type to configuration mapping
# Uses atlas-orchestrator plugin agents for specialized workflows
TASK_CONFIGS = {
    "new_feature": TaskConfig(
        task_type="new_feature",
        template_name="feature_developer",
        plugins=["atlas-orchestrator:feature-implementer"],
        review_focus=["functionality", "integration", "tests"],
    ),
    "refactoring": TaskConfig(
        task_type="refactoring",
        template_name="refactoring_expert",
        plugins=["atlas-orchestrator:refactoring-expert"],
        review_focus=["behavior_preservation", "code_quality", "tests"],
        behavior_critical=True,
    ),
    "project_creation": TaskConfig(
        task_type="project_creation",
        template_name="project_architect",
        plugins=["atlas-orchestrator:project-architect"],
        review_focus=["structure", "conventions", "dependencies"],
        requires_codebase_analysis=False,
    ),
    "integration": TaskConfig(
        task_type="integration",
        template_name="integration_specialist",
        plugins=["atlas-orchestrator:integration-specialist"],
        review_focus=["compatibility", "api_surface", "error_handling"],
    ),
    "bug_fix": TaskConfig(
        task_type="bug_fix",
        template_name="feature_developer",
        plugins=["atlas-orchestrator:feature-implementer"],
        review_focus=["root_cause", "regression", "tests"],
        behavior_critical=True,
    ),
}

# Default config for unknown task types
DEFAULT_CONFIG = TaskConfig(
    task_type="unknown",
    template_name="feature_developer",
    plugins=["atlas-orchestrator:feature-implementer"],
    review_focus=["functionality", "code_quality"],
)

# Keywords for task type detection
TASK_KEYWORDS = {
    "new_feature": [
        "implement", "add", "create", "build", "develop",
        "new feature", "새 기능", "추가", "구현",
    ],
    "refactoring": [
        "refactor", "restructure", "reorganize", "clean up",
        "리팩토링", "재구조화", "정리",
        "extract", "inline", "rename", "move",
    ],
    "project_creation": [
        "create project", "new project", "scaffold", "bootstrap",
        "프로젝트 생성", "새 프로젝트", "초기화",
        "from scratch", "greenfield",
    ],
    "integration": [
        "integrate", "port", "migrate", "import",
        "통합", "이식", "마이그레이션",
        "external", "library", "module",
    ],
    "bug_fix": [
        "fix", "bug", "error", "issue",
        "버그", "수정", "오류",
        "broken", "failing",
    ],
}


def detect_task_type(request: str) -> str:
    """Detect task type from user request.

    Simple keyword-based detection. Returns task type string.
    For more accurate classification, use LLM in the design phase.

    Args:
        request: User's task request

    Returns:
        Task type string (e.g., "new_feature", "refactoring")
    """
    request_lower = request.lower()

    scores = {}
    for task_type, keywords in TASK_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in request_lower)
        scores[task_type] = score

    if not scores or max(scores.values()) == 0:
        return "unknown"

    return max(scores, key=scores.get)


def get_task_config(task_type: str) -> TaskConfig:
    """Get configuration for a task type.

    Args:
        task_type: Task type string

    Returns:
        TaskConfig with template, plugins, and settings
    """
    return TASK_CONFIGS.get(task_type, DEFAULT_CONFIG)


class SystemPromptGenerator:
    """Generates system prompts by combining templates with context.

    Example:
        >>> generator = SystemPromptGenerator()
        >>> task_type = generator.detect_task_type("Implement user auth")
        >>> config = generator.get_config(task_type)
        >>> prompt = generator.generate(
        ...     task_type,
        ...     codebase_context="Python 3.11, FastAPI project",
        ...     project_conventions="Use type hints, docstrings required",
        ... )
    """

    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize generator.

        Args:
            templates_dir: Override template directory
        """
        self.templates_dir = templates_dir or TEMPLATES_DIR
        self._template_cache: dict[str, str] = {}

    def detect_task_type(self, request: str) -> str:
        """Detect task type from request.

        Args:
            request: User's task request

        Returns:
            Task type string
        """
        return detect_task_type(request)

    def get_config(self, task_type: str) -> TaskConfig:
        """Get configuration for task type.

        Args:
            task_type: Task type string

        Returns:
            TaskConfig instance
        """
        return get_task_config(task_type)

    def load_template(self, template_name: str) -> str:
        """Load a template by name.

        Args:
            template_name: Template name (without .md extension)

        Returns:
            Template content

        Raises:
            FileNotFoundError: If template doesn't exist
        """
        if template_name in self._template_cache:
            return self._template_cache[template_name]

        template_path = self.templates_dir / f"{template_name}.md"

        if not template_path.exists():
            logger.warning(f"Template not found: {template_path}")
            # Fall back to feature_developer
            template_path = self.templates_dir / "feature_developer.md"

        if not template_path.exists():
            raise FileNotFoundError(f"No templates found in {self.templates_dir}")

        content = template_path.read_text(encoding="utf-8")
        self._template_cache[template_name] = content

        return content

    def generate(
        self,
        task_type: str,
        *,
        codebase_context: str = "",
        project_conventions: str = "",
        additional_context: Optional[dict[str, str]] = None,
    ) -> str:
        """Generate system prompt for a task type.

        Loads the appropriate template and fills in context variables.

        Args:
            task_type: Task type string
            codebase_context: Description of the codebase
            project_conventions: Project-specific conventions
            additional_context: Additional template variables

        Returns:
            Complete system prompt string
        """
        config = self.get_config(task_type)
        template = self.load_template(config.template_name)

        # Prepare context variables
        context = {
            "codebase_context": codebase_context or "(No codebase context provided)",
            "project_conventions": project_conventions or "(No specific conventions)",
            **(additional_context or {}),
        }

        # Replace template variables
        result = template
        for key, value in context.items():
            placeholder = "{" + key + "}"
            result = result.replace(placeholder, value)

        # Remove any unreplaced placeholders
        result = re.sub(r"\{[a-z_]+\}", "(Not provided)", result)

        return result

    def generate_for_request(
        self,
        request: str,
        *,
        codebase_context: str = "",
        project_conventions: str = "",
    ) -> tuple[str, TaskConfig]:
        """Generate system prompt from a user request.

        Detects task type and generates appropriate prompt.

        Args:
            request: User's task request
            codebase_context: Description of the codebase
            project_conventions: Project-specific conventions

        Returns:
            Tuple of (system_prompt, TaskConfig)
        """
        task_type = self.detect_task_type(request)
        config = self.get_config(task_type)

        prompt = self.generate(
            task_type,
            codebase_context=codebase_context,
            project_conventions=project_conventions,
        )

        return prompt, config


# Singleton instance for convenience
_generator: Optional[SystemPromptGenerator] = None


def get_generator() -> SystemPromptGenerator:
    """Get or create the singleton generator instance."""
    global _generator
    if _generator is None:
        _generator = SystemPromptGenerator()
    return _generator


__all__ = [
    "TaskConfig",
    "TASK_CONFIGS",
    "detect_task_type",
    "get_task_config",
    "SystemPromptGenerator",
    "get_generator",
]
