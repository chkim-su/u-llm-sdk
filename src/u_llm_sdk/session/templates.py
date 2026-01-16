"""Session templates for pre-defined personas.

This module provides built-in templates for common specialized personas
that can be used across all LLM providers.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

# Directory containing template files
_TEMPLATES_DIR = Path(__file__).parent / "templates"


class SessionTemplate(Enum):
    """Built-in templates for specialized personas.

    Each template corresponds to a markdown file containing the
    system prompt for that persona.

    Example:
        >>> template = SessionTemplate.SECURITY_ANALYST
        >>> prompt = get_template_prompt(template)
        >>> print(prompt)  # Security analyst system prompt
    """

    CODE_REVIEWER = "code_reviewer"
    SECURITY_ANALYST = "security_analyst"
    PYTHON_EXPERT = "python_expert"
    TYPESCRIPT_EXPERT = "typescript_expert"
    API_DESIGNER = "api_designer"
    TEST_ENGINEER = "test_engineer"
    PERFORMANCE_ANALYST = "performance_analyst"
    DOCUMENTATION_WRITER = "documentation_writer"
    CRPG_STORYTELLER = "crpg_storyteller"
    CUSTOM_TELLER = "custom_teller"


def get_template_prompt(template: SessionTemplate) -> str:
    """Get the system prompt for a template.

    Args:
        template: The template to get the prompt for

    Returns:
        The system prompt string

    Raises:
        FileNotFoundError: If template file doesn't exist
    """
    template_path = _TEMPLATES_DIR / f"{template.value}.md"

    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    return template_path.read_text(encoding="utf-8").strip()


def list_templates() -> list[SessionTemplate]:
    """List all available templates.

    Returns:
        List of SessionTemplate enum values
    """
    return list(SessionTemplate)


def get_template_info(template: SessionTemplate) -> dict[str, str]:
    """Get information about a template.

    Args:
        template: The template to get info for

    Returns:
        Dict with name, description, and file path
    """
    template_path = _TEMPLATES_DIR / f"{template.value}.md"

    # Read first line as description (if it's a comment or heading)
    description = ""
    if template_path.exists():
        content = template_path.read_text(encoding="utf-8")
        first_line = content.split("\n")[0].strip()
        # Extract description from heading or comment
        if first_line.startswith("#"):
            description = first_line.lstrip("#").strip()
        elif first_line.startswith("<!--"):
            description = first_line.replace("<!--", "").replace("-->", "").strip()

    return {
        "name": template.name,
        "value": template.value,
        "description": description,
        "path": str(template_path),
    }


def create_custom_template(
    name: str,
    system_prompt: str,
    save_path: Optional[str] = None,
) -> str:
    """Create a custom template file.

    Args:
        name: Name for the template (used as filename)
        system_prompt: The system prompt content
        save_path: Optional custom path to save the template.
                   If not provided, saves to the templates directory.

    Returns:
        Path to the created template file

    Example:
        >>> path = create_custom_template(
        ...     "golang_expert",
        ...     "You are a Go expert specializing in concurrency patterns.",
        ... )
    """
    if save_path:
        template_path = Path(save_path)
    else:
        # Ensure valid filename
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        template_path = _TEMPLATES_DIR / f"{safe_name}.md"

    # Ensure directory exists
    template_path.parent.mkdir(parents=True, exist_ok=True)

    # Write template
    template_path.write_text(system_prompt, encoding="utf-8")

    return str(template_path)


__all__ = [
    "SessionTemplate",
    "get_template_prompt",
    "list_templates",
    "get_template_info",
    "create_custom_template",
]
