"""Session management module for all providers.

This module provides unified file-based session management across
Claude, Codex, and Gemini CLI tools.

Each provider stores sessions differently:
- Claude: ~/.claude/projects/<path-key>/<uuid>.jsonl
- Codex: ~/.codex/sessions/YYYY/MM/DD/<uuid>.jsonl
- Gemini: ~/.gemini/tmp/<project-hash>/chats/<uuid>.json

Use the factory function to get the appropriate session manager:

    >>> from u_llm_sdk.session import get_session_manager
    >>> from u_llm_sdk.types import Provider
    >>>
    >>> # Get provider-specific manager
    >>> manager = get_session_manager(Provider.CLAUDE, "/my/project")
    >>>
    >>> # Create session with system context
    >>> session_id = manager.create_from_system_prompt(
    ...     "You are a security analyst.",
    ...     assistant_acknowledgment="Ready to analyze.",
    ... )
    >>>
    >>> # Resume with CLI:
    >>> # claude --resume <session_id>
    >>> # codex resume <session_id>
    >>> # gemini --resume <session_id>

Templates provide pre-defined system prompts for common personas:

    >>> from u_llm_sdk.session import SessionTemplate, get_template_prompt
    >>>
    >>> prompt = get_template_prompt(SessionTemplate.SECURITY_ANALYST)
    >>> session_id = manager.create_from_system_prompt(prompt)
"""

# Base/Factory
from u_llm_sdk.session.base import (
    BaseSessionManager,
    get_session_manager,
    inject_system_prompt,
)

# Provider implementations
from u_llm_sdk.session.claude import ClaudeSessionManager
from u_llm_sdk.session.codex import CodexSessionManager
from u_llm_sdk.session.gemini import GeminiSessionManager

# Message utilities
from u_llm_sdk.session.message import (
    SessionMessage,
    resolve_prompt,
)

# Templates
from u_llm_sdk.session.templates import (
    SessionTemplate,
    get_template_prompt,
    list_templates,
    get_template_info,
    create_custom_template,
)

__all__ = [
    # Base/Factory
    "BaseSessionManager",
    "get_session_manager",
    "inject_system_prompt",
    # Provider implementations
    "ClaudeSessionManager",
    "CodexSessionManager",
    "GeminiSessionManager",
    # Message
    "SessionMessage",
    "resolve_prompt",
    # Templates
    "SessionTemplate",
    "get_template_prompt",
    "list_templates",
    "get_template_info",
    "create_custom_template",
]
