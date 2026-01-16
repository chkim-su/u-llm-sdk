"""RAG integration client.

This module provides the RAGClient for integrating U-llm-sdk with MV-rag API.

The RAGClient implements the InterventionHook protocol via HTTP, allowing
U-llm-sdk to leverage MV-rag's experience learning and context injection
capabilities without tight coupling.

Architecture:
    U-llm-sdk -> RAGClient (HTTP) -> MV-rag API -> TimeCorrector

Key Components:
    - RAGClient: Main HTTP client class
    - RAGClientConfig: Configuration for the client
    - PreActionCache: Local TTL cache for reducing latency

Design Principles:
    1. Fail-open: Never block LLM execution on RAG failures
    2. Timeout enforcement: Strict 500ms timeout for pre-action
    3. Fire-and-forget: Post-action calls are non-blocking
    4. Local caching: Reduce latency for repeated prompts

Auto-Detection:
    >>> from u_llm_sdk.rag_client import is_mv_rag_available, create_rag_client_if_available
    >>>
    >>> # Check if MV-rag is available
    >>> if is_mv_rag_available():
    ...     client = create_rag_client_if_available()

Example:
    >>> from u_llm_sdk.rag_client import RAGClient, RAGClientConfig
    >>>
    >>> # Configure client
    >>> config = RAGClientConfig(
    ...     base_url="http://localhost:8000",
    ...     timeout_seconds=0.5,
    ...     fail_open=True
    ... )
    >>>
    >>> # Use client
    >>> async with RAGClient(config) as client:
    ...     # Get pre-action context
    ...     context = await client.on_pre_action(
    ...         prompt="How do I use asyncio?",
    ...         provider="claude",
    ...         model="opus-4"
    ...     )
    ...
    ...     # ... execute LLM call with injected context ...
    ...
    ...     # Send post-action feedback
    ...     await client.on_post_action(result, context)
"""

import importlib.util
import logging
from typing import Optional

from .cache import PreActionCache
from .client import RAGClient, ServerSchema, FeedbackFieldSpec, CLIENT_VERSION
from .config import RAGClientConfig

logger = logging.getLogger(__name__)

# Default MV-rag server URL
DEFAULT_MV_RAG_URL = "http://localhost:8000"


def is_mv_rag_installed() -> bool:
    """Check if MV-rag package is installed.

    Note: This only checks package installation, not if the server is running.
    Uses importlib.util.find_spec to avoid actual import (per CLAUDE.md rules).

    Returns:
        True if mv_rag package is installed
    """
    return importlib.util.find_spec("mv_rag") is not None


def is_mv_rag_running(base_url: str = DEFAULT_MV_RAG_URL, timeout: float = 0.5) -> bool:
    """Check if MV-rag server is running (synchronous).

    Args:
        base_url: MV-rag server URL
        timeout: Connection timeout in seconds

    Returns:
        True if server responds to health check
    """
    try:
        import httpx
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{base_url}/api/v1/health")
            return response.status_code == 200
    except Exception:
        return False


async def is_mv_rag_running_async(
    base_url: str = DEFAULT_MV_RAG_URL,
    timeout: float = 0.5
) -> bool:
    """Check if MV-rag server is running (async).

    Args:
        base_url: MV-rag server URL
        timeout: Connection timeout in seconds

    Returns:
        True if server responds to health check
    """
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{base_url}/api/v1/health")
            return response.status_code == 200
    except Exception:
        return False


def is_mv_rag_available(base_url: str = DEFAULT_MV_RAG_URL) -> bool:
    """Check if MV-rag is installed AND server is running.

    Args:
        base_url: MV-rag server URL

    Returns:
        True if package is installed and server responds
    """
    if not is_mv_rag_installed():
        return False
    return is_mv_rag_running(base_url)


def create_rag_client_if_available(
    base_url: str = DEFAULT_MV_RAG_URL,
    check_server: bool = True,
    **config_kwargs
) -> Optional[RAGClient]:
    """Create RAGClient if MV-rag is available.

    This is the recommended way to auto-detect and use MV-rag.

    Args:
        base_url: MV-rag server URL
        check_server: If True, verify server is running before creating client
        **config_kwargs: Additional arguments for RAGClientConfig

    Returns:
        RAGClient if available, None otherwise
    """
    if not is_mv_rag_installed():
        logger.debug("MV-rag package not installed, skipping RAG integration")
        return None

    if check_server and not is_mv_rag_running(base_url):
        logger.debug(f"MV-rag server not running at {base_url}, skipping RAG integration")
        return None

    config = RAGClientConfig(base_url=base_url, **config_kwargs)
    logger.info(f"MV-rag detected, creating RAGClient for {base_url}")
    return RAGClient(config)


__all__ = [
    # Core classes
    "RAGClient",
    "RAGClientConfig",
    "PreActionCache",
    "ServerSchema",
    "FeedbackFieldSpec",
    "CLIENT_VERSION",
    # Auto-detection utilities
    "is_mv_rag_installed",
    "is_mv_rag_running",
    "is_mv_rag_running_async",
    "is_mv_rag_available",
    "create_rag_client_if_available",
    "DEFAULT_MV_RAG_URL",
]
