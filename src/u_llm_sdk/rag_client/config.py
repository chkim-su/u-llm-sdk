"""RAGClient Configuration."""

from dataclasses import dataclass


@dataclass
class RAGClientConfig:
    """Configuration for RAGClient HTTP client.

    This configures the HTTP client that connects U-llm-sdk to MV-rag API.

    Attributes:
        base_url: MV-rag API base URL (default: http://localhost:8000)
        enabled: Whether RAG integration is enabled (default: True)
        timeout_seconds: HTTP request timeout for pre-action calls (default: 0.5)
                        Set to 500ms to meet 100ms SLO with buffer for network latency
        cache_ttl_seconds: Local cache TTL in seconds (default: 300 = 5 min)
        fail_open: Continue LLM execution if RAG fails (default: True)
                  When True, errors are logged but don't block LLM calls
        max_retries: Maximum number of retries for failed requests (default: 1)

    Example:
        >>> config = RAGClientConfig(
        ...     base_url="http://rag-server:8000",
        ...     timeout_seconds=0.3,
        ...     fail_open=True
        ... )
    """

    base_url: str = "http://localhost:8000"
    enabled: bool = True
    timeout_seconds: float = 0.5  # 500ms for pre-action (100ms SLO + buffer)
    cache_ttl_seconds: int = 300  # 5 min local cache
    fail_open: bool = True  # Continue if RAG fails
    max_retries: int = 1  # Used by tenacity retry decorator
    eviction_interval_seconds: int = 60  # Run cache eviction every 60s (0 to disable)
