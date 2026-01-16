"""RAGClient - HTTP client for MV-rag API integration."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from u_llm_sdk.types import (
    LLMResult,
    PreActionContext,
    PostActionFeedback,
)

from .cache import PreActionCache
from .config import RAGClientConfig

logger = logging.getLogger(__name__)

# Client version - must be >= MV-rag's min_client_version
CLIENT_VERSION = "0.1.0"


@dataclass
class FeedbackFieldSpec:
    """Specification for a feedback field requested by MV-rag."""

    name: str
    type: str
    required: bool
    description: str
    source: str  # "llm_result", "config", "context", "computed"

    @classmethod
    def from_dict(cls, data: dict) -> "FeedbackFieldSpec":
        return cls(
            name=data["name"],
            type=data.get("type", "string"),
            required=data.get("required", False),
            description=data.get("description", ""),
            source=data.get("source", "llm_result"),
        )


@dataclass
class ServerSchema:
    """MV-rag server schema information."""

    version: str
    min_client_version: str
    endpoints: list
    breaking_changes: list
    is_compatible: bool
    feedback_fields: list  # List[FeedbackFieldSpec]

    @classmethod
    def from_dict(cls, data: dict, client_version: str = CLIENT_VERSION) -> "ServerSchema":
        """Create from API response."""
        min_version = data.get("min_client_version", "0.0.0")

        # Check compatibility
        try:
            from packaging import version
            is_compatible = version.parse(client_version) >= version.parse(min_version)
        except ImportError:
            # Fallback: simple string comparison
            is_compatible = client_version >= min_version
        except Exception:
            is_compatible = True  # Fail-open

        # Parse feedback fields
        feedback_fields = [
            FeedbackFieldSpec.from_dict(f)
            for f in data.get("feedback_fields", [])
        ]

        return cls(
            version=data.get("version", "unknown"),
            min_client_version=min_version,
            endpoints=data.get("endpoints", []),
            breaking_changes=data.get("breaking_changes", []),
            is_compatible=is_compatible,
            feedback_fields=feedback_fields,
        )


class RAGClient:
    """HTTP client for MV-rag API.

    This client implements the InterventionHook protocol by calling MV-rag REST API.
    It replaces direct TimeCorrector integration with HTTP calls to the MV-rag service.

    Design Principles:
        1. Fail-open: Never block LLM execution on RAG failures
        2. Timeout enforcement: Strict 500ms timeout for pre-action
        3. Fire-and-forget: Post-action calls are non-blocking
        4. Local caching: Reduce latency for repeated prompts
        5. Graceful degradation: Continue without RAG if service unavailable

    Architecture:
        U-llm-sdk (this client) -> HTTP -> MV-rag API -> TimeCorrector

        Pre-action flow:
            1. Check local cache
            2. If miss, call POST /api/v1/pre-action
            3. Cache result
            4. Return PreActionContext or None on error

        Post-action flow:
            1. Call POST /api/v1/post-action (fire-and-forget)
            2. Log errors but don't raise

    Example:
        >>> config = RAGClientConfig(base_url="http://localhost:8000")
        >>> client = RAGClient(config)
        >>>
        >>> # Pre-action: get context injection
        >>> context = await client.on_pre_action(
        ...     prompt="How do I use asyncio?",
        ...     provider="claude",
        ...     model="opus-4"
        ... )
        >>>
        >>> # Post-action: send feedback (fire-and-forget)
        >>> await client.on_post_action(result, context, run_id="run-123")
    """

    def __init__(self, config: RAGClientConfig):
        """Initialize RAGClient.

        Args:
            config: RAGClient configuration
        """
        self._config = config
        self._cache = PreActionCache(ttl_seconds=config.cache_ttl_seconds)
        self._schema: Optional[ServerSchema] = None  # Cached schema
        self._eviction_task: Optional[asyncio.Task] = None  # Background eviction

        # Create async HTTP client with timeout
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.timeout_seconds),
            follow_redirects=True,
        )

        # Create retry decorator based on config
        self._retry = retry(
            stop=stop_after_attempt(config.max_retries + 1),
            wait=wait_exponential(multiplier=0.05, min=0.05, max=0.2),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)
            ),
            reraise=True,
        )

    async def on_pre_action(
        self,
        prompt: str,
        provider: str,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Optional[PreActionContext]:
        """Call MV-rag /api/v1/pre-action endpoint for context injection.

        This method is called before LLM execution to get context that should be
        injected into the prompt. It implements:
            1. Early return if disabled
            2. Cache check for repeated prompts
            3. HTTP call to MV-rag API
            4. Cache update on success
            5. Fail-open on errors

        Args:
            prompt: User's prompt/question
            provider: LLM provider name (e.g., "claude", "gemini")
            model: Model name (optional)
            session_id: Session ID for context (optional)
            run_id: Current run ID for tracking (optional)

        Returns:
            PreActionContext with injection data if available, None otherwise
            (Returns None on errors due to fail-open design)

        Raises:
            Never raises - all exceptions are caught and logged (fail-open)
        """
        # Early return if disabled
        if not self._config.enabled:
            logger.debug("RAGClient is disabled, skipping pre-action")
            return None

        # Check cache
        cache_key = self._cache.make_key(prompt, provider, model)
        cached_context = self._cache.get(cache_key)
        if cached_context is not None:
            logger.debug(f"Cache hit for pre-action: {cache_key[:16]}...")
            return cached_context

        # Call MV-rag API with retry
        try:
            response = await self._post_with_retry(
                "/api/v1/pre-action",
                json={
                    "prompt": prompt,
                    "provider": provider,
                    "model": model,
                    "session_id": session_id,
                    "run_id": run_id,
                },
            )
            response.raise_for_status()

            # Parse response
            data = response.json()

            # Handle empty response (no context to inject)
            if not data or not data.get("should_inject", False):
                logger.debug("No context injection from MV-rag")
                return None
            if not data.get("context_text") or not data.get("injection_id"):
                logger.warning(
                    "MV-rag returned should_inject=True but missing context/injection_id; ignoring"
                )
                return None

            # Create PreActionContext from response
            context = PreActionContext.from_dict(data)

            # Cache the result
            self._cache.set(cache_key, context)

            logger.info(
                f"Pre-action context received: confidence={context.confidence:.2f}, "
                f"tokens={context.token_count}"
            )

            return context

        except httpx.TimeoutException as e:
            # Timeout - always fail-open
            log = logger.warning if self._config.fail_open else logger.error
            log(f"RAG pre-action timeout after {self._config.timeout_seconds}s: {e}")
            return None

        except httpx.HTTPStatusError as e:
            # HTTP error - always fail-open
            log = logger.warning if self._config.fail_open else logger.error
            log(f"RAG pre-action HTTP error {e.response.status_code}: {e}")
            return None

        except Exception as e:
            # Any other error - always fail-open
            log = logger.error if self._config.fail_open else logger.critical
            log(f"RAG pre-action unexpected error: {e}", exc_info=True)
            return None

    async def on_post_action(
        self,
        result: LLMResult,
        pre_action_context: Optional[PreActionContext] = None,
        run_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Call MV-rag /api/v1/post-action endpoint (fire-and-forget).

        This method is called after LLM execution to send feedback about the result.
        It's designed as fire-and-forget: errors are logged but not raised.

        The feedback is used by MV-rag to:
            1. Update DejaVu pattern groups
            2. Adjust influence routing scores
            3. Improve future context injections

        Schema-Driven Logging:
            This method queries MV-rag's schema to discover what additional fields
            are requested, and populates them dynamically from the LLMResult.
            New logging fields can be added to MV-rag without U-llm-sdk code changes.

        Args:
            result: LLM execution result
            pre_action_context: Context that was injected (if any)
            run_id: Current run ID for tracking (optional)
            provider: LLM provider name (optional, for schema-driven logging)
            model: Model name (optional, for schema-driven logging)

        Returns:
            None (fire-and-forget)

        Raises:
            Never raises - this is a fire-and-forget operation
        """
        # Early return if disabled
        if not self._config.enabled:
            logger.debug("RAGClient is disabled, skipping post-action")
            return

        # Create feedback from result
        injection_id = (
            pre_action_context.injection_id if pre_action_context else None
        )
        feedback = PostActionFeedback.from_result(
            result=result,
            run_id=run_id or "unknown",
            injection_id=injection_id,
        )

        # Schema-driven logging: populate extra fields requested by MV-rag
        extra_fields = await self._extract_schema_driven_fields(
            result=result,
            provider=provider,
            model=model,
        )
        if extra_fields:
            feedback.extra.update(extra_fields)

        # Fire-and-forget: send feedback with retry
        try:
            await self._post_with_retry(
                "/api/v1/post-action",
                json=feedback.to_dict(),
            )
            logger.debug(
                f"Post-action feedback sent: success={feedback.success}, "
                f"type={feedback.result_type}, extra_fields={list(extra_fields.keys())}"
            )

        except Exception as e:
            # Always fail-open for post-action
            logger.warning(f"RAG post-action failed (ignored): {e}")

    async def _extract_schema_driven_fields(
        self,
        result: LLMResult,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> dict:
        """Extract fields requested by MV-rag schema from LLMResult.

        This enables schema-driven logging: MV-rag declares what fields it needs,
        and U-llm-sdk extracts them dynamically without code changes.

        Args:
            result: LLM execution result
            provider: LLM provider name
            model: Model name

        Returns:
            Dictionary of extra fields to include in feedback
        """
        # Ensure schema is cached
        if self._schema is None:
            self._schema = await self.get_server_schema()

        if self._schema is None or not self._schema.feedback_fields:
            return {}

        extra = {}
        for field_spec in self._schema.feedback_fields:
            value = self._get_field_value(
                field_spec=field_spec,
                result=result,
                provider=provider,
                model=model,
            )
            if value is not None:
                extra[field_spec.name] = value
            elif field_spec.required:
                logger.warning(
                    f"Required feedback field '{field_spec.name}' not available"
                )

        return extra

    def _get_field_value(
        self,
        field_spec: FeedbackFieldSpec,
        result: LLMResult,
        provider: Optional[str],
        model: Optional[str],
    ):
        """Get field value based on source hint.

        Args:
            field_spec: Field specification from schema
            result: LLM execution result
            provider: Provider name
            model: Model name

        Returns:
            Field value or None if not available
        """
        name = field_spec.name
        source = field_spec.source

        # Config-sourced fields
        if source == "config":
            if name == "provider":
                return provider
            if name == "model":
                return model

        # LLM result-sourced fields
        if source == "llm_result":
            # Token usage fields
            # Support both OpenAI-style (prompt_tokens) and llm-types style (input_tokens)
            if result.token_usage:
                if name in ("prompt_tokens", "input_tokens"):
                    return result.token_usage.input_tokens
                if name in ("completion_tokens", "output_tokens"):
                    return result.token_usage.output_tokens
                if name == "total_tokens":
                    return result.token_usage.total_tokens
                if name == "cached_tokens":
                    return result.token_usage.cached_tokens
                if name == "reasoning_tokens":
                    return result.token_usage.reasoning_tokens

            # Direct result attributes
            if hasattr(result, name):
                return getattr(result, name)

        return None

    async def close(self) -> None:
        """Close the HTTP client and clean up resources."""
        # Cancel background eviction task if running
        if self._eviction_task is not None:
            self._eviction_task.cancel()
            try:
                await self._eviction_task
            except asyncio.CancelledError:
                pass
            self._eviction_task = None

        await self._client.aclose()
        self._cache.clear()

    async def get_server_schema(self) -> Optional[ServerSchema]:
        """Get MV-rag server schema information.

        Returns:
            ServerSchema with version and compatibility info, or None on error
        """
        try:
            response = await self._get_with_retry("/api/v1/schema")
            response.raise_for_status()
            return ServerSchema.from_dict(response.json())
        except Exception as e:
            logger.warning(f"Failed to get server schema: {e}")
            return None

    async def _get_with_retry(self, url: str, **kwargs) -> httpx.Response:
        """HTTP GET with tenacity retry logic.

        Args:
            url: Request URL path
            **kwargs: Additional arguments for httpx.get

        Returns:
            HTTP response

        Raises:
            httpx exceptions after retries exhausted
        """
        @self._retry
        async def _do_get():
            return await self._client.get(url, **kwargs)

        return await _do_get()

    async def _post_with_retry(self, url: str, **kwargs) -> httpx.Response:
        """HTTP POST with tenacity retry logic.

        Args:
            url: Request URL path
            **kwargs: Additional arguments for httpx.post

        Returns:
            HTTP response

        Raises:
            httpx exceptions after retries exhausted
        """
        @self._retry
        async def _do_post():
            return await self._client.post(url, **kwargs)

        return await _do_post()

    async def _periodic_cache_eviction(self, interval_seconds: int) -> None:
        """Run periodic cache eviction in background.

        Args:
            interval_seconds: Interval between eviction runs
        """
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                count = self._cache.evict_expired()
                if count > 0:
                    logger.debug(f"Evicted {count} expired cache entries")
            except asyncio.CancelledError:
                logger.debug("Cache eviction task cancelled")
                break
            except Exception as e:
                logger.warning(f"Cache eviction error: {e}")

    async def check_schema_compatibility(self) -> bool:
        """Check if client is compatible with MV-rag server.

        This method queries the server's schema endpoint and verifies that
        the client version is >= the server's min_client_version.

        Returns:
            True if compatible or check failed (fail-open), False if incompatible

        Note:
            This is fail-open: returns True if the check fails for any reason.
            Incompatibility only returns False if we successfully determined
            the versions are incompatible.
        """
        schema = await self.get_server_schema()

        if schema is None:
            logger.debug("Schema check failed (fail-open: assuming compatible)")
            return True  # Fail-open

        if not schema.is_compatible:
            logger.warning(
                f"Client version {CLIENT_VERSION} is incompatible with server. "
                f"Server requires >= {schema.min_client_version}. "
                f"Breaking changes: {schema.breaking_changes}"
            )
            return False

        logger.debug(
            f"Schema compatible: client={CLIENT_VERSION}, "
            f"server={schema.version}, min_required={schema.min_client_version}"
        )
        return True

    async def get_codebase_context(
        self,
        cwd: str,
        query: str = "repository overview",
        stage: str = "INDEX",
        seed_ids: list | None = None,
        limit: int = 20,
    ) -> str | None:
        """Get codebase context via MV-rag SCIP code intelligence.

        This method calls the MV-rag API to get SCIP-based code intelligence
        without directly importing from MV-rag (following architecture rules).

        Supports Progressive Disclosure stages:
        - INDEX: High-level repository map (~800 tokens)
        - DETAILS: Expanded context from seed IDs (~2500 tokens)
        - DEEP_DIVE: Full source excerpts (~4500 tokens)

        Args:
            cwd: Project root directory path
            query: Task description for relevance filtering
            stage: Disclosure stage (INDEX, DETAILS, DEEP_DIVE)
            seed_ids: Seed IDs for DETAILS/DEEP_DIVE expansion
            limit: Maximum number of evidence chunks

        Returns:
            Formatted context text for prompt injection, or None on error

        Example:
            >>> context = await client.get_codebase_context(
            ...     cwd="/path/to/project",
            ...     query="implement authentication",
            ... )
            >>> if context:
            ...     prompt = context + "\\n\\n" + user_prompt
        """
        if not self._config.enabled:
            logger.debug("RAGClient is disabled, skipping codebase context")
            return None

        try:
            response = await self._post_with_retry(
                "/api/v1/inject/codebase-context",
                json={
                    "cwd": cwd,
                    "query": query,
                    "stage": stage,
                    "seed_ids": seed_ids or [],
                    "limit": limit,
                },
            )
            response.raise_for_status()

            data = response.json()

            if not data.get("success", False):
                logger.debug("Codebase context request returned success=False")
                return None

            context_text = data.get("context_text")
            if context_text:
                logger.info(
                    f"Codebase context received: {data.get('chunk_count', 0)} chunks, "
                    f"stage={data.get('stage', 'INDEX')}, "
                    f"latency={data.get('latency_ms', 0):.1f}ms"
                )
            return context_text

        except httpx.TimeoutException as e:
            logger.warning(f"Codebase context timeout: {e}")
            return None

        except httpx.HTTPStatusError as e:
            logger.warning(f"Codebase context HTTP error {e.response.status_code}: {e}")
            return None

        except Exception as e:
            logger.error(f"Codebase context unexpected error: {e}", exc_info=True)
            return None

    # =========================================================================
    # Context Scout API (Active RAG Gathering)
    # =========================================================================

    async def get_file_content(
        self,
        file_path: str,
        cwd: str,
        max_lines: int = 200,
    ) -> str | None:
        """Get file content via MV-rag.

        Args:
            file_path: Relative or absolute file path
            cwd: Working directory
            max_lines: Maximum lines to return

        Returns:
            File content or None on error
        """
        if not self._config.enabled:
            return None

        try:
            response = await self._post_with_retry(
                "/api/v1/scout/file",
                json={
                    "file_path": file_path,
                    "cwd": cwd,
                    "max_lines": max_lines,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("content")

        except Exception as e:
            logger.debug(f"Scout get_file_content failed: {e}")
            return None

    async def search_symbols(
        self,
        query: str,
        cwd: str,
        limit: int = 5,
    ) -> list[dict] | None:
        """Search for code symbols via MV-rag SCIP.

        Args:
            query: Symbol name or pattern
            cwd: Working directory
            limit: Maximum results

        Returns:
            List of symbol matches with location and code, or None on error
        """
        if not self._config.enabled:
            return None

        try:
            response = await self._post_with_retry(
                "/api/v1/scout/symbols",
                json={
                    "query": query,
                    "cwd": cwd,
                    "limit": limit,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])

        except Exception as e:
            logger.debug(f"Scout search_symbols failed: {e}")
            return None

    async def semantic_search(
        self,
        query: str,
        cwd: str,
        limit: int = 5,
    ) -> list[dict] | None:
        """Semantic search for code via MV-rag.

        Args:
            query: Natural language query
            cwd: Working directory
            limit: Maximum results

        Returns:
            List of search results with file and content, or None on error
        """
        if not self._config.enabled:
            return None

        try:
            response = await self._post_with_retry(
                "/api/v1/scout/search",
                json={
                    "query": query,
                    "cwd": cwd,
                    "limit": limit,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])

        except Exception as e:
            logger.debug(f"Scout semantic_search failed: {e}")
            return None

    async def __aenter__(self):
        """Async context manager entry."""
        # Start background cache eviction if enabled
        interval = self._config.eviction_interval_seconds
        if interval > 0:
            self._eviction_task = asyncio.create_task(
                self._periodic_cache_eviction(interval)
            )
            logger.debug(f"Started cache eviction task (interval={interval}s)")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
