"""Gemini API Provider.

Direct API access to Google Gemini using the google-genai SDK.
Unlike GeminiProvider (CLI wrapper), this calls the API directly.

Usage:
    >>> from u_llm_sdk import GeminiAPIProvider, LLMConfig, ModelTier
    >>> provider = GeminiAPIProvider(LLMConfig(tier=ModelTier.HIGH))
    >>> result = await provider.run("What is 2+2?")

Model Naming:
    - CLI (GeminiProvider): gemini-3-pro
    - API (GeminiAPIProvider): gemini-3-pro-preview

Requirements:
    pip install google-genai>=0.8.0
    # Or: pip install u-llm-sdk[gemini-api]
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional

from u_llm_sdk.types import (
    API_MODEL_TIERS,
    CURRENT_MODELS,
    LLMResult,
    ModelTier,
    Provider,
    ResultType,
    TokenUsage,
)
from u_llm_sdk.types.exceptions import AuthenticationError, InvalidConfigError

if TYPE_CHECKING:
    from u_llm_sdk.config import LLMConfig

logger = logging.getLogger(__name__)


# API key environment variable
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"


class GeminiAPIProvider:
    """Gemini API Provider using google-genai SDK.

    This provider directly calls the Google Gemini API, bypassing the CLI.
    Use this when you need:
    - Programmatic API access without CLI installation
    - Faster response times (no subprocess overhead)
    - Direct control over API parameters

    Note:
        Model names differ between CLI and API:
        - CLI: gemini-3-pro (via GeminiProvider)
        - API: gemini-3-pro-preview (via this class)

    Attributes:
        PROVIDER: Provider.GEMINI (same as CLI provider)
        _config: LLMConfig instance
        _client: google.generativeai client
    """

    PROVIDER = Provider.GEMINI

    def __init__(
        self,
        config: "LLMConfig",
        *,
        verify_api_key: bool = True,
    ) -> None:
        """Initialize GeminiAPIProvider.

        Args:
            config: LLMConfig with model/tier settings
            verify_api_key: If True, verify API key exists on init

        Raises:
            AuthenticationError: If API key not found (when verify_api_key=True)
            ImportError: If google-genai package not installed
        """
        self._config = config
        self._client: Any = None
        self._model_name: Optional[str] = None

        # Lazy import to avoid hard dependency
        # Try new google-genai SDK first, fall back to google-generativeai
        try:
            from google import genai

            self._genai = genai
            self._use_new_sdk = True
        except ImportError:
            try:
                import google.generativeai as genai

                self._genai = genai
                self._use_new_sdk = False
            except ImportError as e:
                raise ImportError(
                    "google-genai or google-generativeai package required for GeminiAPIProvider. "
                    "Install with: pip install google-genai>=0.8.0 "
                    "or pip install google-generativeai>=0.3.0"
                ) from e

        # Resolve API model name
        self._model_name = self._resolve_api_model()

        # Verify API key
        if verify_api_key:
            api_key = self._get_api_key()
            if not api_key:
                raise AuthenticationError(
                    f"Gemini API key not found. Set {GEMINI_API_KEY_ENV} environment variable."
                )

    def _get_api_key(self) -> Optional[str]:
        """Get API key from config or environment.

        Priority:
        1. config.api_key (explicit)
        2. GEMINI_API_KEY environment variable
        """
        if self._config.api_key:
            return self._config.api_key
        return os.environ.get(GEMINI_API_KEY_ENV)

    def _resolve_api_model(self) -> str:
        """Resolve model name for API calls.

        Uses API_MODEL_TIERS instead of MODEL_TIERS for correct API model names.

        Returns:
            API model name (e.g., gemini-3-pro-preview)

        Raises:
            InvalidConfigError: If model cannot be resolved
        """
        # If tier is specified, use API_MODEL_TIERS
        if self._config.tier:
            tier_models = API_MODEL_TIERS.get(Provider.GEMINI, {})
            if self._config.tier in tier_models:
                return tier_models[self._config.tier]

        # If model is specified directly
        if self._config.model:
            model = self._config.model
            # Check if it's a current model
            if model in CURRENT_MODELS.get(Provider.GEMINI, set()):
                return model
            # Pass through as-is (user responsibility)
            logger.warning(
                f"Model '{model}' not in CURRENT_MODELS. Using as-is."
            )
            return model

        # Default to HIGH tier API model
        default_model = API_MODEL_TIERS.get(Provider.GEMINI, {}).get(
            ModelTier.HIGH, "gemini-3-pro-preview"
        )
        logger.info(f"No model/tier specified. Using default: {default_model}")
        return default_model

    def _get_client(self) -> Any:
        """Get or create the generative AI client.

        Returns:
            - New SDK: genai.Client instance (use client.aio.models for async)
            - Legacy SDK: genai.GenerativeModel instance

        Note:
            Supports both new google-genai SDK and legacy google-generativeai SDK.
        """
        if self._client is None:
            api_key = self._get_api_key()

            if self._use_new_sdk:
                # New google-genai SDK
                # Store full client, use client.aio.models for async operations
                self._client = self._genai.Client(api_key=api_key)
            else:
                # Legacy google-generativeai SDK
                self._genai.configure(api_key=api_key)
                self._client = self._genai.GenerativeModel(self._model_name)

        return self._client

    @property
    def model_name(self) -> str:
        """Current model name being used."""
        return self._model_name or "unknown"

    @property
    def provider_name(self) -> str:
        """Provider name string."""
        return self.PROVIDER.value

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run a single prompt through the Gemini API.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum output tokens
            **kwargs: Additional parameters passed to generate_content

        Returns:
            LLMResult with API response
        """
        start_time = time.perf_counter()
        client = self._get_client()

        try:
            if self._use_new_sdk:
                response = await self._run_new_sdk(
                    client, prompt, system_prompt, temperature, max_tokens, **kwargs
                )
            else:
                response = await self._run_legacy_sdk(
                    client, prompt, system_prompt, temperature, max_tokens, **kwargs
                )

            duration_ms = int((time.perf_counter() - start_time) * 1000)

            # Extract text
            text = ""
            if hasattr(response, "text") and response.text:
                text = response.text

            # Extract token usage if available
            token_usage = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = response.usage_metadata
                token_usage = TokenUsage(
                    input_tokens=getattr(usage, "prompt_token_count", 0),
                    output_tokens=getattr(usage, "candidates_token_count", 0),
                    total_tokens=getattr(usage, "total_token_count", 0),
                )

            return LLMResult(
                success=True,
                result_type=ResultType.TEXT,
                provider=self.provider_name,
                model=self.model_name,
                text=text,
                summary=text[:100] + "..." if len(text) > 100 else text,
                duration_ms=duration_ms,
                token_usage=token_usage,
                raw=response,
            )

        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"Gemini API error: {e}")
            return LLMResult(
                success=False,
                result_type=ResultType.ERROR,
                provider=self.provider_name,
                model=self.model_name,
                error=str(e),
                duration_ms=duration_ms,
            )

    async def _run_new_sdk(
        self,
        client: Any,
        prompt: str,
        system_prompt: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        **kwargs: Any,
    ) -> Any:
        """Run using new google-genai SDK.

        Note: New SDK uses client.aio.models.generate_content for async.
        """
        from google.genai import types

        # Build config
        config_kwargs: dict[str, Any] = {}
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        # Use client.aio.models for async operations
        return await client.aio.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=config,
            **kwargs,
        )

    async def _run_legacy_sdk(
        self,
        client: Any,
        prompt: str,
        system_prompt: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        **kwargs: Any,
    ) -> Any:
        """Run using legacy google-generativeai SDK."""
        # Build generation config
        generation_config: dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens

        # Build contents
        contents = []
        if system_prompt:
            # Legacy SDK: prepend system prompt to user message
            contents.append(f"[System: {system_prompt}]\n\n{prompt}")
        else:
            contents.append(prompt)

        return await client.generate_content_async(
            contents,
            generation_config=generation_config if generation_config else None,
            **kwargs,
        )

    async def stream(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream response from Gemini API.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum output tokens
            **kwargs: Additional parameters

        Yields:
            Text chunks as they arrive
        """
        client = self._get_client()

        try:
            if self._use_new_sdk:
                async for chunk in self._stream_new_sdk(
                    client, prompt, system_prompt, temperature, max_tokens, **kwargs
                ):
                    yield chunk
            else:
                async for chunk in self._stream_legacy_sdk(
                    client, prompt, system_prompt, temperature, max_tokens, **kwargs
                ):
                    yield chunk

        except Exception as e:
            logger.error(f"Gemini API streaming error: {e}")
            raise

    async def _stream_new_sdk(
        self,
        client: Any,
        prompt: str,
        system_prompt: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream using new google-genai SDK."""
        from google.genai import types

        config_kwargs: dict[str, Any] = {}
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        # Use client.aio.models for async streaming
        async for chunk in client.aio.models.generate_content_stream(
            model=self._model_name,
            contents=prompt,
            config=config,
            **kwargs,
        ):
            if hasattr(chunk, "text") and chunk.text:
                yield chunk.text

    async def _stream_legacy_sdk(
        self,
        client: Any,
        prompt: str,
        system_prompt: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream using legacy google-generativeai SDK."""
        generation_config: dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens

        contents = []
        if system_prompt:
            contents.append(f"[System: {system_prompt}]\n\n{prompt}")
        else:
            contents.append(prompt)

        response = await client.generate_content_async(
            contents,
            generation_config=generation_config if generation_config else None,
            stream=True,
            **kwargs,
        )

        async for chunk in response:
            if chunk.text:
                yield chunk.text


__all__ = ["GeminiAPIProvider"]
