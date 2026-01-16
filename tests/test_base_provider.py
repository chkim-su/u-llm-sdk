"""Tests for BaseProvider abstract class."""

import os
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator, Optional
from unittest.mock import AsyncMock, patch

import pytest

from u_llm_sdk.types import LLMResult, PreActionContext, Provider, ResultType
from u_llm_sdk import LLMConfig, ModelTier
from u_llm_sdk.llm.providers import BaseProvider, NoOpHook


class ConcreteProvider(BaseProvider):
    """Concrete implementation for testing."""

    PROVIDER = Provider.CLAUDE
    CLI_NAME = "claude"

    def _build_args(
        self,
        prompt: str,
        session_id: Optional[str] = None,
    ) -> list[str]:
        """Build CLI arguments for testing."""
        return [self.cli_executable, "-p", prompt]

    def _parse_output(
        self,
        stdout: str,
        stderr: str,
        success: bool,
        duration_ms: int,
        **kwargs: Any,
    ) -> LLMResult:
        """Parse CLI output for testing."""
        return LLMResult(
            success=success,
            result_type=ResultType.TEXT,
            provider=self.provider_name,
            model="test-model",
            text=stdout,
            summary="Test response",
            duration_ms=duration_ms,
        )


class TestBaseProviderInit:
    """Test BaseProvider initialization."""

    def test_init_with_default_config(self):
        """Test init with default config."""
        with patch.object(ConcreteProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ConcreteProvider()
            assert provider.PROVIDER == Provider.CLAUDE
            assert provider.config.provider == Provider.CLAUDE

    def test_init_with_custom_config(self):
        """Test init with custom config."""
        config = LLMConfig(provider=Provider.CLAUDE, tier=ModelTier.HIGH)
        with patch.object(ConcreteProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ConcreteProvider(config=config)
            assert provider.config.tier == ModelTier.HIGH

    def test_init_with_intervention_hook(self):
        """Test init with intervention hook."""
        hook = NoOpHook()
        with patch.object(ConcreteProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ConcreteProvider(intervention_hook=hook)
            assert provider.intervention_hook is hook

    def test_init_verify_cli_failure(self):
        """Test init fails when CLI not found and verify_cli=True."""
        from u_llm_sdk.types import ProviderNotFoundError

        with patch.object(ConcreteProvider, "get_cli_path", return_value=None):
            with pytest.raises(ProviderNotFoundError):
                ConcreteProvider(verify_cli=True)

    def test_init_verify_cli_skip(self):
        """Test init succeeds when verify_cli=False."""
        with patch.object(ConcreteProvider, "get_cli_path", return_value=None):
            provider = ConcreteProvider(verify_cli=False)
            assert provider._cli_path is None


class TestCwdValidation:
    """Test working directory validation."""

    def test_valid_cwd(self):
        """Test valid cwd passes validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMConfig(cwd=tmpdir)
            with patch.object(ConcreteProvider, "get_cli_path", return_value="/usr/bin/claude"):
                provider = ConcreteProvider(config=config)
                assert provider.config.cwd == tmpdir

    def test_nonexistent_cwd_raises(self):
        """Test nonexistent cwd raises ValueError."""
        config = LLMConfig(cwd="/nonexistent/path/12345")
        with patch.object(ConcreteProvider, "get_cli_path", return_value="/usr/bin/claude"):
            with pytest.raises(ValueError, match="does not exist"):
                ConcreteProvider(config=config)

    def test_file_cwd_raises(self):
        """Test file path as cwd raises ValueError."""
        with tempfile.NamedTemporaryFile() as f:
            config = LLMConfig(cwd=f.name)
            with patch.object(ConcreteProvider, "get_cli_path", return_value="/usr/bin/claude"):
                with pytest.raises(ValueError, match="not a directory"):
                    ConcreteProvider(config=config)


class TestCliExecutable:
    """Test CLI executable resolution."""

    def test_cli_executable_with_path(self):
        """Test cli_executable returns discovered path."""
        with patch.object(ConcreteProvider, "get_cli_path", return_value="/usr/local/bin/claude"):
            provider = ConcreteProvider()
            assert provider.cli_executable == "/usr/local/bin/claude"

    def test_cli_executable_fallback(self):
        """Test cli_executable falls back to CLI_NAME."""
        with patch.object(ConcreteProvider, "get_cli_path", return_value=None):
            provider = ConcreteProvider(verify_cli=False)
            assert provider.cli_executable == "claude"


class TestSessionManagement:
    """Test session management."""

    def test_session_id_property(self):
        """Test session_id property."""
        with patch.object(ConcreteProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ConcreteProvider()
            assert provider.session_id is None

            provider._session_id = "test-session"
            assert provider.session_id == "test-session"

    def test_resume(self):
        """Test resume method."""
        with patch.object(ConcreteProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ConcreteProvider()
            result = provider.resume("session-123")

            assert result is provider  # Returns self
            assert provider.session_id == "session-123"


class TestEffectiveValues:
    """Test effective value resolution."""

    def test_effective_timeout(self):
        """Test _get_effective_timeout."""
        config = LLMConfig(timeout=300.0)
        with patch.object(ConcreteProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ConcreteProvider(config=config)

            # Explicit timeout takes precedence
            assert provider._get_effective_timeout(100.0) == 100.0

            # Falls back to config
            assert provider._get_effective_timeout(None) == 300.0

    def test_effective_session_id(self):
        """Test _get_effective_session_id."""
        config = LLMConfig(session_id="config-session")
        with patch.object(ConcreteProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ConcreteProvider(config=config)

            # Explicit session_id takes precedence
            assert provider._get_effective_session_id("explicit") == "explicit"

            # Falls back to instance session
            provider._session_id = "instance-session"
            assert provider._get_effective_session_id(None) == "instance-session"


class TestEnvFile:
    """Test .env file loading via ApiKeyResolver service."""

    def test_load_api_key_from_env_file(self):
        """Test loading API key from .env file via ApiKeyResolver."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("ANTHROPIC_API_KEY=test-key-123\n")
            f.write("OTHER_VAR=other-value\n")
            f.flush()

            try:
                # Set permissions
                os.chmod(f.name, 0o600)

                config = LLMConfig(provider=Provider.CLAUDE, env_file=f.name)
                with patch.object(ConcreteProvider, "get_cli_path", return_value="/usr/bin/claude"):
                    provider = ConcreteProvider(config=config)
                    # Access via the service
                    key = provider._api_key_resolver.resolve()
                    assert key == "test-key-123"
            finally:
                os.unlink(f.name)

    def test_load_api_key_nonexistent_file(self):
        """Test loading from nonexistent file returns None."""
        config = LLMConfig(provider=Provider.CLAUDE, env_file="/nonexistent/file.env")
        with patch.object(ConcreteProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ConcreteProvider(config=config)
            key = provider._api_key_resolver.resolve()
            assert key is None


class TestInterventionHooks:
    """Test intervention hook integration."""

    @pytest.mark.asyncio
    async def test_apply_injection_prepend(self):
        """Test injection with prepend position."""
        with patch.object(ConcreteProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ConcreteProvider()

            context = PreActionContext.create(
                context_text="Injected context",
                confidence=0.9,
                injection_position="prepend",
            )

            result = provider._apply_injection("Original prompt", context)
            assert "Injected context" in result
            assert "Original prompt" in result
            assert result.index("Injected context") < result.index("Original prompt")

    @pytest.mark.asyncio
    async def test_apply_injection_append(self):
        """Test injection with append position."""
        with patch.object(ConcreteProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ConcreteProvider()

            context = PreActionContext.create(
                context_text="Appended context",
                confidence=0.9,
                injection_position="append",
            )

            result = provider._apply_injection("Original prompt", context)
            assert result.index("Original prompt") < result.index("Appended context")

    @pytest.mark.asyncio
    async def test_call_pre_action_hook(self):
        """Test HookManager.call_pre_action calls hook."""
        mock_hook = AsyncMock()
        mock_hook.on_pre_action.return_value = PreActionContext.create(
            context_text="Test",
            confidence=0.9,
        )

        with patch.object(ConcreteProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ConcreteProvider(intervention_hook=mock_hook)
            result = await provider._hook_manager.call_pre_action("test prompt")

            assert result is not None
            mock_hook.on_pre_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_pre_action_hook_no_hook(self):
        """Test HookManager.call_pre_action returns None when no hook."""
        with patch.object(ConcreteProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ConcreteProvider()
            result = await provider._hook_manager.call_pre_action("test prompt")

            assert result is None

    @pytest.mark.asyncio
    async def test_call_post_action_hook(self):
        """Test HookManager.call_post_action calls hook."""
        mock_hook = AsyncMock()

        result = LLMResult(
            success=True,
            result_type=ResultType.TEXT,
            provider="claude",
            model="test",
            text="Response",
            summary="Test",
        )

        with patch.object(ConcreteProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ConcreteProvider(intervention_hook=mock_hook)
            await provider._hook_manager.call_post_action(result, None)

            mock_hook.on_post_action.assert_called_once()


class TestParallelRun:
    """Test parallel_run method."""

    @pytest.mark.asyncio
    async def test_parallel_run(self):
        """Test parallel execution of multiple prompts."""
        # Mock the subprocess to return expected output
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"mock response", b"")
        mock_process.returncode = 0

        with patch.object(ConcreteProvider, "get_cli_path", return_value="/usr/bin/claude"):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                provider = ConcreteProvider()
                prompts = ["prompt1", "prompt2", "prompt3"]

                results = await provider.parallel_run(prompts)

                assert len(results) == 3
                assert all(r.success for r in results)


class TestNoOpHook:
    """Test NoOpHook implementation."""

    @pytest.mark.asyncio
    async def test_on_pre_action_returns_none(self):
        """Test NoOpHook.on_pre_action returns None."""
        hook = NoOpHook()
        result = await hook.on_pre_action("test", "claude")
        assert result is None

    @pytest.mark.asyncio
    async def test_on_post_action_does_nothing(self):
        """Test NoOpHook.on_post_action completes without error."""
        hook = NoOpHook()
        result = LLMResult(
            success=True,
            result_type=ResultType.TEXT,
            provider="claude",
            model="test",
            text="Response",
            summary="Test",
        )
        # Should complete without error
        await hook.on_post_action(result, None)
