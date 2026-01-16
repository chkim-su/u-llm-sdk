"""Tests for ClaudeProvider implementation."""

import json
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from u_llm_sdk.types import LLMResult, Provider, ResultType, TokenUsage
from u_llm_sdk import LLMConfig, ModelTier
from u_llm_sdk.llm.providers.claude import ClaudeProvider


class TestClaudeProviderInit:
    """Test ClaudeProvider initialization."""

    def test_provider_constant(self):
        """Test PROVIDER constant."""
        assert ClaudeProvider.PROVIDER == Provider.CLAUDE

    def test_cli_name(self):
        """Test CLI_NAME constant."""
        assert ClaudeProvider.CLI_NAME == "claude"

    def test_init_with_default_config(self):
        """Test init with default config."""
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider()
            assert provider.PROVIDER == Provider.CLAUDE
            assert provider.config.provider == Provider.CLAUDE

    def test_init_with_custom_config(self):
        """Test init with custom config."""
        config = LLMConfig(provider=Provider.CLAUDE, tier=ModelTier.HIGH)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            assert provider.config.tier == ModelTier.HIGH


class TestBuildArgs:
    """Test CLI argument building."""

    def test_build_args_basic(self):
        """Test basic argument building."""
        # Use NONE reasoning to avoid thinking prefix
        from u_llm_sdk.config import ReasoningLevel as RL

        config = LLMConfig(provider=Provider.CLAUDE, reasoning_level=RL.NONE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            args = provider._build_args("test prompt")

            assert "/usr/bin/claude" in args
            assert "-p" in args
            assert "test prompt" in args
            assert "--output-format" in args
            assert "stream-json" in args
            assert "--verbose" in args

    def test_build_args_with_model(self):
        """Test argument building with model.

        Note: Legacy model "claude-3-opus" routes to HIGH tier → "opus"
        """
        config = LLMConfig(provider=Provider.CLAUDE, model="claude-3-opus")
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            args = provider._build_args("test prompt")

            assert "--model" in args
            idx = args.index("--model")
            # Legacy claude-3-opus routes to HIGH tier → "opus"
            assert args[idx + 1] == "opus"

    def test_build_args_with_current_model(self):
        """Test argument building with current model (no routing)."""
        config = LLMConfig(provider=Provider.CLAUDE, model="opus")
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            args = provider._build_args("test prompt")

            assert "--model" in args
            idx = args.index("--model")
            # Current model passes through as-is
            assert args[idx + 1] == "opus"

    def test_build_args_with_session_id(self):
        """Test argument building with session ID."""
        config = LLMConfig(provider=Provider.CLAUDE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            args = provider._build_args("test prompt", session_id="session-123")

            assert "--resume" in args
            idx = args.index("--resume")
            assert args[idx + 1] == "session-123"

    def test_build_args_with_system_prompt(self):
        """Test argument building with system prompt."""
        config = LLMConfig(provider=Provider.CLAUDE, system_prompt="You are a helpful assistant")
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            args = provider._build_args("test prompt")

            assert "--system-prompt" in args
            idx = args.index("--system-prompt")
            assert args[idx + 1] == "You are a helpful assistant"

    def test_build_args_with_allowed_tools(self):
        """Test argument building with allowed tools."""
        config = LLMConfig(
            provider=Provider.CLAUDE,
            provider_options={"allowed_tools": ["Read", "Write", "Bash"]},
        )
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            args = provider._build_args("test prompt")

            assert "--allowed-tools" in args
            idx = args.index("--allowed-tools")
            assert args[idx + 1] == "Read,Write,Bash"

    def test_build_args_with_disallowed_tools(self):
        """Test argument building with disallowed tools."""
        config = LLMConfig(
            provider=Provider.CLAUDE,
            provider_options={"disallowed_tools": ["Bash"]},
        )
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            args = provider._build_args("test prompt")

            assert "--disallowed-tools" in args
            idx = args.index("--disallowed-tools")
            assert args[idx + 1] == "Bash"

    def test_build_args_with_mcp_config(self):
        """Test argument building with MCP config."""
        config = LLMConfig(
            provider=Provider.CLAUDE,
            provider_options={"mcp_config": "/path/to/mcp.json"},
        )
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            args = provider._build_args("test prompt")

            assert "--mcp-config" in args
            idx = args.index("--mcp-config")
            assert args[idx + 1] == "/path/to/mcp.json"

    def test_build_args_with_max_turns(self):
        """Test argument building with max turns."""
        config = LLMConfig(
            provider=Provider.CLAUDE,
            provider_options={"max_turns": 10},
        )
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            args = provider._build_args("test prompt")

            assert "--max-turns" in args
            idx = args.index("--max-turns")
            assert args[idx + 1] == "10"

    def test_build_args_with_setting_sources(self):
        """Test argument building with setting sources."""
        config = LLMConfig(
            provider=Provider.CLAUDE,
            provider_options={"setting_sources": ["user", "project"]},
        )
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            args = provider._build_args("test prompt")

            assert "--setting-sources" in args
            idx = args.index("--setting-sources")
            assert args[idx + 1] == "user,project"


class TestThinkingPrefix:
    """Test thinking prefix application.

    ReasoningLevel enum values:
    - NONE: No prefix
    - LOW: No prefix
    - MEDIUM: "think: " prefix (default!)
    - HIGH: "think hard: " prefix
    - XHIGH: "ultrathink: " prefix
    """

    def test_apply_thinking_prefix_none(self):
        """Test no prefix when reasoning is NONE."""
        from u_llm_sdk.config import ReasoningLevel

        config = LLMConfig(provider=Provider.CLAUDE, reasoning_level=ReasoningLevel.NONE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            result = provider._apply_thinking_prefix("test prompt")
            assert result == "test prompt"

    def test_apply_thinking_prefix_default_medium(self):
        """Test MEDIUM reasoning does NOT add prefix (deprecated).

        NOTE (2026-01-08 verified): Only ultrathink triggers extended thinking
        in Claude CLI v2.0.0+. MEDIUM and HIGH are deprecated.
        """
        config = LLMConfig(provider=Provider.CLAUDE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            result = provider._apply_thinking_prefix("test prompt")
            # MEDIUM does NOT add prefix (deprecated)
            assert result == "test prompt"

    def test_apply_thinking_prefix_already_has_ultrathink(self):
        """Test no prefix when prompt already starts with ultrathink."""
        from u_llm_sdk.config import ReasoningLevel

        config = LLMConfig(provider=Provider.CLAUDE, reasoning_level=ReasoningLevel.XHIGH)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            result = provider._apply_thinking_prefix("ultrathink about this")
            assert result == "ultrathink about this"

    def test_apply_thinking_prefix_high(self):
        """Test HIGH reasoning does NOT add prefix (deprecated).

        NOTE (2026-01-08 verified): Only ultrathink triggers extended thinking
        in Claude CLI v2.0.0+. HIGH ('think hard') is deprecated.
        """
        from u_llm_sdk.config import ReasoningLevel

        config = LLMConfig(provider=Provider.CLAUDE, reasoning_level=ReasoningLevel.HIGH)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            result = provider._apply_thinking_prefix("test prompt")
            # HIGH does NOT add prefix (deprecated)
            assert result == "test prompt"

    def test_apply_thinking_prefix_xhigh(self):
        """Test XHIGH reasoning adds 'ultrathink:' prefix."""
        from u_llm_sdk.config import ReasoningLevel

        config = LLMConfig(provider=Provider.CLAUDE, reasoning_level=ReasoningLevel.XHIGH)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            result = provider._apply_thinking_prefix("test prompt")
            assert result == "ultrathink: test prompt"


class TestParseOutput:
    """Test output parsing."""

    def test_parse_output_empty(self):
        """Test parsing empty output."""
        config = LLMConfig(provider=Provider.CLAUDE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            result = provider._parse_output("", "", True, 100)

            assert result.success is True
            assert result.result_type == ResultType.TEXT
            assert result.text == ""

    def test_parse_output_text_only(self):
        """Test parsing text-only output."""
        events = [
            {"type": "system", "session_id": "sess-123"},
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Hello, world!"}]
                },
            },
            {"type": "result", "usage": {"input_tokens": 10, "output_tokens": 20}},
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        config = LLMConfig(provider=Provider.CLAUDE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            result = provider._parse_output(stdout, "", True, 100)

            assert result.success is True
            assert result.result_type == ResultType.TEXT
            assert result.text == "Hello, world!"
            assert result.session_id == "sess-123"
            assert result.token_usage is not None
            assert result.token_usage.input_tokens == 10
            assert result.token_usage.output_tokens == 20

    def test_parse_output_with_thinking(self):
        """Test parsing output with thinking block."""
        events = [
            {"type": "system", "session_id": "sess-123"},
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "Let me think..."},
                        {"type": "text", "text": "The answer is 42."},
                    ]
                },
            },
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        config = LLMConfig(provider=Provider.CLAUDE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            result = provider._parse_output(stdout, "", True, 100)

            assert result.success is True
            assert result.text == "The answer is 42."
            assert result.thinking == "Let me think..."

    def test_parse_output_with_file_edit(self):
        """Test parsing output with file edit."""
        events = [
            {"type": "system", "session_id": "sess-123"},
            {
                "type": "tool_use",
                "name": "Edit",
                "input": {
                    "file_path": "/test/file.py",
                    "old_string": "old",
                    "new_string": "new",
                },
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "File edited."}]},
            },
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        config = LLMConfig(provider=Provider.CLAUDE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            result = provider._parse_output(stdout, "", True, 100)

            assert result.success is True
            assert result.result_type == ResultType.MIXED
            assert len(result.files_modified) == 1
            assert result.files_modified[0].path == "/test/file.py"
            assert result.files_modified[0].action == "modified"

    def test_parse_output_with_file_write(self):
        """Test parsing output with file write."""
        events = [
            {"type": "system", "session_id": "sess-123"},
            {
                "type": "tool_use",
                "name": "Write",
                "input": {
                    "file_path": "/test/newfile.py",
                    "content": "print('hello')",
                },
            },
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        config = LLMConfig(provider=Provider.CLAUDE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            result = provider._parse_output(stdout, "", True, 100)

            assert result.success is True
            assert result.result_type == ResultType.FILE_EDIT
            assert len(result.files_modified) == 1
            assert result.files_modified[0].path == "/test/newfile.py"
            assert result.files_modified[0].action == "created"

    def test_parse_output_with_command(self):
        """Test parsing output with shell command."""
        events = [
            {"type": "system", "session_id": "sess-123"},
            {
                "type": "tool_use",
                "name": "Bash",
                "input": {"command": "ls -la"},
            },
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        config = LLMConfig(provider=Provider.CLAUDE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            result = provider._parse_output(stdout, "", True, 100)

            assert result.success is True
            assert result.result_type == ResultType.COMMAND
            assert len(result.commands_run) == 1
            assert result.commands_run[0].command == "ls -la"

    def test_parse_output_error(self):
        """Test parsing failed output."""
        config = LLMConfig(provider=Provider.CLAUDE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            result = provider._parse_output("", "Error message", False, 100)

            assert result.success is False
            assert result.result_type == ResultType.ERROR
            assert result.error == "Error message"

    def test_parse_output_with_cache_tokens(self):
        """Test parsing output with cache tokens."""
        events = [
            {"type": "system", "session_id": "sess-123"},
            {
                "type": "result",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_input_tokens": 30,
                },
            },
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        config = LLMConfig(provider=Provider.CLAUDE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            result = provider._parse_output(stdout, "", True, 100)

            assert result.token_usage is not None
            assert result.token_usage.input_tokens == 100
            assert result.token_usage.output_tokens == 50
            assert result.token_usage.cached_tokens == 30
            assert result.token_usage.total_tokens == 150


class TestDetermineResultType:
    """Test result type determination."""

    def test_determine_error(self):
        """Test error result type."""
        config = LLMConfig(provider=Provider.CLAUDE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            result = provider._determine_result_type([], [], [], False)
            assert result == ResultType.ERROR

    def test_determine_text_only(self):
        """Test text-only result type."""
        config = LLMConfig(provider=Provider.CLAUDE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            result = provider._determine_result_type(["hello"], [], [], True)
            assert result == ResultType.TEXT

    def test_determine_file_edit(self):
        """Test file edit result type."""
        from u_llm_sdk.types import FileChange

        config = LLMConfig(provider=Provider.CLAUDE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            files = [FileChange(path="/test.py", action="modified")]
            result = provider._determine_result_type([], files, [], True)
            assert result == ResultType.FILE_EDIT

    def test_determine_command(self):
        """Test command result type."""
        from u_llm_sdk.types import CommandRun

        config = LLMConfig(provider=Provider.CLAUDE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            commands = [CommandRun(command="ls", exit_code=0)]
            result = provider._determine_result_type([], [], commands, True)
            assert result == ResultType.COMMAND

    def test_determine_mixed(self):
        """Test mixed result type."""
        from u_llm_sdk.types import FileChange

        config = LLMConfig(provider=Provider.CLAUDE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            files = [FileChange(path="/test.py", action="modified")]
            result = provider._determine_result_type(["hello"], files, [], True)
            assert result == ResultType.MIXED


class TestBuildSummary:
    """Test summary building."""

    def test_build_summary_text_only(self):
        """Test summary for text-only result."""
        config = LLMConfig(provider=Provider.CLAUDE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            summary = provider._build_summary(ResultType.TEXT, ["Hello world"], [], [])
            assert summary == "Hello world"

    def test_build_summary_text_truncated(self):
        """Test summary truncation for long text."""
        config = LLMConfig(provider=Provider.CLAUDE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            long_text = "x" * 200
            summary = provider._build_summary(ResultType.TEXT, [long_text], [], [])
            assert len(summary) < 110  # 100 + "..."

    def test_build_summary_files_modified(self):
        """Test summary for file modifications."""
        from u_llm_sdk.types import FileChange

        config = LLMConfig(provider=Provider.CLAUDE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            files = [
                FileChange(path="/path/to/file1.py", action="modified"),
                FileChange(path="/path/to/file2.py", action="modified"),
            ]
            summary = provider._build_summary(ResultType.FILE_EDIT, [], files, [])
            assert "2 files modified" in summary
            assert "file1.py" in summary
            assert "file2.py" in summary

    def test_build_summary_commands_run(self):
        """Test summary for commands."""
        from u_llm_sdk.types import CommandRun

        config = LLMConfig(provider=Provider.CLAUDE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)
            commands = [
                CommandRun(command="ls", exit_code=0),
                CommandRun(command="pwd", exit_code=0),
            ]
            summary = provider._build_summary(ResultType.COMMAND, [], [], commands)
            assert "2 commands executed" in summary


class TestRunMethod:
    """Test run method with mocked subprocess."""

    @pytest.mark.asyncio
    async def test_run_success(self):
        """Test successful run."""
        events = [
            {"type": "system", "session_id": "sess-123"},
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Response"}]},
            },
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (stdout.encode(), b"")
        mock_process.returncode = 0

        config = LLMConfig(provider=Provider.CLAUDE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)

            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                result = await provider.run("test prompt")

                assert result.success is True
                assert result.text == "Response"
                assert result.session_id == "sess-123"

    @pytest.mark.asyncio
    async def test_run_timeout(self):
        """Test run with timeout."""
        import asyncio

        mock_process = AsyncMock()
        mock_process.communicate.side_effect = asyncio.TimeoutError()
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()

        config = LLMConfig(provider=Provider.CLAUDE, timeout=1.0)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)

            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                result = await provider.run("test prompt", timeout=1.0)

                assert result.success is False
                assert result.result_type == ResultType.ERROR
                assert "Timeout" in result.error

    @pytest.mark.asyncio
    async def test_run_exception(self):
        """Test run with exception."""
        config = LLMConfig(provider=Provider.CLAUDE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config)

            with patch(
                "asyncio.create_subprocess_exec",
                side_effect=Exception("Process error"),
            ):
                result = await provider.run("test prompt")

                assert result.success is False
                assert result.result_type == ResultType.ERROR
                assert "Process error" in result.error

    @pytest.mark.asyncio
    async def test_run_with_intervention_hook(self):
        """Test run with intervention hook."""
        from u_llm_sdk.types import PreActionContext

        events = [
            {"type": "system", "session_id": "sess-123"},
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Response"}]},
            },
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (stdout.encode(), b"")
        mock_process.returncode = 0

        mock_hook = AsyncMock()
        mock_hook.on_pre_action.return_value = PreActionContext.create(
            context_text="Injected context",
            confidence=0.9,
            injection_position="prepend",
        )

        config = LLMConfig(provider=Provider.CLAUDE)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config, intervention_hook=mock_hook)

            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                result = await provider.run("test prompt")

                assert result.success is True
                mock_hook.on_pre_action.assert_called_once()
                mock_hook.on_post_action.assert_called_once()


class TestStreamMethod:
    """Test stream method."""

    @pytest.mark.asyncio
    async def test_stream_basic(self):
        """Test basic streaming."""
        events = [
            {"type": "system", "session_id": "sess-123"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello"}]}},
        ]

        async def mock_readline():
            for event in events:
                yield json.dumps(event).encode() + b"\n"
            yield b""

        mock_stdout = AsyncMock()
        readline_gen = mock_readline()
        mock_stdout.readline = lambda: readline_gen.__anext__()

        mock_process = AsyncMock()
        mock_process.stdout = mock_stdout
        mock_process.returncode = None
        # kill() is synchronous in asyncio.subprocess.Process, not async
        mock_process.kill = MagicMock()

        config = LLMConfig(provider=Provider.CLAUDE, timeout=10.0)
        with patch.object(ClaudeProvider, "get_cli_path", return_value="/usr/bin/claude"):
            provider = ClaudeProvider(config=config, validate_features=False)

            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                collected = []
                async for event in provider.stream("test"):
                    collected.append(event)

                assert len(collected) == 2
                assert collected[0]["type"] == "system"
                assert provider.session_id == "sess-123"


class TestExport:
    """Test module exports."""

    def test_all_export(self):
        """Test __all__ export."""
        from u_llm_sdk.llm.providers.claude import __all__

        assert "ClaudeProvider" in __all__
