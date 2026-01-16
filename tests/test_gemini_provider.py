"""Tests for GeminiProvider implementation."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from u_llm_sdk.types import LLMResult, Provider, ResultType, SandboxMode
from u_llm_sdk import LLMConfig, ModelTier
from u_llm_sdk.config import AutoApproval
from u_llm_sdk.llm.providers.gemini import GeminiProvider


class TestGeminiProviderInit:
    """Test GeminiProvider initialization."""

    def test_provider_constant(self):
        """Test PROVIDER constant."""
        assert GeminiProvider.PROVIDER == Provider.GEMINI

    def test_cli_name(self):
        """Test CLI_NAME constant."""
        assert GeminiProvider.CLI_NAME == "gemini"

    def test_init_with_default_config(self):
        """Test init with default config."""
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider()
            assert provider.PROVIDER == Provider.GEMINI
            assert provider.config.provider == Provider.GEMINI

    def test_init_with_custom_config(self):
        """Test init with custom config."""
        config = LLMConfig(provider=Provider.GEMINI, tier=ModelTier.HIGH)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            assert provider.config.tier == ModelTier.HIGH


class TestBuildArgs:
    """Test CLI argument building."""

    def test_build_args_basic(self):
        """Test basic argument building."""
        config = LLMConfig(provider=Provider.GEMINI)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            args = provider._build_args("test prompt")

            assert "/usr/bin/gemini" in args
            assert "test prompt" in args
            assert "-o" in args
            assert "stream-json" in args

    def test_build_args_with_model(self):
        """Test argument building with model."""
        config = LLMConfig(provider=Provider.GEMINI, model="gemini-3-pro-preview")
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            args = provider._build_args("test prompt")

            assert "-m" in args
            idx = args.index("-m")
            assert args[idx + 1] == "gemini-3-pro-preview"

    def test_build_args_with_session_id(self):
        """Test argument building with session ID (resume)."""
        config = LLMConfig(provider=Provider.GEMINI)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            args = provider._build_args("test prompt", session_id="session-123")

            # Resume uses --resume flag
            assert "--resume" in args
            idx = args.index("--resume")
            assert args[idx + 1] == "session-123"
            # Prompt should NOT be present for resume
            assert "test prompt" not in args

    def test_build_args_with_approval_auto_edit(self):
        """Test argument building with auto_edit approval."""
        config = LLMConfig(provider=Provider.GEMINI, auto_approval=AutoApproval.EDITS_ONLY)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            args = provider._build_args("test prompt")

            assert "--approval-mode" in args
            idx = args.index("--approval-mode")
            assert args[idx + 1] == "auto_edit"

    def test_build_args_with_yolo(self):
        """Test argument building with YOLO mode (full auto)."""
        config = LLMConfig(provider=Provider.GEMINI, auto_approval=AutoApproval.FULL)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            args = provider._build_args("test prompt")

            assert "-y" in args
            assert "--approval-mode" not in args

    def test_build_args_with_sandbox(self):
        """Test argument building with sandbox.

        NOTE: Gemini CLI uses -s as a BOOLEAN flag, not mode-based like Codex.
        Any non-NONE sandbox mode just enables the flag without a value.
        """
        config = LLMConfig(provider=Provider.GEMINI, sandbox=SandboxMode.READ_ONLY)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            args = provider._build_args("test prompt")

            # -s is a boolean flag in Gemini, no value follows it
            assert "-s" in args
            idx = args.index("-s")
            # Verify no sandbox mode value follows (it's boolean, not mode-based)
            if idx + 1 < len(args):
                assert args[idx + 1] not in ["read-only", "workspace-write", "full-access"]

    def test_build_args_no_sandbox_when_none(self):
        """Test no sandbox flag when SandboxMode.NONE."""
        config = LLMConfig(provider=Provider.GEMINI, sandbox=SandboxMode.NONE)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            args = provider._build_args("test prompt")

            assert "-s" not in args

    def test_build_args_with_temperature(self):
        """Test argument building with temperature."""
        config = LLMConfig(
            provider=Provider.GEMINI,
            provider_options={"temperature": 0.7},
        )
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            args = provider._build_args("test prompt")

            # Gemini CLI does NOT support --temperature flag (use ~/.gemini/settings.json)
            # Verified 2026-01-08: These options are ignored with a warning
            assert "--temperature" not in args

    def test_build_args_with_top_p(self):
        """Test argument building with top_p."""
        config = LLMConfig(
            provider=Provider.GEMINI,
            provider_options={"top_p": 0.9},
        )
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            args = provider._build_args("test prompt")

            # Gemini CLI does NOT support --top-p flag (use ~/.gemini/settings.json)
            # Verified 2026-01-08: These options are ignored with a warning
            assert "--top-p" not in args

    def test_build_args_with_top_k(self):
        """Test argument building with top_k."""
        config = LLMConfig(
            provider=Provider.GEMINI,
            provider_options={"top_k": 40},
        )
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            args = provider._build_args("test prompt")

            # Gemini CLI does NOT support --top-k flag (use ~/.gemini/settings.json)
            # Verified 2026-01-08: These options are ignored with a warning
            assert "--top-k" not in args


class TestParseOutput:
    """Test output parsing."""

    def test_parse_output_empty(self):
        """Test parsing empty output."""
        config = LLMConfig(provider=Provider.GEMINI)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            result = provider._parse_output("", "", True, 100)

            assert result.success is True
            assert result.result_type == ResultType.TEXT
            assert result.text == ""

    def test_parse_output_text_only(self):
        """Test parsing text-only output."""
        events = [
            {"type": "message", "text": "Hello, world!"},
            {"session_id": "sess-123"},
            {"type": "done", "usage": {"promptTokenCount": 10, "candidatesTokenCount": 20}},
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        config = LLMConfig(provider=Provider.GEMINI)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            result = provider._parse_output(stdout, "", True, 100)

            assert result.success is True
            assert result.result_type == ResultType.TEXT
            assert result.text == "Hello, world!"
            assert result.session_id == "sess-123"
            assert result.token_usage is not None
            assert result.token_usage.input_tokens == 10
            assert result.token_usage.output_tokens == 20

    def test_parse_output_assistant_format(self):
        """Test parsing assistant event format."""
        events = [
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Parsed content"}]
                },
            },
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        config = LLMConfig(provider=Provider.GEMINI)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            result = provider._parse_output(stdout, "", True, 100)

            assert result.text == "Parsed content"

    def test_parse_output_with_file_edit(self):
        """Test parsing output with file edit."""
        events = [
            {"session_id": "sess-123"},
            {
                "type": "tool_use",
                "name": "edit_file",
                "input": {
                    "file_path": "/test/file.py",
                    "content": "print('hello')",
                },
            },
            {"type": "message", "text": "File edited."},
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        config = LLMConfig(provider=Provider.GEMINI)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            result = provider._parse_output(stdout, "", True, 100)

            assert result.success is True
            assert result.result_type == ResultType.MIXED
            assert len(result.files_modified) == 1
            assert result.files_modified[0].path == "/test/file.py"

    def test_parse_output_with_command(self):
        """Test parsing output with shell command."""
        events = [
            {"session_id": "sess-123"},
            {
                "type": "tool_use",
                "name": "shell",
                "input": {"command": "ls -la"},
            },
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        config = LLMConfig(provider=Provider.GEMINI)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            result = provider._parse_output(stdout, "", True, 100)

            assert result.success is True
            assert result.result_type == ResultType.COMMAND
            assert len(result.commands_run) == 1
            assert result.commands_run[0].command == "ls -la"

    def test_parse_output_error(self):
        """Test parsing failed output."""
        config = LLMConfig(provider=Provider.GEMINI)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            result = provider._parse_output("", "Error message", False, 100)

            assert result.success is False
            assert result.result_type == ResultType.ERROR
            assert result.error == "Error message"

    def test_parse_output_tool_call_format(self):
        """Test parsing tool_call event format."""
        events = [
            {
                "type": "tool_call",
                "tool": "run_command",
                "args": {"cmd": "pwd"},
            },
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        config = LLMConfig(provider=Provider.GEMINI)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            result = provider._parse_output(stdout, "", True, 100)

            assert result.result_type == ResultType.COMMAND
            assert len(result.commands_run) == 1
            assert result.commands_run[0].command == "pwd"


class TestDetermineResultType:
    """Test result type determination."""

    def test_determine_error(self):
        """Test error result type."""
        config = LLMConfig(provider=Provider.GEMINI)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            result = provider._determine_result_type([], [], [], False)
            assert result == ResultType.ERROR

    def test_determine_text_only(self):
        """Test text-only result type."""
        config = LLMConfig(provider=Provider.GEMINI)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            result = provider._determine_result_type(["hello"], [], [], True)
            assert result == ResultType.TEXT

    def test_determine_file_edit(self):
        """Test file edit result type."""
        from u_llm_sdk.types import FileChange

        config = LLMConfig(provider=Provider.GEMINI)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            files = [FileChange(path="/test.py", action="modified")]
            result = provider._determine_result_type([], files, [], True)
            assert result == ResultType.FILE_EDIT

    def test_determine_command(self):
        """Test command result type."""
        from u_llm_sdk.types import CommandRun

        config = LLMConfig(provider=Provider.GEMINI)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            commands = [CommandRun(command="ls", exit_code=0)]
            result = provider._determine_result_type([], [], commands, True)
            assert result == ResultType.COMMAND

    def test_determine_mixed(self):
        """Test mixed result type."""
        from u_llm_sdk.types import FileChange

        config = LLMConfig(provider=Provider.GEMINI)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            files = [FileChange(path="/test.py", action="modified")]
            result = provider._determine_result_type(["hello"], files, [], True)
            assert result == ResultType.MIXED


class TestRunMethod:
    """Test run method with mocked subprocess."""

    @pytest.mark.asyncio
    async def test_run_success(self):
        """Test successful run."""
        events = [
            {"session_id": "sess-123"},
            {"type": "message", "text": "Response"},
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (stdout.encode(), b"")
        mock_process.returncode = 0

        config = LLMConfig(provider=Provider.GEMINI)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)

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

        config = LLMConfig(provider=Provider.GEMINI, timeout=1.0)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)

            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                result = await provider.run("test prompt", timeout=1.0)

                assert result.success is False
                assert result.result_type == ResultType.ERROR
                assert "Timeout" in result.error

    @pytest.mark.asyncio
    async def test_run_exception(self):
        """Test run with exception."""
        config = LLMConfig(provider=Provider.GEMINI)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)

            with patch(
                "asyncio.create_subprocess_exec",
                side_effect=Exception("Process error"),
            ):
                result = await provider.run("test prompt")

                assert result.success is False
                assert result.result_type == ResultType.ERROR
                assert "Process error" in result.error


class TestStreamMethod:
    """Test stream method."""

    @pytest.mark.asyncio
    async def test_stream_basic(self):
        """Test basic streaming."""
        events = [
            {"session_id": "sess-123"},
            {"type": "message", "text": "Hello"},
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

        config = LLMConfig(provider=Provider.GEMINI, timeout=10.0)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config, validate_features=False)

            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                collected = []
                async for event in provider.stream("test"):
                    collected.append(event)

                assert len(collected) == 2
                assert provider.session_id == "sess-123"


class TestNewCLIFeatures:
    """Test new CLI features added for Gemini provider."""

    def test_build_args_with_allowed_tools_list(self):
        """Test argument building with allowed_tools as list."""
        config = LLMConfig(
            provider=Provider.GEMINI,
            provider_options={"allowed_tools": ["edit", "bash", "read"]},
        )
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            args = provider._build_args("test prompt")

            assert "--allowed-tools" in args
            idx = args.index("--allowed-tools")
            # Should be followed by the tools as separate arguments
            assert args[idx + 1] == "edit"
            assert args[idx + 2] == "bash"
            assert args[idx + 3] == "read"

    def test_build_args_with_allowed_tools_string(self):
        """Test argument building with allowed_tools as string."""
        config = LLMConfig(
            provider=Provider.GEMINI,
            provider_options={"allowed_tools": "edit bash"},
        )
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            args = provider._build_args("test prompt")

            assert "--allowed-tools" in args
            idx = args.index("--allowed-tools")
            assert args[idx + 1] == "edit bash"

    def test_build_args_with_extensions_list(self):
        """Test argument building with extensions as list."""
        config = LLMConfig(
            provider=Provider.GEMINI,
            provider_options={"extensions": ["code-review", "analysis"]},
        )
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            args = provider._build_args("test prompt")

            # Each extension should have its own -e flag
            assert args.count("-e") == 2
            e_indices = [i for i, x in enumerate(args) if x == "-e"]
            assert args[e_indices[0] + 1] == "code-review"
            assert args[e_indices[1] + 1] == "analysis"

    def test_build_args_with_extensions_string(self):
        """Test argument building with extensions as string."""
        config = LLMConfig(
            provider=Provider.GEMINI,
            provider_options={"extensions": "code-review"},
        )
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            args = provider._build_args("test prompt")

            assert "-e" in args
            idx = args.index("-e")
            assert args[idx + 1] == "code-review"

    def test_build_args_with_include_directories_list(self):
        """Test argument building with include_directories as list."""
        config = LLMConfig(
            provider=Provider.GEMINI,
            provider_options={"include_directories": ["/path/to/dir1", "/path/to/dir2"]},
        )
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            args = provider._build_args("test prompt")

            assert "--include-directories" in args
            idx = args.index("--include-directories")
            assert args[idx + 1] == "/path/to/dir1"
            assert args[idx + 2] == "/path/to/dir2"

    def test_build_args_with_include_directories_string(self):
        """Test argument building with include_directories as string."""
        config = LLMConfig(
            provider=Provider.GEMINI,
            provider_options={"include_directories": "/path/to/dir"},
        )
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            args = provider._build_args("test prompt")

            assert "--include-directories" in args
            idx = args.index("--include-directories")
            assert args[idx + 1] == "/path/to/dir"

    def test_build_args_with_allowed_mcp_server_names_list(self):
        """Test argument building with allowed_mcp_server_names as list."""
        config = LLMConfig(
            provider=Provider.GEMINI,
            provider_options={"allowed_mcp_server_names": ["server1", "server2"]},
        )
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            args = provider._build_args("test prompt")

            assert "--allowed-mcp-server-names" in args
            idx = args.index("--allowed-mcp-server-names")
            assert args[idx + 1] == "server1"
            assert args[idx + 2] == "server2"

    def test_build_args_with_allowed_mcp_server_names_string(self):
        """Test argument building with allowed_mcp_server_names as string."""
        config = LLMConfig(
            provider=Provider.GEMINI,
            provider_options={"allowed_mcp_server_names": "server1"},
        )
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            args = provider._build_args("test prompt")

            assert "--allowed-mcp-server-names" in args
            idx = args.index("--allowed-mcp-server-names")
            assert args[idx + 1] == "server1"

    def test_build_args_with_session_resume_latest(self):
        """Test argument building with 'latest' session resume."""
        config = LLMConfig(provider=Provider.GEMINI)
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            args = provider._build_args("test prompt", session_id="latest")

            assert "--resume" in args
            idx = args.index("--resume")
            assert args[idx + 1] == "latest"
            # Prompt should NOT be present for resume
            assert "test prompt" not in args

    def test_build_args_combined_features(self):
        """Test argument building with multiple new features combined.

        Note: temperature/top_p/top_k are NOT CLI flags in Gemini CLI.
        Use ~/.gemini/settings.json for sampling parameters.
        These options will be ignored with a warning.
        """
        config = LLMConfig(
            provider=Provider.GEMINI,
            model="gemini-3-pro-preview",
            auto_approval=AutoApproval.EDITS_ONLY,
            sandbox=SandboxMode.WORKSPACE_WRITE,
            provider_options={
                "temperature": 0.8,  # Ignored - settings-based only
                "allowed_tools": ["edit", "bash"],
                "extensions": ["analysis"],
                "include_directories": ["/path/to/dir"],
                "allowed_mcp_server_names": ["server1"],
            },
        )
        with patch.object(GeminiProvider, "get_cli_path", return_value="/usr/bin/gemini"):
            provider = GeminiProvider(config=config)
            args = provider._build_args("test prompt")

            # Verify supported features are present
            assert "-m" in args
            assert "gemini-3-pro-preview" in args
            assert "--approval-mode" in args
            assert "auto_edit" in args
            assert "-s" in args
            # temperature is NOT a CLI flag (use settings.json)
            assert "--temperature" not in args
            assert "--allowed-tools" in args
            assert "edit" in args
            assert "-e" in args
            assert "analysis" in args
            assert "--include-directories" in args
            assert "/path/to/dir" in args
            assert "--allowed-mcp-server-names" in args
            assert "server1" in args


class TestExport:
    """Test module exports."""

    def test_all_export(self):
        """Test __all__ export."""
        from u_llm_sdk.llm.providers.gemini import __all__

        assert "GeminiProvider" in __all__

    def test_provider_import(self):
        """Test provider can be imported from parent module."""
        from u_llm_sdk.llm.providers import GeminiProvider as GP

        assert GP.PROVIDER == Provider.GEMINI
