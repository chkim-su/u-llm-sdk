"""Tests for CodexProvider implementation."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from u_llm_sdk.types import LLMResult, Provider, ResultType, SandboxMode
from u_llm_sdk import LLMConfig, ModelTier
from u_llm_sdk.config import AutoApproval, ReasoningLevel
from u_llm_sdk.llm.providers.codex import CodexProvider


class TestCodexProviderInit:
    """Test CodexProvider initialization."""

    def test_provider_constant(self):
        """Test PROVIDER constant."""
        assert CodexProvider.PROVIDER == Provider.CODEX

    def test_cli_name(self):
        """Test CLI_NAME constant."""
        assert CodexProvider.CLI_NAME == "codex"

    def test_init_with_default_config(self):
        """Test init with default config."""
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider()
            assert provider.PROVIDER == Provider.CODEX
            assert provider.config.provider == Provider.CODEX

    def test_init_with_custom_config(self):
        """Test init with custom config."""
        config = LLMConfig(provider=Provider.CODEX, tier=ModelTier.HIGH)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            assert provider.config.tier == ModelTier.HIGH


class TestBuildArgs:
    """Test CLI argument building."""

    def test_build_args_basic(self):
        """Test basic argument building."""
        config = LLMConfig(provider=Provider.CODEX, reasoning_level=ReasoningLevel.NONE)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            args = provider._build_args("test prompt")

            assert "/usr/bin/codex" in args
            assert "exec" in args
            assert "test prompt" in args
            assert "--json" in args

    def test_build_args_with_model(self):
        """Test argument building with model."""
        config = LLMConfig(provider=Provider.CODEX, model="gpt-5.2")
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            args = provider._build_args("test prompt")

            assert "-m" in args
            idx = args.index("-m")
            assert args[idx + 1] == "gpt-5.2"

    def test_build_args_with_session_id(self):
        """Test argument building with session ID (resume)."""
        config = LLMConfig(provider=Provider.CODEX)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            args = provider._build_args("test prompt", session_id="session-123")

            # Resume uses different subcommand
            assert "resume" in args
            assert "session-123" in args
            # exec should NOT be present for resume
            assert "exec" not in args

    def test_build_args_with_approval_none(self):
        """Test argument building with NONE approval (default behavior, no flags).

        NOTE: codex exec does NOT support -a flag. AutoApproval.NONE
        results in default interactive behavior with no additional flags.
        """
        config = LLMConfig(provider=Provider.CODEX, auto_approval=AutoApproval.NONE)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config, validate_features=False)
            args = provider._build_args("test prompt")

            # -a flag doesn't exist in codex exec
            assert "-a" not in args
            # No auto flags should be added
            assert "--full-auto" not in args

    def test_build_args_with_approval_edits_only(self):
        """Test argument building with EDITS_ONLY approval (not supported).

        NOTE: codex exec does NOT support -a on-failure. AutoApproval.EDITS_ONLY
        has no CLI equivalent and falls back to default behavior with a warning.
        """
        config = LLMConfig(provider=Provider.CODEX, auto_approval=AutoApproval.EDITS_ONLY)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            # Suppress warning log during test
            with patch("u_llm_sdk.llm.providers.codex.logger"):
                provider = CodexProvider(config=config, validate_features=False)
                args = provider._build_args("test prompt")

                # -a flag doesn't exist in codex exec
                assert "-a" not in args
                # Feature is unsupported, no auto flags added
                assert "--full-auto" not in args

    def test_build_args_with_approval_full(self):
        """Test argument building with FULL approval (--full-auto).

        NOTE: codex exec uses --full-auto for automatic approval, NOT -a flag.
        """
        config = LLMConfig(provider=Provider.CODEX, auto_approval=AutoApproval.FULL)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config, validate_features=False)
            args = provider._build_args("test prompt")

            # Should use --full-auto, not -a
            assert "--full-auto" in args
            # -a flag doesn't exist in codex exec
            assert "-a" not in args

    def test_build_args_with_sandbox_read_only(self):
        """Test argument building with read-only sandbox."""
        config = LLMConfig(provider=Provider.CODEX, sandbox=SandboxMode.READ_ONLY)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            args = provider._build_args("test prompt")

            assert "-s" in args
            idx = args.index("-s")
            assert args[idx + 1] == "read-only"

    def test_build_args_with_sandbox_workspace_write(self):
        """Test argument building with workspace-write sandbox."""
        config = LLMConfig(provider=Provider.CODEX, sandbox=SandboxMode.WORKSPACE_WRITE)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            args = provider._build_args("test prompt")

            assert "-s" in args
            idx = args.index("-s")
            assert args[idx + 1] == "workspace-write"

    def test_build_args_no_sandbox_when_none(self):
        """Test no sandbox flag when SandboxMode.NONE."""
        config = LLMConfig(provider=Provider.CODEX, sandbox=SandboxMode.NONE)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            args = provider._build_args("test prompt")

            assert "-s" not in args

    def test_build_args_with_reasoning_effort(self):
        """Test argument building with reasoning effort.

        NOTE: --model-reasoning-effort flag is NOT available in codex exec CLI.
        Reasoning level is ignored for Codex provider.
        """
        config = LLMConfig(provider=Provider.CODEX, reasoning_level=ReasoningLevel.HIGH)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            args = provider._build_args("test prompt")

            # --model-reasoning-effort is NOT available in codex exec
            assert "--model-reasoning-effort" not in args

    def test_build_args_with_cwd(self):
        """Test argument building with working directory."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMConfig(provider=Provider.CODEX, cwd=tmpdir)
            with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
                provider = CodexProvider(config=config)
                args = provider._build_args("test prompt")

                assert "-C" in args
                idx = args.index("-C")
                assert args[idx + 1] == tmpdir

    def test_build_args_with_skip_git_check(self):
        """Test argument building with skip git check."""
        config = LLMConfig(
            provider=Provider.CODEX,
            provider_options={"skip_git_repo_check": True},
        )
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            args = provider._build_args("test prompt")

            assert "--skip-git-repo-check" in args

    def test_build_args_with_config_file(self):
        """Test argument building with config file.

        Note: The old config file option has been replaced by config_overrides.
        This test has been updated to reflect the new behavior.
        """
        # Old behavior: {"config": "/path/to/config.json"} is no longer supported
        # New behavior: Use config_overrides for key-value pairs
        config = LLMConfig(
            provider=Provider.CODEX,
            provider_options={"config_overrides": {"config_file": "/path/to/config.json"}},
        )
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            args = provider._build_args("test prompt")

            assert "-c" in args
            assert "config_file=/path/to/config.json" in args

    def test_build_args_with_images(self):
        """Test argument building with image inputs."""
        config = LLMConfig(
            provider=Provider.CODEX,
            provider_options={"images": ["/path/to/image1.png", "/path/to/image2.jpg"]},
        )
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            args = provider._build_args("test prompt")

            # Check for -i flags
            assert args.count("-i") == 2
            assert "/path/to/image1.png" in args
            assert "/path/to/image2.jpg" in args

    def test_build_args_with_search(self):
        """Test argument building with web search (not supported).

        NOTE: --search flag does NOT exist in codex exec CLI.
        The option is ignored with a warning.
        """
        config = LLMConfig(
            provider=Provider.CODEX,
            provider_options={"search": True},
        )
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            # Suppress warning log during test
            with patch("u_llm_sdk.llm.providers.codex.logger"):
                provider = CodexProvider(config=config, validate_features=False)
                args = provider._build_args("test prompt")

                # --search IS supported in codex exec CLI v0.71.0+ (verified 2026-01-08)
                assert "--search" in args

    def test_build_args_with_full_auto(self):
        """Test argument building with full auto mode."""
        config = LLMConfig(
            provider=Provider.CODEX,
            provider_options={"full_auto": True},
        )
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            args = provider._build_args("test prompt")

            assert "--full-auto" in args

    def test_build_args_with_output_schema(self):
        """Test argument building with output schema."""
        config = LLMConfig(
            provider=Provider.CODEX,
            provider_options={"output_schema": "/path/to/schema.json"},
        )
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            args = provider._build_args("test prompt")

            assert "--output-schema" in args
            idx = args.index("--output-schema")
            assert args[idx + 1] == "/path/to/schema.json"

    def test_build_args_with_add_dirs(self):
        """Test argument building with additional directories."""
        config = LLMConfig(
            provider=Provider.CODEX,
            provider_options={"add_dirs": ["/path/to/dir1", "/path/to/dir2"]},
        )
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            args = provider._build_args("test prompt")

            assert args.count("--add-dir") == 2
            assert "/path/to/dir1" in args
            assert "/path/to/dir2" in args

    def test_build_args_with_features_enable(self):
        """Test argument building with feature enable flags."""
        config = LLMConfig(
            provider=Provider.CODEX,
            provider_options={"features_enable": ["feature1", "feature2"]},
        )
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            args = provider._build_args("test prompt")

            assert args.count("--enable") == 2
            assert "feature1" in args
            assert "feature2" in args

    def test_build_args_with_features_disable(self):
        """Test argument building with feature disable flags."""
        config = LLMConfig(
            provider=Provider.CODEX,
            provider_options={"features_disable": ["feature1", "feature2"]},
        )
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            args = provider._build_args("test prompt")

            assert args.count("--disable") == 2
            assert "feature1" in args
            assert "feature2" in args

    def test_build_args_with_config_overrides(self):
        """Test argument building with config overrides."""
        config = LLMConfig(
            provider=Provider.CODEX,
            provider_options={"config_overrides": {"key1": "value1", "key2": "value2"}},
        )
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            args = provider._build_args("test prompt")

            assert args.count("-c") == 2
            # Check that both key=value pairs are present
            assert "key1=value1" in args
            assert "key2=value2" in args

    def test_build_args_with_all_features(self):
        """Test argument building with all features combined.

        NOTE: Tests corrected CLI flags for codex exec 0.71.0+:
        - AutoApproval.FULL → --full-auto (NOT -a never)
        - search option is NOT supported (ignored with warning)
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LLMConfig(
                provider=Provider.CODEX,
                model="gpt-5.2",
                auto_approval=AutoApproval.FULL,
                sandbox=SandboxMode.WORKSPACE_WRITE,
                cwd=tmpdir,
                provider_options={
                    "images": ["/path/to/image.png"],
                    # "search": True is intentionally omitted (not supported)
                    "full_auto": True,  # Redundant with AutoApproval.FULL but valid
                    "output_schema": "/path/to/schema.json",
                    "add_dirs": ["/path/to/dir"],
                    "features_enable": ["feature1"],
                    "features_disable": ["feature2"],
                    "config_overrides": {"key": "value"},
                    "skip_git_repo_check": True,
                },
            )
            with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
                provider = CodexProvider(config=config, validate_features=False)
                args = provider._build_args("test prompt")

                # Check all SUPPORTED features are present
                assert "exec" in args
                assert "test prompt" in args
                assert "--json" in args
                assert "-m" in args
                assert "gpt-5.2" in args
                # AutoApproval.FULL → --full-auto (NOT -a)
                assert "--full-auto" in args
                assert "-a" not in args  # -a flag doesn't exist in codex exec
                assert "-s" in args
                assert "workspace-write" in args
                assert "-C" in args
                assert tmpdir in args
                assert "-i" in args
                assert "/path/to/image.png" in args
                # --search is NOT supported, should not be present
                assert "--search" not in args
                assert "--output-schema" in args
                assert "/path/to/schema.json" in args
                assert "--add-dir" in args
                assert "/path/to/dir" in args
                assert "--enable" in args
                assert "feature1" in args
                assert "--disable" in args
                assert "feature2" in args
                assert "-c" in args
                assert "key=value" in args
                assert "--skip-git-repo-check" in args


class TestParseOutput:
    """Test output parsing."""

    def test_parse_output_empty(self):
        """Test parsing empty output."""
        config = LLMConfig(provider=Provider.CODEX)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            result = provider._parse_output("", "", True, 100)

            assert result.success is True
            assert result.result_type == ResultType.TEXT
            assert result.text == ""

    def test_parse_output_text_only(self):
        """Test parsing text-only output."""
        events = [
            {"type": "message", "content": "Hello, world!"},
            {"session_id": "sess-123"},
            {"type": "done", "usage": {"prompt_tokens": 10, "completion_tokens": 20}},
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        config = LLMConfig(provider=Provider.CODEX)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            result = provider._parse_output(stdout, "", True, 100)

            assert result.success is True
            assert result.result_type == ResultType.TEXT
            assert result.text == "Hello, world!"
            assert result.session_id == "sess-123"
            assert result.token_usage is not None
            assert result.token_usage.input_tokens == 10
            assert result.token_usage.output_tokens == 20

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
            {"type": "message", "content": "File edited."},
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        config = LLMConfig(provider=Provider.CODEX)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
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

        config = LLMConfig(provider=Provider.CODEX)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            result = provider._parse_output(stdout, "", True, 100)

            assert result.success is True
            assert result.result_type == ResultType.COMMAND
            assert len(result.commands_run) == 1
            assert result.commands_run[0].command == "ls -la"

    def test_parse_output_resume(self):
        """Test parsing resume output (plain text)."""
        config = LLMConfig(provider=Provider.CODEX)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            result = provider._parse_output("Session resumed.\nContinuing...", "", True, 100, is_resume=True)

            assert result.success is True
            assert result.result_type == ResultType.TEXT
            assert result.text == "Session resumed.\nContinuing..."
            # Text is parsed as summary for resume mode
            assert "Session resumed" in result.summary

    def test_parse_output_error(self):
        """Test parsing failed output."""
        config = LLMConfig(provider=Provider.CODEX)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            result = provider._parse_output("", "Error message", False, 100)

            assert result.success is False
            assert result.result_type == ResultType.ERROR
            assert result.error == "Error message"

    def test_parse_output_function_call_format(self):
        """Test parsing function_call event format."""
        events = [
            {
                "type": "function_call",
                "function": "shell_command",
                "arguments": json.dumps({"cmd": "pwd"}),
            },
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        config = LLMConfig(provider=Provider.CODEX)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            result = provider._parse_output(stdout, "", True, 100)

            assert result.result_type == ResultType.COMMAND
            assert len(result.commands_run) == 1
            assert result.commands_run[0].command == "pwd"


class TestDetermineResultType:
    """Test result type determination."""

    def test_determine_error(self):
        """Test error result type."""
        config = LLMConfig(provider=Provider.CODEX)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            result = provider._determine_result_type([], [], [], False)
            assert result == ResultType.ERROR

    def test_determine_text_only(self):
        """Test text-only result type."""
        config = LLMConfig(provider=Provider.CODEX)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            result = provider._determine_result_type(["hello"], [], [], True)
            assert result == ResultType.TEXT

    def test_determine_file_edit(self):
        """Test file edit result type."""
        from u_llm_sdk.types import FileChange

        config = LLMConfig(provider=Provider.CODEX)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            files = [FileChange(path="/test.py", action="modified")]
            result = provider._determine_result_type([], files, [], True)
            assert result == ResultType.FILE_EDIT

    def test_determine_command(self):
        """Test command result type."""
        from u_llm_sdk.types import CommandRun

        config = LLMConfig(provider=Provider.CODEX)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
            commands = [CommandRun(command="ls", exit_code=0)]
            result = provider._determine_result_type([], [], commands, True)
            assert result == ResultType.COMMAND

    def test_determine_mixed(self):
        """Test mixed result type."""
        from u_llm_sdk.types import FileChange

        config = LLMConfig(provider=Provider.CODEX)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)
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
            {"type": "message", "content": "Response"},
        ]
        stdout = "\n".join(json.dumps(e) for e in events)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (stdout.encode(), b"")
        mock_process.returncode = 0

        config = LLMConfig(provider=Provider.CODEX)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)

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

        config = LLMConfig(provider=Provider.CODEX, timeout=1.0)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)

            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                result = await provider.run("test prompt", timeout=1.0)

                assert result.success is False
                assert result.result_type == ResultType.ERROR
                assert "Timeout" in result.error

    @pytest.mark.asyncio
    async def test_run_exception(self):
        """Test run with exception."""
        config = LLMConfig(provider=Provider.CODEX)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)

            with patch(
                "asyncio.create_subprocess_exec",
                side_effect=Exception("Process error"),
            ):
                result = await provider.run("test prompt")

                assert result.success is False
                assert result.result_type == ResultType.ERROR
                assert "Process error" in result.error

    @pytest.mark.asyncio
    async def test_run_resume(self):
        """Test run with resume."""
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"Resumed content", b"")
        mock_process.returncode = 0

        config = LLMConfig(provider=Provider.CODEX)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config)

            with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
                result = await provider.run("test prompt", session_id="old-session")

                assert result.success is True
                assert result.text == "Resumed content"
                # Check that resume subcommand was used
                call_args = mock_exec.call_args
                assert "resume" in call_args[0]
                assert "old-session" in call_args[0]


class TestStreamMethod:
    """Test stream method."""

    @pytest.mark.asyncio
    async def test_stream_basic(self):
        """Test basic streaming."""
        events = [
            {"session_id": "sess-123"},
            {"type": "message", "content": "Hello"},
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

        config = LLMConfig(provider=Provider.CODEX, timeout=10.0)
        with patch.object(CodexProvider, "get_cli_path", return_value="/usr/bin/codex"):
            provider = CodexProvider(config=config, validate_features=False)

            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                collected = []
                async for event in provider.stream("test"):
                    collected.append(event)

                assert len(collected) == 2
                assert provider.session_id == "sess-123"


class TestExport:
    """Test module exports."""

    def test_all_export(self):
        """Test __all__ export."""
        from u_llm_sdk.llm.providers.codex import __all__

        assert "CodexProvider" in __all__

    def test_provider_import(self):
        """Test provider can be imported from parent module."""
        from u_llm_sdk.llm.providers import CodexProvider as CP

        assert CP.PROVIDER == Provider.CODEX
