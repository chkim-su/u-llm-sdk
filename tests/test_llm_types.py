"""Tests for llm-types package.

These tests verify that the shared type definitions work correctly
and serialize/deserialize properly for HTTP communication.
"""

import pytest
from u_llm_sdk.types import (
    Provider,
    ModelTier,
    ResultType,
    LLMResult,
    TokenUsage,
    FileChange,
    CommandRun,
    CodeBlock,
    PreActionContext,
    PostActionFeedback,
    UnifiedLLMError,
    RAGConnectionError,
    RAGTimeoutError,
)


class TestEnums:
    """Test enum definitions."""

    def test_provider_enum(self):
        """Verify Provider enum values."""
        assert Provider.CLAUDE.value == "claude"
        assert Provider.CODEX.value == "codex"
        assert Provider.GEMINI.value == "gemini"

    def test_model_tier_enum(self):
        """Verify ModelTier enum values."""
        assert ModelTier.HIGH is not None
        assert ModelTier.LOW is not None

    def test_result_type_enum(self):
        """Verify ResultType enum values."""
        assert ResultType.TEXT.value == "text"
        assert ResultType.CODE.value == "code"
        assert ResultType.ERROR.value == "error"


class TestTokenUsage:
    """Test TokenUsage serialization."""

    def test_to_dict(self):
        """Verify TokenUsage serializes correctly."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cached_tokens=20,
            reasoning_tokens=10,
        )
        data = usage.to_dict()

        assert data["input_tokens"] == 100
        assert data["output_tokens"] == 50
        assert data["total_tokens"] == 150
        assert data["cached_tokens"] == 20
        assert data["reasoning_tokens"] == 10

    def test_from_dict(self):
        """Verify TokenUsage deserializes correctly."""
        data = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "cached_tokens": 20,
            "reasoning_tokens": 10,
        }
        usage = TokenUsage.from_dict(data)

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_roundtrip(self):
        """Verify TokenUsage roundtrip serialization."""
        original = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
        data = original.to_dict()
        restored = TokenUsage.from_dict(data)

        assert restored.input_tokens == original.input_tokens
        assert restored.output_tokens == original.output_tokens
        assert restored.total_tokens == original.total_tokens


class TestFileChange:
    """Test FileChange serialization."""

    def test_to_dict(self):
        """Verify FileChange serializes correctly."""
        change = FileChange(
            path="/path/to/file.py",
            action="modified",
            diff="+ new line",
        )
        data = change.to_dict()

        assert data["path"] == "/path/to/file.py"
        assert data["action"] == "modified"
        assert data["diff"] == "+ new line"

    def test_from_dict(self):
        """Verify FileChange deserializes correctly."""
        data = {
            "path": "/path/to/file.py",
            "action": "created",
            "diff": None,
        }
        change = FileChange.from_dict(data)

        assert change.path == "/path/to/file.py"
        assert change.action == "created"
        assert change.diff is None


class TestLLMResult:
    """Test LLMResult serialization."""

    def test_to_dict_basic(self):
        """Verify LLMResult serializes basic fields."""
        result = LLMResult(
            success=True,
            provider="claude",
            model="opus-4",
            text="Hello, world!",
            result_type=ResultType.TEXT,
        )
        data = result.to_dict()

        assert data["success"] is True
        assert data["text"] == "Hello, world!"
        assert data["result_type"] == "text"
        assert data["provider"] == "claude"
        assert data["model"] == "opus-4"

    def test_to_dict_with_nested(self):
        """Verify LLMResult serializes nested objects."""
        result = LLMResult(
            success=True,
            provider="claude",
            model="opus-4",
            text="Done",
            result_type=ResultType.MIXED,
            token_usage=TokenUsage(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
            ),
            files_modified=[
                FileChange(path="test.py", action="modified"),
            ],
        )
        data = result.to_dict()

        assert data["token_usage"]["input_tokens"] == 100
        assert len(data["files_modified"]) == 1
        assert data["files_modified"][0]["path"] == "test.py"

    def test_from_dict(self):
        """Verify LLMResult deserializes correctly."""
        data = {
            "success": True,
            "text": "Test result",
            "result_type": "text",
            "provider": "claude",
            "model": "opus-4",
            "token_usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        }
        result = LLMResult.from_dict(data)

        assert result.success is True
        assert result.text == "Test result"
        assert result.token_usage.input_tokens == 100


class TestPreActionContext:
    """Test PreActionContext serialization."""

    def test_create_and_serialize(self):
        """Verify PreActionContext creation and serialization."""
        context = PreActionContext.create(
            context_text="Historical context here",
            confidence=0.85,
        )

        assert context.context_text == "Historical context here"
        assert context.confidence == 0.85
        assert context.injection_id is not None

        # Verify serialization
        data = context.to_dict()
        assert data["context_text"] == "Historical context here"
        assert data["confidence"] == 0.85

    def test_roundtrip(self):
        """Verify PreActionContext roundtrip serialization."""
        original = PreActionContext.create(
            context_text="Test context",
            confidence=0.9,
        )
        data = original.to_dict()
        restored = PreActionContext.from_dict(data)

        assert restored.context_text == original.context_text
        assert restored.confidence == original.confidence
        assert restored.injection_id == original.injection_id


class TestPostActionFeedback:
    """Test PostActionFeedback serialization."""

    def test_from_result(self):
        """Verify PostActionFeedback created from LLMResult."""
        result = LLMResult(
            success=True,
            provider="claude",
            model="opus-4",
            text="Done",
            result_type=ResultType.TEXT,
            token_usage=TokenUsage(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
            ),
        )

        feedback = PostActionFeedback.from_result(
            result=result,
            run_id="run-123",
            injection_id="inj-456",
        )

        assert feedback.success is True
        assert feedback.result_type == "text"
        assert feedback.run_id == "run-123"
        assert feedback.injection_id == "inj-456"

    def test_roundtrip(self):
        """Verify PostActionFeedback roundtrip serialization."""
        result = LLMResult(
            success=True,
            provider="claude",
            model="opus-4",
            text="Test",
            result_type=ResultType.CODE,
        )
        original = PostActionFeedback.from_result(result, "run-1")

        data = original.to_dict()
        restored = PostActionFeedback.from_dict(data)

        assert restored.success == original.success
        assert restored.result_type == original.result_type
        assert restored.run_id == original.run_id


class TestExceptions:
    """Test exception classes."""

    def test_rag_connection_error(self):
        """Verify RAGConnectionError attributes."""
        error = RAGConnectionError(url="http://localhost:8000", reason="timeout")
        assert "localhost:8000" in str(error)
        assert error.url == "http://localhost:8000"
        assert error.reason == "timeout"

    def test_rag_timeout_error(self):
        """Verify RAGTimeoutError attributes."""
        error = RAGTimeoutError(timeout_ms=500, endpoint="/pre-action")
        assert "500ms" in str(error)
        assert error.timeout_ms == 500
        assert error.endpoint == "/pre-action"

    def test_exception_hierarchy(self):
        """Verify exception inheritance."""
        assert issubclass(RAGConnectionError, UnifiedLLMError)
        assert issubclass(RAGTimeoutError, UnifiedLLMError)
