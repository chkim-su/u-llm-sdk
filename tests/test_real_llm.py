"""Real LLM calling tests - requires actual API access."""
import pytest
from u_llm_sdk import LLM, LLMSync, Provider, LLMResult
from u_llm_sdk.config import LLMConfig


class TestRealLLMCalls:
    """Test actual LLM API calls."""

    @pytest.mark.asyncio
    async def test_claude_real_call(self):
        """Test real Claude API call."""
        config = LLMConfig(
            provider=Provider.CLAUDE,
            timeout=60,
        )

        async with LLM(config=config) as llm:
            result = await llm.run("Say 'hello' and nothing else.")

        assert isinstance(result, LLMResult)
        assert result.success is True
        assert "hello" in result.text.lower()
        print(f"\nClaude response: {result.text[:100]}")
        print(f"Result type: {result.result_type}")

    def test_claude_sync_real_call(self):
        """Test real Claude API call (sync)."""
        config = LLMConfig(
            provider=Provider.CLAUDE,
            timeout=60,
        )

        llm = LLMSync(config=config)
        result = llm.run("Say 'world' and nothing else.")

        assert isinstance(result, LLMResult)
        assert result.success is True
        assert "world" in result.text.lower()
        print(f"\nClaude sync response: {result.text[:100]}")

    @pytest.mark.asyncio
    async def test_claude_streaming(self):
        """Test Claude streaming."""
        config = LLMConfig(
            provider=Provider.CLAUDE,
            timeout=60,
        )

        chunks = []
        async with LLM(config=config) as llm:
            async for chunk in llm.stream("Count from 1 to 3."):
                chunks.append(chunk)
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    print(chunk.get("content", ""), end="", flush=True)

        print()  # newline
        assert len(chunks) > 0
        print(f"Received {len(chunks)} chunks")


if __name__ == "__main__":
    import asyncio

    async def main():
        test = TestRealLLMCalls()
        print("Testing Claude async call...")
        await test.test_claude_real_call()
        print("\nTesting Claude sync call...")
        test.test_claude_sync_real_call()
        print("\nTesting Claude streaming...")
        await test.test_claude_streaming()
        print("\nAll real LLM tests passed!")

    asyncio.run(main())
