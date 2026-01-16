"""Example usage of RAGClient for MV-rag API integration.

This example demonstrates how to use RAGClient to integrate U-llm-sdk with MV-rag
for experience learning and context injection.
"""

import asyncio
from u_llm_sdk.types import LLMResult, ResultType, TokenUsage
from u_llm_sdk.rag_client import RAGClient, RAGClientConfig


async def main():
    """Example: Using RAGClient with MV-rag API."""

    # 1. Configure RAGClient
    config = RAGClientConfig(
        base_url="http://localhost:8000",  # MV-rag API endpoint
        enabled=True,
        timeout_seconds=0.5,  # 500ms timeout for pre-action
        cache_ttl_seconds=300,  # 5 min cache
        fail_open=True,  # Continue on errors
    )

    # 2. Create client (use async context manager)
    async with RAGClient(config) as client:

        # 3. Pre-action: Get context injection before LLM call
        print("📥 Calling pre-action...")
        pre_context = await client.on_pre_action(
            prompt="How do I implement async/await in Python?",
            provider="claude",
            model="opus-4",
            session_id="session-123",
            run_id="run-456",
        )

        if pre_context:
            print(f"✅ Context received:")
            print(f"   - Injection ID: {pre_context.injection_id}")
            print(f"   - Confidence: {pre_context.confidence:.2f}")
            print(f"   - Token count: {pre_context.token_count}")
            print(f"   - Position: {pre_context.injection_position}")
            print(f"   - Pattern: {pre_context.pattern_summary}")
            print(f"\n   Context text:\n   {pre_context.context_text[:100]}...")
        else:
            print("ℹ️  No context injection (cache miss or RAG unavailable)")

        # 4. Simulate LLM execution (normally you'd call your LLM here)
        print("\n🤖 Executing LLM call...")

        # Create mock result for demonstration
        result = LLMResult(
            success=True,
            result_type=ResultType.TEXT,
            provider="claude",
            model="opus-4",
            text="Here's how to use async/await in Python...",
            summary="Explained async/await with examples",
            token_usage=TokenUsage(
                input_tokens=150,
                output_tokens=300,
                total_tokens=450,
            ),
            duration_ms=1200,
        )

        print(f"✅ LLM result: {result.summary}")

        # 5. Post-action: Send feedback (fire-and-forget)
        print("\n📤 Sending post-action feedback...")
        await client.on_post_action(
            result=result,
            pre_action_context=pre_context,
            run_id="run-456",
        )
        print("✅ Feedback sent")


async def example_with_cache():
    """Example: Demonstrating cache behavior."""

    config = RAGClientConfig(
        base_url="http://localhost:8000",
        cache_ttl_seconds=60,  # 1 min cache for demo
    )

    async with RAGClient(config) as client:
        prompt = "What is machine learning?"

        # First call - cache miss
        print("🔍 First call (cache miss)...")
        context1 = await client.on_pre_action(
            prompt=prompt,
            provider="claude",
            model="opus-4",
        )

        # Second call - cache hit
        print("🔍 Second call (cache hit)...")
        context2 = await client.on_pre_action(
            prompt=prompt,
            provider="claude",
            model="opus-4",
        )

        if context1 and context2:
            print(f"✅ Same injection ID: {context1.injection_id == context2.injection_id}")


async def example_fail_open():
    """Example: Fail-open behavior when RAG is unavailable."""

    config = RAGClientConfig(
        base_url="http://invalid-server:9999",  # Invalid server
        timeout_seconds=0.1,
        fail_open=True,  # Fail-open enabled
    )

    async with RAGClient(config) as client:
        print("🔍 Calling pre-action with invalid server...")

        # This will fail, but return None instead of raising
        context = await client.on_pre_action(
            prompt="Test prompt",
            provider="claude",
        )

        if context is None:
            print("✅ Fail-open: Returned None, LLM execution can continue")
        else:
            print("❌ Unexpected: Context received")


if __name__ == "__main__":
    print("=" * 60)
    print("RAGClient Example 1: Basic Usage")
    print("=" * 60)
    asyncio.run(main())

    print("\n" + "=" * 60)
    print("RAGClient Example 2: Cache Behavior")
    print("=" * 60)
    asyncio.run(example_with_cache())

    print("\n" + "=" * 60)
    print("RAGClient Example 3: Fail-Open Behavior")
    print("=" * 60)
    asyncio.run(example_fail_open())
