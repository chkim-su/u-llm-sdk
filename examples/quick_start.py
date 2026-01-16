"""Quick Start Example - Basic LLM Usage.

This example demonstrates the basic usage of U-llm-sdk for LLM interactions.
"""

import asyncio

from u_llm_sdk.types import Provider


async def example_async_llm():
    """Example: Using async LLM client."""
    from u_llm_sdk import LLM

    print("=== Async LLM Example ===")

    async with LLM(provider=Provider.CLAUDE) as llm:
        result = await llm.run("Explain what a Python decorator is in 2 sentences.")

        print(f"Success: {result.success}")
        print(f"Text: {result.text}")
        if result.token_usage:
            print(f"Tokens: {result.token_usage.total_tokens}")


def example_sync_llm():
    """Example: Using sync LLM client."""
    from u_llm_sdk import LLMSync

    print("\n=== Sync LLM Example ===")

    llm = LLMSync(provider=Provider.CLAUDE)
    result = llm.run("What is the capital of France? Answer in one word.")

    print(f"Success: {result.success}")
    print(f"Text: {result.text}")


async def example_quick_utils():
    """Example: Using quick utility functions."""
    from u_llm_sdk.core.utils import quick_run, quick_text

    print("\n=== Quick Utils Example ===")

    # Get full LLMResult
    result = await quick_run(
        "What is 2 + 2?",
        provider=Provider.CLAUDE,
    )
    print(f"quick_run result: {result.text}")

    # Get just the text
    text = await quick_text(
        "Say 'Hello World'",
        provider=Provider.GEMINI,
    )
    print(f"quick_text result: {text}")


async def main():
    """Run all examples."""
    print("U-llm-sdk Quick Start Examples")
    print("=" * 50)

    # Note: These examples require actual CLI tools installed
    # (claude, codex, gemini) and will fail without them.
    # For testing, you can mock the providers.

    try:
        await example_async_llm()
    except Exception as e:
        print(f"Async example skipped: {e}")

    try:
        example_sync_llm()
    except Exception as e:
        print(f"Sync example skipped: {e}")

    try:
        await example_quick_utils()
    except Exception as e:
        print(f"Quick utils example skipped: {e}")


if __name__ == "__main__":
    asyncio.run(main())
