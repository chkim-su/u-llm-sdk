"""Streaming Example - Real-time LLM Output.

This example demonstrates how to stream responses from LLM providers.
"""

import asyncio

from u_llm_sdk.types import Provider


async def example_basic_streaming():
    """Example: Basic streaming output."""
    from u_llm_sdk import LLM

    print("=== Basic Streaming Example ===")
    print("Response: ", end="", flush=True)

    try:
        async with LLM(provider=Provider.CLAUDE) as llm:
            async for chunk in llm.stream("Write a haiku about programming."):
                # Each chunk may contain different fields
                if "text" in chunk:
                    print(chunk["text"], end="", flush=True)
        print()  # Newline after streaming
    except Exception as e:
        print(f"\nSkipped: {e}")


async def example_streaming_with_metadata():
    """Example: Streaming with metadata tracking."""
    from u_llm_sdk import LLM

    print("\n=== Streaming with Metadata Example ===")

    chunk_count = 0
    total_text = ""

    try:
        async with LLM(provider=Provider.CLAUDE) as llm:
            async for chunk in llm.stream("Count from 1 to 5."):
                chunk_count += 1

                if "text" in chunk:
                    total_text += chunk["text"]
                    print(f"Chunk {chunk_count}: {chunk['text']!r}")

                # Check for completion
                if chunk.get("done"):
                    print(f"\n--- Streaming complete ---")
                    print(f"Total chunks: {chunk_count}")
                    print(f"Total text: {total_text}")
    except Exception as e:
        print(f"Skipped: {e}")


async def example_streaming_error_handling():
    """Example: Handling errors during streaming."""
    from u_llm_sdk import LLM

    print("\n=== Streaming Error Handling Example ===")

    try:
        async with LLM(provider=Provider.CLAUDE) as llm:
            buffer = []
            async for chunk in llm.stream("Explain async/await briefly."):
                if "text" in chunk:
                    buffer.append(chunk["text"])
                    # Process in real-time
                    print(".", end="", flush=True)

                if "error" in chunk:
                    print(f"\nError in stream: {chunk['error']}")
                    break

            print()
            full_response = "".join(buffer)
            print(f"Complete response ({len(full_response)} chars):")
            print(full_response[:200] + "..." if len(full_response) > 200 else full_response)
    except Exception as e:
        print(f"Skipped: {e}")


async def example_streaming_with_timeout():
    """Example: Streaming with timeout."""
    from u_llm_sdk import LLM

    print("\n=== Streaming with Timeout Example ===")

    try:
        async with LLM(provider=Provider.CLAUDE) as llm:
            # Use asyncio.timeout for Python 3.11+
            try:
                async with asyncio.timeout(30):  # 30 second timeout
                    async for chunk in llm.stream("Write a short poem."):
                        if "text" in chunk:
                            print(chunk["text"], end="", flush=True)
                    print()
            except TimeoutError:
                print("\nStreaming timed out!")
    except Exception as e:
        print(f"Skipped: {e}")


async def main():
    """Run all examples."""
    print("U-llm-sdk Streaming Examples")
    print("=" * 50)

    await example_basic_streaming()
    await example_streaming_with_metadata()
    await example_streaming_error_handling()
    await example_streaming_with_timeout()


if __name__ == "__main__":
    asyncio.run(main())
