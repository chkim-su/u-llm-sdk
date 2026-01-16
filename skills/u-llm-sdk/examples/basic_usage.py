"""Basic u-llm-sdk usage examples."""
import asyncio
from u_llm_sdk import LLM, LLMSync, Provider


async def async_example():
    """Async LLM call example."""
    async with LLM(provider=Provider.CLAUDE) as llm:
        result = await llm.run("Explain async/await in Python")
        print(result.text)


def sync_example():
    """Sync LLM call example."""
    llm = LLMSync(provider=Provider.CLAUDE)
    result = llm.run("What is the difference between sync and async?")
    print(result.text)


async def streaming_example():
    """Streaming response example."""
    async with LLM(provider=Provider.CLAUDE) as llm:
        print("Streaming response: ", end="")
        async for chunk in llm.stream("Tell me a short joke"):
            if chunk.get("type") == "text":
                print(chunk.get("content", ""), end="", flush=True)
        print()


if __name__ == "__main__":
    print("=== Async Example ===")
    asyncio.run(async_example())

    print("\n=== Sync Example ===")
    sync_example()

    print("\n=== Streaming Example ===")
    asyncio.run(streaming_example())
