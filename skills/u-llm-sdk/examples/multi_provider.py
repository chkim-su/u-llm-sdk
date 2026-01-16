"""Multi-provider comparison example."""
import asyncio
from u_llm_sdk import LLM, Provider


async def compare_providers():
    """Compare responses from different providers."""
    prompt = "What are the three laws of robotics?"

    providers = [Provider.CLAUDE, Provider.CODEX, Provider.GEMINI]

    for provider in providers:
        print(f"\n=== {provider.value.upper()} ===")
        try:
            async with LLM(provider=provider) as llm:
                result = await llm.run(prompt)
                print(result.text[:200] + "..." if len(result.text) > 200 else result.text)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(compare_providers())
