"""Multi-Provider Example - Using Different LLM Providers.

This example demonstrates how to use multiple LLM providers with U-llm-sdk.
"""

import asyncio

from u_llm_sdk.types import ModelTier, Provider


async def example_different_providers():
    """Example: Using different providers for different tasks."""
    from u_llm_sdk.core.utils import quick_run

    print("=== Different Providers Example ===")

    tasks = [
        ("claude", Provider.CLAUDE, "What is recursion? (1 sentence)"),
        ("gemini", Provider.GEMINI, "What is iteration? (1 sentence)"),
    ]

    for name, provider, prompt in tasks:
        print(f"\n{name.upper()}:")
        try:
            result = await quick_run(prompt, provider=provider)
            print(f"  {result.text[:100]}...")
        except Exception as e:
            print(f"  Skipped: {e}")


async def example_parallel_providers():
    """Example: Running multiple providers in parallel."""
    from u_llm_sdk.core.utils import quick_run

    print("\n=== Parallel Providers Example ===")

    prompt = "What is the answer to life, the universe, and everything?"

    # Create tasks for parallel execution
    tasks = [
        quick_run(prompt, provider=Provider.CLAUDE),
        quick_run(prompt, provider=Provider.GEMINI),
    ]

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, (name, result) in enumerate(
            zip(["Claude", "Gemini"], results)
        ):
            if isinstance(result, Exception):
                print(f"{name}: Error - {result}")
            else:
                print(f"{name}: {result.text[:80]}...")
    except Exception as e:
        print(f"Parallel execution skipped: {e}")


async def example_tier_selection():
    """Example: Using different model tiers."""
    from u_llm_sdk.core.utils import quick_run

    print("\n=== Model Tier Example ===")

    prompt = "Solve: x^2 - 5x + 6 = 0"

    tiers = [
        (ModelTier.LOW, "Low tier (faster, cheaper)"),
        (ModelTier.HIGH, "High tier (more capable)"),
    ]

    for tier, description in tiers:
        print(f"\n{description}:")
        try:
            result = await quick_run(
                prompt,
                provider=Provider.CLAUDE,
                tier=tier,
            )
            print(f"  {result.text[:100]}...")
            print(f"  Model: {result.model}")
        except Exception as e:
            print(f"  Skipped: {e}")


async def example_provider_factory():
    """Example: Using provider factory pattern."""
    from u_llm_sdk.config import LLMConfig
    from u_llm_sdk.llm.providers import ClaudeProvider, GeminiProvider

    print("\n=== Provider Factory Example ===")

    providers = {
        "claude": (ClaudeProvider, Provider.CLAUDE),
        "gemini": (GeminiProvider, Provider.GEMINI),
    }

    for name, (ProviderClass, provider_enum) in providers.items():
        print(f"\n{name.upper()} Provider:")
        try:
            if ProviderClass.is_available():
                config = LLMConfig(provider=provider_enum)
                provider = ProviderClass(config)
                print(f"  Available at: {ProviderClass.get_cli_path()}")
            else:
                print("  Not available (CLI not found)")
        except Exception as e:
            print(f"  Error: {e}")


async def main():
    """Run all examples."""
    print("U-llm-sdk Multi-Provider Examples")
    print("=" * 50)

    await example_different_providers()
    await example_parallel_providers()
    await example_tier_selection()
    await example_provider_factory()


if __name__ == "__main__":
    asyncio.run(main())
