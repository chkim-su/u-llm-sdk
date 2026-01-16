"""Quick utility functions examples for u-llm-sdk."""
import asyncio
from u_llm_sdk.core.utils import (
    quick_run, quick_text, parallel_run, template_run,
    quick_run_sync, quick_text_sync, parallel_run_sync,
)
from u_llm_sdk.types import Provider
from u_llm_sdk.session import SessionTemplate


async def quick_run_example():
    """One-shot execution with full result."""
    result = await quick_run("What are the SOLID principles?", provider=Provider.CLAUDE)
    print(f"Success: {result.success}")
    print(f"Type: {result.result_type}")
    print(f"Text: {result.text[:200]}...")


async def quick_text_example():
    """Get just the text response."""
    answer = await quick_text("What is 2 + 2?")
    print(f"Answer: {answer}")


async def parallel_execution_example():
    """Execute multiple prompts in parallel."""
    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
    ]
    
    results = await parallel_run(questions, provider=Provider.CLAUDE)
    
    for question, result in zip(questions, results):
        print(f"\nQ: {question}")
        print(f"A: {result.text[:100]}...")


async def template_execution_example():
    """Execute with specialized template persona."""
    result = await template_run(
        "Review this code:\ndef login(user, pass): return db.check(user, pass)",
        SessionTemplate.SECURITY_ANALYST,
        provider=Provider.CLAUDE,
    )
    print(f"Security Analysis:\n{result.text}")


def sync_examples():
    """Sync versions of quick utilities."""
    # Quick run sync
    result = quick_run_sync("Hello!", provider=Provider.GEMINI)
    print(f"Sync result: {result.text[:50]}...")
    
    # Quick text sync
    answer = quick_text_sync("What is 1 + 1?", provider=Provider.CLAUDE)
    print(f"Sync answer: {answer}")


if __name__ == "__main__":
    print("=== Quick Run Example ===")
    asyncio.run(quick_run_example())
    
    print("\n=== Quick Text Example ===")
    asyncio.run(quick_text_example())
    
    print("\n=== Parallel Execution Example ===")
    asyncio.run(parallel_execution_example())
    
    print("\n=== Template Execution Example ===")
    asyncio.run(template_execution_example())
    
    print("\n=== Sync Examples ===")
    sync_examples()
