#!/usr/bin/env python3
"""Example demonstrating advanced Gemini CLI features.

This example shows how to use the new CLI features added to GeminiProvider:
- allowed_tools: Restrict which tools Gemini can use
- extensions: Enable Gemini CLI extensions
- include_directories: Add additional directories to context
- allowed_mcp_server_names: Filter MCP servers
- Session resume with 'latest'
"""

import asyncio

from u_llm_sdk.types import AutoApproval, Provider, SandboxMode
from u_llm_sdk import LLM, LLMConfig, ModelTier


async def example_allowed_tools():
    """Example: Restrict Gemini to specific tools."""
    print("\n=== Example 1: Allowed Tools ===")

    config = LLMConfig(
        provider=Provider.GEMINI,
        tier=ModelTier.LOW,  # Use fast model for demo
        auto_approval=AutoApproval.EDITS_ONLY,
        provider_options={
            "allowed_tools": ["edit", "read", "bash"],  # Only allow these tools
        },
    )

    async with LLM(config) as llm:
        result = await llm.run("What tools are available to you?")
        print(f"Result: {result.text[:200]}...")


async def example_extensions():
    """Example: Enable Gemini CLI extensions."""
    print("\n=== Example 2: Extensions ===")

    config = LLMConfig(
        provider=Provider.GEMINI,
        tier=ModelTier.LOW,
        provider_options={
            "extensions": ["code-review", "analysis"],  # Enable extensions
        },
    )

    async with LLM(config) as llm:
        result = await llm.run("Analyze this code structure")
        print(f"Result: {result.summary}")


async def example_include_directories():
    """Example: Add additional directories to context."""
    print("\n=== Example 3: Include Directories ===")

    config = LLMConfig(
        provider=Provider.GEMINI,
        tier=ModelTier.LOW,
        provider_options={
            "include_directories": [
                "/path/to/shared/libs",
                "/path/to/config",
            ],
        },
    )

    async with LLM(config) as llm:
        result = await llm.run("What directories are included?")
        print(f"Result: {result.summary}")


async def example_mcp_server_filter():
    """Example: Filter MCP servers."""
    print("\n=== Example 4: MCP Server Filter ===")

    config = LLMConfig(
        provider=Provider.GEMINI,
        tier=ModelTier.LOW,
        provider_options={
            "allowed_mcp_server_names": [
                "filesystem",
                "github",
            ],  # Only allow these MCP servers
        },
    )

    async with LLM(config) as llm:
        result = await llm.run("Which MCP servers are available?")
        print(f"Result: {result.summary}")


async def example_session_resume():
    """Example: Resume latest session."""
    print("\n=== Example 5: Session Resume ===")

    # First, create a session
    config = LLMConfig(
        provider=Provider.GEMINI,
        tier=ModelTier.LOW,
    )

    async with LLM(config) as llm:
        result = await llm.run("Remember: my favorite color is blue")
        session_id = result.session_id
        print(f"Created session: {session_id}")

    # Now resume it using 'latest'
    async with LLM(config) as llm:
        result = await llm.run("", session_id="latest")
        print(f"Resumed latest session: {result.session_id}")


async def example_combined_features():
    """Example: Combine multiple advanced features."""
    print("\n=== Example 6: Combined Features ===")

    config = LLMConfig(
        provider=Provider.GEMINI,
        model="gemini-2.5-flash-lite",
        auto_approval=AutoApproval.EDITS_ONLY,
        sandbox=SandboxMode.WORKSPACE_WRITE,
        timeout=300.0,
        provider_options={
            "temperature": 0.7,
            "top_p": 0.9,
            "allowed_tools": ["edit", "read", "bash"],
            "extensions": ["code-review"],
            "include_directories": ["/workspace/shared"],
            "allowed_mcp_server_names": ["filesystem"],
        },
    )

    print("Configuration:")
    print(f"  Model: {config.get_model()}")
    print(f"  Approval: {config.auto_approval.value}")
    print(f"  Sandbox: {config.sandbox.value}")
    print(f"  Temperature: {config.provider_options.get('temperature')}")
    print(f"  Allowed Tools: {config.provider_options.get('allowed_tools')}")
    print(f"  Extensions: {config.provider_options.get('extensions')}")

    async with LLM(config) as llm:
        result = await llm.run("Review this project structure")
        print(f"\nResult: {result.summary}")
        print(f"Files modified: {len(result.files_modified)}")
        print(f"Commands run: {len(result.commands_run)}")


async def main():
    """Run all examples."""
    print("=" * 60)
    print("Gemini Advanced Features Examples")
    print("=" * 60)

    # Run examples (comment out the ones you don't want to run)
    # await example_allowed_tools()
    # await example_extensions()
    # await example_include_directories()
    # await example_mcp_server_filter()
    # await example_session_resume()
    await example_combined_features()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
