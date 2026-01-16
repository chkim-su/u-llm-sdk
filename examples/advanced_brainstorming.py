#!/usr/bin/env python
"""Advanced Brainstorming Example.

This example demonstrates advanced multi-provider brainstorming patterns:
1. Full brainstorming sessions with discussion preservation
2. Custom brainstorm configuration
3. Handling consensus and escalation
4. RAG integration for pattern learning

Requirements:
    - API keys configured for providers
    - u-llm-sdk installed
    - (Optional) MV-rag server for RAG features

Usage:
    python advanced_brainstorming.py
"""

import asyncio
import logging
from typing import Dict, List, Optional

from u_llm_sdk.types import (
    BrainstormConfig,
    ConsensusResult,
    Provider,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Mock Providers for Demonstration
# =============================================================================


class MockProvider:
    """Mock provider that simulates different perspectives."""

    def __init__(self, provider: Provider, perspective: str):
        self.provider = provider
        self.perspective = perspective

    async def run(self, prompt: str) -> str:
        """Simulate provider response based on perspective."""
        await asyncio.sleep(0.1)

        # Simulate different provider perspectives
        if "vote" in prompt.lower():
            return self._generate_vote(prompt)
        elif "discuss" in prompt.lower():
            return self._generate_discussion(prompt)
        else:
            return self._generate_perspective(prompt)

    def _generate_vote(self, prompt: str) -> str:
        """Generate a vote based on provider perspective."""
        votes = {
            "strategic": "microservices",
            "practical": "monolith",
            "analytical": "hybrid",
        }
        return f"Vote: {votes.get(self.perspective, 'abstain')}"

    def _generate_discussion(self, prompt: str) -> str:
        """Generate discussion contribution."""
        contributions = {
            "strategic": (
                "From a strategic standpoint, we should consider "
                "long-term scalability and team structure."
            ),
            "practical": (
                "Looking at implementation effort, we need to balance "
                "complexity with delivery timeline."
            ),
            "analytical": (
                "Based on the requirements analysis, there are "
                "trade-offs in both approaches we should examine."
            ),
        }
        return contributions.get(self.perspective, "I agree with the discussion.")

    def _generate_perspective(self, prompt: str) -> str:
        """Generate general perspective."""
        return f"[{self.provider.value} - {self.perspective}] {prompt[:50]}..."


# =============================================================================
# Example 1: Basic Brainstorming Session
# =============================================================================


async def example_basic_brainstorm():
    """Demonstrate basic brainstorming session.

    Shows how to:
    - Configure brainstorming parameters
    - Run a session
    - Access full discussion log
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Brainstorming Session")
    print("=" * 60)

    from u_llm_sdk.multi_llm import BrainstormModule

    providers = {
        Provider.GEMINI: MockProvider(Provider.GEMINI, "strategic"),
        Provider.CLAUDE: MockProvider(Provider.CLAUDE, "practical"),
        Provider.CODEX: MockProvider(Provider.CODEX, "analytical"),
    }

    # Configure brainstorming
    config = BrainstormConfig(
        max_rounds=3,
        consensus_threshold=0.67,  # 2/3 majority
        low_agreement_threshold=0.4,  # Below this: ask user
        preserve_full_discussion=True,  # Never summarize!
    )

    module = BrainstormModule(providers, config)

    # Run brainstorming session
    topic = "Should we use microservices or monolith architecture for the new platform?"

    print(f"\nTopic: {topic}")
    print("\nRunning brainstorming session...")

    result = await module.run_session(topic)

    # Display results
    print(f"\n{'='*40}")
    print("BRAINSTORMING RESULTS")
    print(f"{'='*40}")

    print(f"\nTotal rounds: {len(result.rounds)}")
    print(f"Total time: {result.total_time_ms:.0f}ms")

    # Show discussion entries (preserved, not summarized!)
    print(f"\nDiscussion log ({len(result.all_discussion_entries)} entries):")
    for entry in result.all_discussion_entries:
        print(f"\n  [{entry.round}] {entry.speaker.value}:")
        print(f"      Type: {entry.message_type}")
        print(f"      Content: {entry.content[:80]}...")

    # Show consensus
    print(f"\n{'='*40}")
    print("CONSENSUS")
    print(f"{'='*40}")

    if result.consensus.success:
        print(f"\nConsensus REACHED!")
        print(f"Decision: {result.consensus.final_decision}")
        print(f"Agreement: {result.consensus.agreement_score:.0%}")
        print(f"Rounds taken: {result.consensus.rounds_taken}")
    else:
        print(f"\nConsensus NOT reached")
        print("Questions for user:")
        for q in result.consensus.user_questions:
            print(f"  - {q}")


# =============================================================================
# Example 2: Custom Configuration
# =============================================================================


async def example_custom_config():
    """Demonstrate custom brainstorm configuration.

    Shows how to:
    - Adjust thresholds
    - Handle low agreement scenarios
    - Configure round behavior
    """
    print("\n" + "=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)

    from u_llm_sdk.multi_llm import BrainstormModule

    providers = {
        Provider.GEMINI: MockProvider(Provider.GEMINI, "strategic"),
        Provider.CLAUDE: MockProvider(Provider.CLAUDE, "practical"),
        Provider.CODEX: MockProvider(Provider.CODEX, "analytical"),
    }

    # Configuration scenarios
    configs = [
        (
            "Strict consensus (75%)",
            BrainstormConfig(
                max_rounds=3,
                consensus_threshold=0.75,
                low_agreement_threshold=0.5,
                preserve_full_discussion=True,
            ),
        ),
        (
            "Relaxed consensus (50%)",
            BrainstormConfig(
                max_rounds=3,
                consensus_threshold=0.50,
                low_agreement_threshold=0.3,
                preserve_full_discussion=True,
            ),
        ),
        (
            "Extended discussion (5 rounds)",
            BrainstormConfig(
                max_rounds=5,
                consensus_threshold=0.67,
                low_agreement_threshold=0.4,
                preserve_full_discussion=True,
            ),
        ),
    ]

    topic = "Which cloud provider should we use for deployment?"

    for name, config in configs:
        print(f"\n--- {name} ---")
        module = BrainstormModule(providers, config)
        result = await module.run_session(topic)

        print(f"Consensus: {result.consensus.success}")
        print(f"Agreement: {result.consensus.agreement_score:.0%}")
        print(f"Rounds: {result.consensus.rounds_taken}/{config.max_rounds}")


# =============================================================================
# Example 3: Escalation Handling
# =============================================================================


async def example_escalation_handling():
    """Demonstrate escalation when consensus fails.

    Shows how to:
    - Detect low agreement
    - Generate user questions
    - Handle user input
    """
    print("\n" + "=" * 60)
    print("Example 3: Escalation Handling")
    print("=" * 60)

    from u_llm_sdk.multi_llm import BrainstormModule, ConsensusLoop

    providers = {
        Provider.GEMINI: MockProvider(Provider.GEMINI, "strategic"),
        Provider.CLAUDE: MockProvider(Provider.CLAUDE, "practical"),
        Provider.CODEX: MockProvider(Provider.CODEX, "analytical"),
    }

    # Configure with strict threshold to force escalation
    config = BrainstormConfig(
        max_rounds=2,
        consensus_threshold=0.90,  # Very strict
        low_agreement_threshold=0.6,
        preserve_full_discussion=True,
    )

    # Use ConsensusLoop directly
    loop = ConsensusLoop(providers, config)
    topic = "Should we rewrite the legacy system or maintain it?"

    print(f"\nTopic: {topic}")
    print("(Using strict 90% threshold to demonstrate escalation)")

    result = await loop.run(topic)

    if result.escalated_to_user:
        print(f"\n{'='*40}")
        print("ESCALATION TO USER")
        print(f"{'='*40}")

        print(f"\nAgreement was only {result.agreement_score:.0%}")
        print(f"Threshold was {config.consensus_threshold:.0%}")
        print("\nQuestions for the user:")
        for i, question in enumerate(result.user_questions, 1):
            print(f"  {i}. {question}")

        # Simulate user input
        print("\n--- Simulating user response ---")
        user_answer = "Prefer incremental migration approach"

        # Re-run with user context
        print(f"\nUser provided: '{user_answer}'")
        print("Would re-run brainstorm with additional context...")

    else:
        print(f"\nConsensus reached: {result.final_decision}")


# =============================================================================
# Example 4: RAG-Enhanced Brainstorming
# =============================================================================


async def example_rag_enhanced():
    """Demonstrate RAG integration for pattern learning.

    Shows how to:
    - Save brainstorm outcomes
    - Get hints from past sessions
    - Learn from decisions
    """
    print("\n" + "=" * 60)
    print("Example 4: RAG-Enhanced Brainstorming")
    print("=" * 60)

    from u_llm_sdk.multi_llm import (
        BrainstormModule,
        MultiLLMRAGHook,
        NoOpMultiLLMRAGHook,
    )

    providers = {
        Provider.GEMINI: MockProvider(Provider.GEMINI, "strategic"),
        Provider.CLAUDE: MockProvider(Provider.CLAUDE, "practical"),
        Provider.CODEX: MockProvider(Provider.CODEX, "analytical"),
    }

    # Use NoOp hook for demo (no actual RAG server needed)
    rag_hook = NoOpMultiLLMRAGHook()

    config = BrainstormConfig(
        max_rounds=3,
        consensus_threshold=0.67,
        preserve_full_discussion=True,
    )

    module = BrainstormModule(providers, config)
    topic = "What authentication strategy should we use?"

    print(f"\nTopic: {topic}")

    # Get hints from past sessions
    print("\n1. Checking for past patterns...")
    hints = await rag_hook.get_orchestration_hints(topic)

    if hints.similar_patterns:
        print("Found similar past decisions:")
        for pattern in hints.similar_patterns:
            print(f"  - {pattern.topic}: {pattern.decision}")
            print(f"    Confidence: {pattern.confidence:.0%}")
    else:
        print("No similar patterns found")

    # Run brainstorm
    print("\n2. Running brainstorm session...")
    result = await module.run_session(topic)

    print(f"Decision: {result.consensus.final_decision}")

    # Save outcome for learning
    print("\n3. Saving outcome for future learning...")
    await rag_hook.save_brainstorm_outcome(
        topic=topic,
        consensus_result=result.consensus,
        rounds_taken=result.consensus.rounds_taken,
        escalated_to_user=result.consensus.escalated_to_user,
        success=result.consensus.success,
    )

    print("Outcome saved successfully!")

    # In real usage with MV-rag:
    print("""
Note: In production, use MultiLLMRAGHook with MV-rag server:

    from u_llm_sdk.rag_client import RAGClientConfig

    config = RAGClientConfig(
        base_url="http://localhost:8000",
        timeout_seconds=0.5,
        fail_open=True,
    )
    rag_hook = MultiLLMRAGHook(config)
""")


# =============================================================================
# Example 5: Discussion Analysis
# =============================================================================


async def example_discussion_analysis():
    """Demonstrate analysis of preserved discussions.

    Shows how to:
    - Access full discussion log
    - Analyze provider contributions
    - Track opinion evolution
    """
    print("\n" + "=" * 60)
    print("Example 5: Discussion Analysis")
    print("=" * 60)

    from u_llm_sdk.multi_llm import BrainstormModule
    from collections import Counter

    providers = {
        Provider.GEMINI: MockProvider(Provider.GEMINI, "strategic"),
        Provider.CLAUDE: MockProvider(Provider.CLAUDE, "practical"),
        Provider.CODEX: MockProvider(Provider.CODEX, "analytical"),
    }

    config = BrainstormConfig(
        max_rounds=3,
        consensus_threshold=0.67,
        preserve_full_discussion=True,
    )

    module = BrainstormModule(providers, config)
    topic = "How should we structure our API versioning?"

    print(f"\nTopic: {topic}")
    print("Running brainstorm...")

    result = await module.run_session(topic)

    # Analyze discussion
    print(f"\n{'='*40}")
    print("DISCUSSION ANALYSIS")
    print(f"{'='*40}")

    # Count contributions per provider
    contributions_per_provider = Counter(
        entry.speaker for entry in result.all_discussion_entries
    )

    print("\nContributions per provider:")
    for provider, count in contributions_per_provider.items():
        print(f"  {provider.value}: {count} entries")

    # Analyze contribution types
    contribution_types = Counter(
        entry.message_type for entry in result.all_discussion_entries
    )

    print("\nContribution types:")
    for ctype, count in contribution_types.items():
        print(f"  {ctype}: {count}")

    # Track opinion evolution across rounds
    print("\nOpinion evolution by round:")
    entries_by_round = {}
    for entry in result.all_discussion_entries:
        round_num = entry.round
        if round_num not in entries_by_round:
            entries_by_round[round_num] = []
        entries_by_round[round_num].append(entry)

    for round_num in sorted(entries_by_round.keys()):
        print(f"\n  Round {round_num}:")
        for entry in entries_by_round[round_num]:
            print(f"    {entry.speaker.value}: {entry.content[:40]}...")

    # Key insight: Full discussion is preserved!
    print(f"""
{'='*40}
KEY INSIGHT
{'='*40}

The full discussion log is ALWAYS preserved (never summarized).
This allows:
- Audit trail of all decisions
- Post-hoc analysis of provider contributions
- Learning from discussion patterns
- Debugging consensus failures

Total entries preserved: {len(result.all_discussion_entries)}
""")


# =============================================================================
# Example 6: Orchestrator-Facilitated Brainstorm
# =============================================================================


async def example_orchestrator_brainstorm():
    """Demonstrate orchestrator-facilitated brainstorming.

    Shows how to:
    - Use master orchestrator to trigger brainstorms
    - Integrate brainstorm decisions with task routing
    """
    print("\n" + "=" * 60)
    print("Example 6: Orchestrator-Facilitated Brainstorm")
    print("=" * 60)

    from u_llm_sdk.multi_llm import OrchestratorFactory

    providers = {
        Provider.GEMINI: MockProvider(Provider.GEMINI, "strategic"),
        Provider.CLAUDE: MockProvider(Provider.CLAUDE, "practical"),
        Provider.CODEX: MockProvider(Provider.CODEX, "analytical"),
    }

    factory = OrchestratorFactory(providers)
    master = factory.create_master()

    # Process request that triggers brainstorm
    request = "Design the data layer architecture for our new microservices platform"

    print(f"\nRequest: {request}")
    print("\nProcessing through master orchestrator...")

    response = await master.process_request(request)

    if response.needs_brainstorm:
        print(f"\nOrchestrator determined brainstorming needed!")
        print(f"Topic: {response.brainstorm_topic}")

        # Facilitate brainstorm
        print("\nFacilitating brainstorm session...")
        consensus = await master.facilitate_brainstorm(response.brainstorm_topic)

        print(f"\n{'='*40}")
        print("BRAINSTORM RESULT")
        print(f"{'='*40}")

        print(f"Consensus: {consensus.success}")
        print(f"Decision: {consensus.final_decision}")
        print(f"Agreement: {consensus.agreement_score:.0%}")

        # Use decision to inform task routing
        if consensus.success:
            print("\nUsing decision to inform task routing...")
            print(f"Tasks would be routed based on: {consensus.final_decision}")

    else:
        print("\nNo brainstorming needed - proceeding with tasks")
        for task in response.tasks:
            routing = await master.route_task(task)
            print(f"Task: {task.objective} -> {routing.recommended_worker.value}")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Multi-LLM Orchestration - Advanced Brainstorming")
    print("=" * 60)

    examples = [
        ("Basic Brainstorm", example_basic_brainstorm),
        ("Custom Configuration", example_custom_config),
        ("Escalation Handling", example_escalation_handling),
        ("RAG-Enhanced", example_rag_enhanced),
        ("Discussion Analysis", example_discussion_analysis),
        ("Orchestrator Brainstorm", example_orchestrator_brainstorm),
    ]

    for name, func in examples:
        try:
            await func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            logger.exception(f"Example {name} failed")

        print("\n" + "-" * 60)

    print("\nAll examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
