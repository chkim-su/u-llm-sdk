#!/usr/bin/env python
"""Consensus Workflow Example.

This example demonstrates consensus-based decision making workflows:
1. 3-round majority voting
2. Different consensus scenarios
3. Vote evolution tracking
4. Hybrid orchestration with consensus

Requirements:
    - API keys configured for providers
    - u-llm-sdk installed

Usage:
    python consensus_workflow.py
"""

import asyncio
import logging
from typing import Dict, List, Optional

from u_llm_sdk.types import (
    BrainstormConfig,
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


class MockVotingProvider:
    """Mock provider that simulates voting behavior."""

    def __init__(self, provider: Provider, initial_vote: str, flexibility: float = 0.3):
        """Initialize mock provider.

        Args:
            provider: Provider type
            initial_vote: Initial vote preference
            flexibility: Probability of changing vote (0-1)
        """
        self.provider = provider
        self.initial_vote = initial_vote
        self.current_vote = initial_vote
        self.flexibility = flexibility
        self.vote_history: List[str] = []

    async def run(self, prompt: str) -> str:
        """Simulate provider response."""
        await asyncio.sleep(0.1)

        if "vote" in prompt.lower() or "position" in prompt.lower():
            # Record vote
            self.vote_history.append(self.current_vote)

            # Simulate vote change based on flexibility
            import random

            if random.random() < self.flexibility:
                # Might change vote
                self.current_vote = self.initial_vote

            return f"My position is: {self.current_vote}"

        return f"[{self.provider.value}] Response to: {prompt[:30]}..."


# =============================================================================
# Example 1: Basic Consensus Loop
# =============================================================================


async def example_basic_consensus():
    """Demonstrate basic 3-round consensus loop.

    Shows how to:
    - Configure consensus parameters
    - Run voting rounds
    - Track vote evolution
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Consensus Loop")
    print("=" * 60)

    from u_llm_sdk.multi_llm import ConsensusLoop

    # Create providers with different initial votes
    providers = {
        Provider.GEMINI: MockVotingProvider(Provider.GEMINI, "Option A", 0.1),
        Provider.CLAUDE: MockVotingProvider(Provider.CLAUDE, "Option A", 0.2),
        Provider.CODEX: MockVotingProvider(Provider.CODEX, "Option B", 0.5),
    }

    config = BrainstormConfig(
        max_rounds=3,
        consensus_threshold=0.67,  # 2/3 majority
        low_agreement_threshold=0.4,
        preserve_full_discussion=True,
    )

    loop = ConsensusLoop(providers, config)
    topic = "Should we use PostgreSQL or MongoDB for the primary database?"

    print(f"\nTopic: {topic}")
    print(f"Threshold for consensus: {config.consensus_threshold:.0%}")
    print(f"Max rounds: {config.max_rounds}")

    print("\nInitial positions:")
    for p in [Provider.GEMINI, Provider.CLAUDE, Provider.CODEX]:
        print(f"  {p.value}: {providers[p].initial_vote}")

    print("\nRunning consensus loop...")
    result = await loop.run(topic)

    print(f"\n{'='*40}")
    print("CONSENSUS RESULT")
    print(f"{'='*40}")

    print(f"\nSuccess: {result.success}")
    print(f"Final decision: {result.final_decision}")
    print(f"Agreement score: {result.agreement_score:.0%}")
    print(f"Rounds taken: {result.rounds_taken}")

    print("\nVote breakdown:")
    for provider, vote in result.vote_breakdown.items():
        print(f"  {provider}: {vote}")


# =============================================================================
# Example 2: Different Consensus Scenarios
# =============================================================================


async def example_consensus_scenarios():
    """Demonstrate different consensus scenarios.

    Shows how to:
    - Handle unanimous agreement
    - Handle split votes
    - Handle no consensus
    """
    print("\n" + "=" * 60)
    print("Example 2: Consensus Scenarios")
    print("=" * 60)

    from u_llm_sdk.multi_llm import ConsensusLoop

    config = BrainstormConfig(
        max_rounds=3,
        consensus_threshold=0.67,
        low_agreement_threshold=0.4,
        preserve_full_discussion=True,
    )

    scenarios = [
        (
            "Unanimous Agreement",
            {
                Provider.GEMINI: MockVotingProvider(Provider.GEMINI, "A", 0.0),
                Provider.CLAUDE: MockVotingProvider(Provider.CLAUDE, "A", 0.0),
                Provider.CODEX: MockVotingProvider(Provider.CODEX, "A", 0.0),
            },
        ),
        (
            "Majority Agreement (2/3)",
            {
                Provider.GEMINI: MockVotingProvider(Provider.GEMINI, "A", 0.0),
                Provider.CLAUDE: MockVotingProvider(Provider.CLAUDE, "A", 0.0),
                Provider.CODEX: MockVotingProvider(Provider.CODEX, "B", 0.0),
            },
        ),
        (
            "Three-Way Split",
            {
                Provider.GEMINI: MockVotingProvider(Provider.GEMINI, "A", 0.0),
                Provider.CLAUDE: MockVotingProvider(Provider.CLAUDE, "B", 0.0),
                Provider.CODEX: MockVotingProvider(Provider.CODEX, "C", 0.0),
            },
        ),
    ]

    for name, providers in scenarios:
        print(f"\n--- {name} ---")

        loop = ConsensusLoop(providers, config)
        result = await loop.run("What approach should we take?")

        print(f"  Consensus: {result.success}")
        print(f"  Agreement: {result.agreement_score:.0%}")
        print(f"  Decision: {result.final_decision or 'None'}")
        if result.escalated_to_user:
            print(f"  Escalated: Yes")


# =============================================================================
# Example 3: Vote Evolution Tracking
# =============================================================================


async def example_vote_evolution():
    """Demonstrate tracking vote changes across rounds.

    Shows how to:
    - Track how positions evolve
    - Identify influencers
    - Understand convergence patterns
    """
    print("\n" + "=" * 60)
    print("Example 3: Vote Evolution Tracking")
    print("=" * 60)

    from u_llm_sdk.multi_llm import ConsensusLoop

    # Create providers with varying flexibility
    providers = {
        Provider.GEMINI: MockVotingProvider(
            Provider.GEMINI, "Option A", flexibility=0.1
        ),  # Stable
        Provider.CLAUDE: MockVotingProvider(
            Provider.CLAUDE, "Option B", flexibility=0.5
        ),  # Flexible
        Provider.CODEX: MockVotingProvider(
            Provider.CODEX, "Option B", flexibility=0.3
        ),  # Moderate
    }

    config = BrainstormConfig(
        max_rounds=3,
        consensus_threshold=0.67,
        low_agreement_threshold=0.4,
        preserve_full_discussion=True,
    )

    loop = ConsensusLoop(providers, config)
    topic = "Which testing framework should we adopt?"

    print(f"\nTopic: {topic}")
    print("\nProvider flexibility:")
    print("  GEMINI: Low (0.1) - Tends to maintain position")
    print("  CLAUDE: High (0.5) - Open to changing")
    print("  CODEX: Moderate (0.3) - Sometimes changes")

    print("\nRunning consensus loop...")
    result = await loop.run(topic)

    print(f"\n{'='*40}")
    print("VOTE EVOLUTION")
    print(f"{'='*40}")

    print("\nFinal result:")
    print(f"  Consensus: {result.success}")
    print(f"  Decision: {result.final_decision}")
    print(f"  Agreement: {result.agreement_score:.0%}")

    print("\nVote history per provider:")
    for provider, mock in providers.items():
        print(f"\n  {provider.value}:")
        print(f"    Initial: {mock.initial_vote}")
        print(f"    Final: {mock.current_vote}")
        print(f"    History: {mock.vote_history}")

    # Analyze convergence
    print("\nConvergence Analysis:")
    final_votes = [mock.current_vote for mock in providers.values()]
    from collections import Counter

    vote_counts = Counter(final_votes)
    most_common = vote_counts.most_common(1)[0]
    print(f"  Most common position: {most_common[0]} ({most_common[1]}/3)")


# =============================================================================
# Example 4: Consensus with Escalation
# =============================================================================


async def example_consensus_escalation():
    """Demonstrate escalation when consensus fails.

    Shows how to:
    - Detect when escalation is needed
    - Generate questions for user
    - Resume after user input
    """
    print("\n" + "=" * 60)
    print("Example 4: Consensus with Escalation")
    print("=" * 60)

    from u_llm_sdk.multi_llm import ConsensusLoop

    # Create providers that won't reach consensus
    providers = {
        Provider.GEMINI: MockVotingProvider(Provider.GEMINI, "REST API", 0.0),
        Provider.CLAUDE: MockVotingProvider(Provider.CLAUDE, "GraphQL", 0.0),
        Provider.CODEX: MockVotingProvider(Provider.CODEX, "gRPC", 0.0),
    }

    config = BrainstormConfig(
        max_rounds=3,
        consensus_threshold=0.67,
        low_agreement_threshold=0.4,
        preserve_full_discussion=True,
    )

    loop = ConsensusLoop(providers, config)
    topic = "What API style should we use for the new service?"

    print(f"\nTopic: {topic}")
    print("(Providers have incompatible positions)")

    result = await loop.run(topic)

    if result.escalated_to_user:
        print(f"\n{'='*40}")
        print("ESCALATION REQUIRED")
        print(f"{'='*40}")

        print(f"\nNo consensus after {result.rounds_taken} rounds")
        print(f"Agreement: {result.agreement_score:.0%}")
        print(f"Threshold needed: {config.consensus_threshold:.0%}")

        print("\nPositions held:")
        for provider, vote in result.vote_breakdown.items():
            print(f"  {provider}: {vote}")

        print("\nQuestions for user:")
        for i, question in enumerate(result.user_questions, 1):
            print(f"  {i}. {question}")

        # Simulate user response
        print("\n--- Simulating user intervention ---")
        user_decision = "Use REST API for public, gRPC for internal services"
        print(f"User decided: {user_decision}")

        # Would normally update providers and re-run
        print("\nNext steps:")
        print("1. Update context with user decision")
        print("2. Re-run consensus for implementation details")
        print("3. Route tasks based on combined decision")


# =============================================================================
# Example 5: Hybrid Orchestration with Consensus
# =============================================================================


async def example_hybrid_consensus():
    """Demonstrate hybrid orchestration using consensus decisions.

    Shows how to:
    - Use consensus for architectural decisions
    - Feed decisions into task routing
    - Handle the full workflow
    """
    print("\n" + "=" * 60)
    print("Example 5: Hybrid Orchestration with Consensus")
    print("=" * 60)

    from u_llm_sdk.multi_llm import (
        OrchestratorFactory,
        HybridOrchestrator,
        ExecutionMode,
        MigrationHelper,
    )

    providers = {
        Provider.GEMINI: MockVotingProvider(Provider.GEMINI, "microservices", 0.2),
        Provider.CLAUDE: MockVotingProvider(Provider.CLAUDE, "microservices", 0.2),
        Provider.CODEX: MockVotingProvider(Provider.CODEX, "modular-monolith", 0.3),
    }

    factory = OrchestratorFactory(providers)
    master = factory.create_master()

    request = "Design and implement user authentication for our platform"

    print(f"\nRequest: {request}")

    # Step 1: Check if complex enough for multi-LLM
    helper = MigrationHelper()
    use_multi = helper.should_use_multi_llm(request)
    print(f"\nNeed multi-LLM? {use_multi}")

    # Step 2: Process request
    print("\nProcessing request...")
    response = await master.process_request(request)

    if response.needs_brainstorm:
        print(f"\nBrainstorming needed!")
        print(f"Topic: {response.brainstorm_topic}")

        # Step 3: Run consensus
        print("\nRunning consensus loop...")
        consensus = await master.facilitate_brainstorm(response.brainstorm_topic)

        print(f"\n{'='*40}")
        print("ARCHITECTURAL DECISION")
        print(f"{'='*40}")
        print(f"Decision: {consensus.final_decision}")
        print(f"Agreement: {consensus.agreement_score:.0%}")

        # Step 4: Use decision to guide implementation
        if consensus.success:
            print("\nUsing consensus to guide implementation...")
            print(f"Architecture: {consensus.final_decision}")

            # Create tasks based on decision
            print("\nGenerated tasks:")
            for i, task in enumerate(response.tasks, 1):
                routing = await master.route_task(task)
                print(f"  {i}. {task.objective}")
                print(f"     -> {routing.recommended_worker.value}")
                print(f"     Reason: {routing.routing_reason}")

    else:
        print("\nDirect task routing (no brainstorm needed)")
        for task in response.tasks:
            routing = await master.route_task(task)
            print(f"  {task.objective} -> {routing.recommended_worker.value}")


# =============================================================================
# Example 6: Custom Consensus Strategy
# =============================================================================


async def example_custom_strategy():
    """Demonstrate custom consensus strategies.

    Shows how to:
    - Implement weighted voting
    - Add domain expertise factors
    - Customize decision logic
    """
    print("\n" + "=" * 60)
    print("Example 6: Custom Consensus Strategy")
    print("=" * 60)

    from u_llm_sdk.multi_llm import ConsensusLoop
    from collections import Counter

    # Define weights based on task type
    expertise_weights = {
        "code": {Provider.CLAUDE: 1.5, Provider.GEMINI: 1.0, Provider.CODEX: 1.2},
        "architecture": {Provider.GEMINI: 1.5, Provider.CLAUDE: 1.2, Provider.CODEX: 1.0},
        "analysis": {Provider.CODEX: 1.5, Provider.GEMINI: 1.0, Provider.CLAUDE: 1.0},
    }

    def weighted_consensus(
        votes: Dict[Provider, str], task_type: str
    ) -> tuple[str, float]:
        """Calculate weighted consensus."""
        weights = expertise_weights.get(task_type, {})

        weighted_votes = Counter()
        total_weight = 0

        for provider, vote in votes.items():
            weight = weights.get(provider, 1.0)
            weighted_votes[vote] += weight
            total_weight += weight

        if not weighted_votes:
            return None, 0.0

        winner, winner_weight = weighted_votes.most_common(1)[0]
        agreement = winner_weight / total_weight

        return winner, agreement

    # Test different task types
    providers = {
        Provider.GEMINI: MockVotingProvider(Provider.GEMINI, "Option A", 0.0),
        Provider.CLAUDE: MockVotingProvider(Provider.CLAUDE, "Option B", 0.0),
        Provider.CODEX: MockVotingProvider(Provider.CODEX, "Option A", 0.0),
    }

    votes = {p: providers[p].initial_vote for p in providers}

    print("Votes: ", {p.value: v for p, v in votes.items()})

    for task_type in ["code", "architecture", "analysis"]:
        decision, agreement = weighted_consensus(votes, task_type)
        print(f"\nTask type: {task_type}")
        print(f"  Weights: {expertise_weights[task_type]}")
        print(f"  Decision: {decision}")
        print(f"  Agreement: {agreement:.0%}")

    print("""
Note: This shows how expertise-based weighting can influence decisions.
In production, you would extend ConsensusLoop with custom voting logic.
""")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Multi-LLM Orchestration - Consensus Workflows")
    print("=" * 60)

    examples = [
        ("Basic Consensus", example_basic_consensus),
        ("Consensus Scenarios", example_consensus_scenarios),
        ("Vote Evolution", example_vote_evolution),
        ("Consensus Escalation", example_consensus_escalation),
        ("Hybrid Consensus", example_hybrid_consensus),
        ("Custom Strategy", example_custom_strategy),
    ]

    for name, func in examples:
        try:
            await func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            logger.exception(f"Example {name} failed")

        print("\n" + "-" * 60)

    print("\nAll examples completed!")
    print("""
Summary of Consensus Patterns:
─────────────────────────────
1. 3-round voting with 2/3 majority threshold
2. Escalate to user when agreement < 40%
3. Track vote evolution across rounds
4. Use consensus decisions to guide implementation
5. Custom weighting based on expertise
""")


if __name__ == "__main__":
    asyncio.run(main())
