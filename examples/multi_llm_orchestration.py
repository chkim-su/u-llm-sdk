#!/usr/bin/env python
"""Multi-LLM Orchestration Examples.

This module demonstrates the Multi-LLM orchestration system that coordinates
multiple LLM providers (Gemini, Claude, Codex) for complex tasks.

Examples:
1. Basic orchestration with factory
2. Brainstorming with consensus
3. ClarityGate for task validation
4. Hybrid orchestrator for code tasks
5. RAG integration for pattern learning
6. State recovery and graceful degradation

Requirements:
    - Configure API keys for all providers
    - MV-rag server for RAG features (optional)

Usage:
    # Run all examples
    python multi_llm_orchestration.py

    # Run specific example
    python multi_llm_orchestration.py basic
    python multi_llm_orchestration.py brainstorm
    python multi_llm_orchestration.py hybrid
"""

import asyncio
import sys
from pathlib import Path

from u_llm_sdk.types import (
    BrainstormConfig,
    ClarityLevel,
    Provider,
    Task,
)


# =============================================================================
# Example 1: Basic Orchestration with Factory
# =============================================================================


async def basic_orchestration_example():
    """Demonstrate basic multi-LLM orchestration.

    This example shows how to:
    - Create orchestrators using OrchestratorFactory
    - Process requests with the master orchestrator
    - Route tasks to appropriate workers
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Orchestration with Factory")
    print("=" * 60)

    from u_llm_sdk import LLM
    from u_llm_sdk.multi_llm import (
        GeminiOrchestrator,
        OrchestratorFactory,
    )

    # In real usage, you'd create actual providers
    # Here we show the pattern
    print("""
# Create providers
async with LLM(provider=Provider.GEMINI) as gemini, \\
           LLM(provider=Provider.CLAUDE) as claude, \\
           LLM(provider=Provider.CODEX) as codex:

    providers = {
        Provider.GEMINI: gemini,
        Provider.CLAUDE: claude,
        Provider.CODEX: codex,
    }

    # Create factory
    factory = OrchestratorFactory(providers)

    # Create master orchestrator (Gemini by default)
    master = factory.create_master()

    # Process user request
    response = await master.process_request(
        "Design and implement user authentication"
    )

    if response.needs_brainstorm:
        print(f"Complex task: need to brainstorm {response.brainstorm_topic}")
        consensus = await master.facilitate_brainstorm(response.brainstorm_topic)
        print(f"Consensus reached: {consensus.final_decision}")

    elif response.needs_clarification:
        print("Questions for user:")
        for q in response.clarification_questions:
            print(f"  - {q}")

    else:
        # Route tasks to workers
        for task in response.tasks:
            routing = await master.route_task(task)
            print(f"Task: {task.objective}")
            print(f"  -> Assigned to: {routing.recommended_worker.value}")
            print(f"  -> Reason: {routing.routing_reason}")
""")

    print("\nPattern: Gemini (master) coordinates, delegates to specialists.")


# =============================================================================
# Example 2: Brainstorming with Consensus
# =============================================================================


async def brainstorming_example():
    """Demonstrate multi-provider brainstorming.

    This example shows how to:
    - Configure brainstorming sessions
    - Run 3-round consensus building
    - Handle low agreement scenarios
    """
    print("\n" + "=" * 60)
    print("Example 2: Brainstorming with Consensus")
    print("=" * 60)

    print("""
from u_llm_sdk.multi_llm import (
    BrainstormModule,
    ConsensusLoop,
)
from u_llm_sdk.types import BrainstormConfig

# Configure brainstorming
config = BrainstormConfig(
    max_rounds=3,                    # Maximum discussion rounds
    consensus_threshold=0.67,        # 2/3 majority needed
    low_agreement_threshold=0.4,     # Below this: ask user
    preserve_full_discussion=True,   # Keep complete log
)

# Create brainstorm module
module = BrainstormModule(providers, config)

# Run session on architecture decision
result = await module.run_session(
    context="Should we use microservices or monolith for the new platform?"
)

# Analyze result
print(f"Consensus reached: {result.success}")
print(f"Final decision: {result.final_decision}")
print(f"Vote breakdown: {result.vote_breakdown}")

if result.escalated_to_user:
    print("Low agreement - asking user:")
    for q in result.user_questions:
        print(f"  - {q}")

# Full discussion preserved (never summarized)
print(f"\\nDiscussion log ({len(result.full_discussion_log)} entries):")
for entry in result.full_discussion_log:
    print(f"  [{entry.speaker.value}] {entry.message_type}: {entry.content[:50]}...")
""")

    print("\nKey principle: 3-round voting with full log preservation.")


# =============================================================================
# Example 3: ClarityGate for Task Validation
# =============================================================================


async def clarity_gate_example():
    """Demonstrate ClarityGate for task validation.

    This example shows how to:
    - Assess task clarity before execution
    - Handle unclear tasks with escalation
    """
    print("\n" + "=" * 60)
    print("Example 3: ClarityGate for Task Validation")
    print("=" * 60)

    print("""
from u_llm_sdk.multi_llm import ClarityGate, EscalationProtocol
from u_llm_sdk.types import Task, ClarityLevel

# Create ClarityGate with Claude (code worker)
gate = ClarityGate(claude_provider)

# Create task
task = Task(
    task_id="auth-001",
    objective="Add authentication to the API",
    context="FastAPI backend with PostgreSQL",
    constraints=["Use JWT", "Support OAuth2"],
)

# Assess clarity BEFORE execution
assessment = await gate.assess(task)

print(f"Clarity level: {assessment.level.value}")
print(f"Clarity score: {assessment.score:.2f}")
print(f"Recommendation: {assessment.recommendation}")

if assessment.level == ClarityLevel.CLEAR:
    print("Task is clear - proceeding with execution")
    # Execute task...

elif assessment.level == ClarityLevel.NEEDS_CLARIFICATION:
    print("Some aspects need clarification:")
    for aspect in assessment.unclear_aspects:
        print(f"  - {aspect.aspect_type}: {aspect.description}")
        print(f"    Need: {aspect.clarification_needed}")

elif assessment.level == ClarityLevel.AMBIGUOUS:
    print("Task is ambiguous - escalating to orchestrator")

    # Create escalation request
    from u_llm_sdk.types import EscalationRequest

    request = EscalationRequest(
        source_worker=Provider.CLAUDE,
        original_task=task,
        clarity_assessment=assessment,
        specific_questions=assessment.self_questions,
    )

    # Escalate to master orchestrator
    protocol = EscalationProtocol(gemini_orchestrator)
    response = await protocol.escalate(request)

    if response.refined_task:
        print("Received refined task - retrying")
        # Re-assess refined task...
""")

    print("\nPattern: Worker self-assesses BEFORE execution, escalates if unclear.")


# =============================================================================
# Example 4: Hybrid Orchestrator for Code Tasks
# =============================================================================


async def hybrid_orchestrator_example():
    """Demonstrate HybridOrchestrator combining multi-LLM with MergeExecutor.

    This example shows how to:
    - Use hybrid mode for code generation tasks
    - Combine brainstorming decisions with code execution
    - Handle state recovery
    """
    print("\n" + "=" * 60)
    print("Example 4: Hybrid Orchestrator for Code Tasks")
    print("=" * 60)

    print("""
from u_llm_sdk.multi_llm.migration import (
    HybridOrchestrator,
    ExecutionMode,
    MigrationHelper,
)
from u_llm_sdk.llm.orchestration import MergeExecutorConfig

# Configure MergeExecutor for code tasks
merge_config = MergeExecutorConfig(
    integration_branch="llm/auth-feature",
    create_pr=True,
    require_tests=True,
    require_typecheck=True,
)

# Create hybrid orchestrator
orchestrator = HybridOrchestrator(
    providers=providers,
    merge_config=merge_config,
    brainstorm_threshold=0.6,  # Trigger brainstorm if ambiguity > 60%
)

# Run hybrid orchestration
result = await orchestrator.run(
    request="Implement OAuth2 authentication with Google provider",
    cwd="/path/to/project",
    session_id="auth-session-001",  # For recovery
)

print(f"Success: {result.success}")
print(f"Mode: {result.execution_mode.value}")
print(f"Time: {result.execution_time_ms}ms")

if result.brainstorm_result:
    print(f"Brainstorm decision: {result.brainstorm_result.final_decision}")

if result.merge_result:
    print(f"PR created: {result.merge_result.pr_url}")

# Migration helper for gradual adoption
helper = MigrationHelper()

# Check if request benefits from multi-LLM
request = "Design the authentication system architecture"
if helper.should_use_multi_llm(request):
    print("Using multi-LLM orchestration")
    result = await orchestrator.run(request, cwd="/project")
else:
    print("Using single-provider execution")
    # Direct MergeExecutor call...
""")

    print("\nPattern: Gemini plans, Claude executes via MergeExecutor.")


# =============================================================================
# Example 5: RAG Integration for Pattern Learning
# =============================================================================


async def rag_integration_example():
    """Demonstrate RAG integration for pattern learning.

    This example shows how to:
    - Save routing decisions for learning
    - Search similar patterns for hints
    - Use hints to improve routing
    """
    print("\n" + "=" * 60)
    print("Example 5: RAG Integration for Pattern Learning")
    print("=" * 60)

    print("""
from u_llm_sdk.multi_llm.rag_integration import (
    MultiLLMRAGHook,
    TaskType,
    NoOpMultiLLMRAGHook,  # For testing without MV-rag
)
from u_llm_sdk.rag_client import RAGClientConfig

# Configure RAG hook
rag_config = RAGClientConfig(
    base_url="http://localhost:8000",  # MV-rag server
    timeout_seconds=0.5,
    fail_open=True,  # Continue even if RAG fails
)

rag_hook = MultiLLMRAGHook(rag_config)

# Save routing decision after task assignment
await rag_hook.save_routing_decision(
    request="Implement user authentication",
    task_type=TaskType.CODE_IMPLEMENTATION,
    assigned_provider=Provider.CLAUDE,
    routing_reason="Code implementation task -> Claude specialist",
    confidence=0.95,
    context_factors=["FastAPI", "JWT", "PostgreSQL"],
)

# Save brainstorm outcome
from u_llm_sdk.types import ConsensusResult

consensus = ConsensusResult(
    success=True,
    final_decision="Use microservices with event-driven communication",
    vote_breakdown={
        "gemini": "microservices",
        "claude": "microservices",
        "codex": "hybrid",
    },
    discussion_summary="Discussed scalability vs complexity tradeoffs",
)

await rag_hook.save_brainstorm_outcome(
    topic="Architecture decision: microservices vs monolith",
    consensus_result=consensus,
    rounds_taken=2,
    escalated_to_user=False,
    success=True,
)

# Get hints for new request based on patterns
hints = await rag_hook.get_orchestration_hints(
    request="Build OAuth2 login flow"
)

if hints.suggested_provider:
    print(f"Suggested provider: {hints.suggested_provider.value}")
    print(f"Confidence: {hints.confidence:.2f}")

if hints.warnings:
    print("Warnings from similar past requests:")
    for w in hints.warnings:
        print(f"  ⚠️ {w}")

# Record task outcome for feedback
await rag_hook.record_task_outcome(
    request="Implement user authentication",
    task_type=TaskType.CODE_IMPLEMENTATION,
    assigned_provider=Provider.CLAUDE,
    success=True,
    execution_time_ms=15000,
)
""")

    print("\nPattern: Learn from past routing decisions to improve future ones.")


# =============================================================================
# Example 6: State Recovery and Graceful Degradation
# =============================================================================


async def recovery_example():
    """Demonstrate state recovery and graceful degradation.

    This example shows how to:
    - Persist state for recovery
    - Resume interrupted sessions
    - Handle provider failures gracefully
    """
    print("\n" + "=" * 60)
    print("Example 6: State Recovery and Graceful Degradation")
    print("=" * 60)

    print("""
from u_llm_sdk.multi_llm.migration import (
    StateRecoveryManager,
    HybridOrchestrator,
    ExecutionMode,
)
from pathlib import Path

# Setup state recovery
state_dir = Path("~/.cache/u-llm-sdk/state").expanduser()
recovery = StateRecoveryManager(state_dir)

# Save checkpoint during execution
await recovery.save_state(
    session_id="long-task-001",
    state={
        "in_progress": True,
        "current_phase": "brainstorming",
        "partial_results": {
            "clarity_check": "passed",
            "brainstorm_round": 2,
        },
    },
)

# Later: resume from checkpoint
saved_state = await recovery.load_state("long-task-001")
if saved_state and saved_state.get("in_progress"):
    print(f"Resuming from: {saved_state['current_phase']}")
    # Continue from checkpoint...

# Graceful degradation when providers fail
orchestrator = HybridOrchestrator(
    providers={
        Provider.CLAUDE: claude_provider,
        # Gemini unavailable
    }
)

# Will automatically fall back to MERGE_EXECUTOR_ONLY mode
result = await orchestrator.run(
    request="Add authentication",
    cwd="/project",
)

print(f"Execution mode: {result.execution_mode.value}")
if result.fallback_reason:
    print(f"Fallback: {result.fallback_reason}")

# Cleanup old sessions
cleaned = await recovery.cleanup_old_sessions(max_age_hours=24)
print(f"Cleaned up {cleaned} old session states")
""")

    print("\nPattern: Checkpoint state, resume on failure, degrade gracefully.")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all examples or specific example."""
    examples = {
        "basic": basic_orchestration_example,
        "brainstorm": brainstorming_example,
        "clarity": clarity_gate_example,
        "hybrid": hybrid_orchestrator_example,
        "rag": rag_integration_example,
        "recovery": recovery_example,
    }

    print("\n" + "=" * 60)
    print("Multi-LLM Orchestration Examples")
    print("=" * 60)
    print("""
This module demonstrates the Multi-LLM orchestration system:
- Gemini: Master orchestrator (strategy, routing, coordination)
- Claude: Code worker (implementation, code-heavy tasks)
- Codex: Deep analyzer (debugging, analysis, verification)

Key Concepts:
1. Master-Worker pattern with provider specialization
2. ClarityGate: Workers self-assess before execution
3. ConsensusLoop: 3-round majority voting for decisions
4. BrainstormModule: Full discussion with preserved logs
5. HybridOrchestrator: Multi-LLM + MergeExecutor integration
6. RAG Integration: Learn from past routing decisions
""")

    # Run specified example or all
    if len(sys.argv) > 1:
        name = sys.argv[1]
        if name in examples:
            await examples[name]()
        else:
            print(f"Unknown example: {name}")
            print(f"Available: {', '.join(examples.keys())}")
    else:
        # Run all examples
        for name, func in examples.items():
            await func()
            print("\n" + "-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
