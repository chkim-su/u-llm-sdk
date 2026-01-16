#!/usr/bin/env python
"""Basic Multi-LLM Orchestration Example.

This example demonstrates the fundamental patterns for multi-LLM orchestration:
1. Creating providers and orchestrators
2. Processing requests with the master orchestrator
3. Routing tasks to specialized workers
4. Basic error handling

Requirements:
    - API keys configured for providers
    - u-llm-sdk installed

Usage:
    python basic_orchestration.py
"""

import asyncio
import logging
from typing import Dict, Optional

from u_llm_sdk.types import Provider, Task

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
    """Mock provider for demonstration purposes.

    In real usage, replace with actual LLM instances.
    """

    def __init__(self, provider: Provider):
        self.provider = provider

    async def run(self, prompt: str) -> str:
        """Simulate LLM response."""
        logger.info(f"{self.provider.value} processing: {prompt[:50]}...")
        await asyncio.sleep(0.1)  # Simulate latency
        return f"Response from {self.provider.value}: Processed '{prompt[:30]}...'"


# =============================================================================
# Example 1: Basic Factory Usage
# =============================================================================


async def example_factory_usage():
    """Demonstrate OrchestratorFactory usage.

    Shows how to:
    - Create providers
    - Initialize the factory
    - Create master and sub-orchestrators
    """
    print("\n" + "=" * 60)
    print("Example 1: OrchestratorFactory Usage")
    print("=" * 60)

    from u_llm_sdk.multi_llm import OrchestratorFactory

    # In real usage, these would be actual LLM instances
    # from u_llm_sdk import LLM
    # async with LLM(provider=Provider.GEMINI) as gemini, ...

    providers = {
        Provider.GEMINI: MockProvider(Provider.GEMINI),
        Provider.CLAUDE: MockProvider(Provider.CLAUDE),
        Provider.CODEX: MockProvider(Provider.CODEX),
    }

    # Create factory
    factory = OrchestratorFactory(providers)

    # Create master orchestrator (Gemini by default)
    master = factory.create_master()
    print(f"Master orchestrator: {master.provider.value}")

    # Create sub-orchestrators for specific providers
    claude_worker = factory.create_sub(Provider.CLAUDE)
    codex_analyzer = factory.create_sub(Provider.CODEX)

    print(f"Claude worker: {claude_worker.provider.value}")
    print(f"Codex analyzer: {codex_analyzer.provider.value}")

    # You can also create master from different provider
    claude_master = factory.create_master(Provider.CLAUDE)
    print(f"Alternative master: {claude_master.provider.value}")


# =============================================================================
# Example 2: Request Processing
# =============================================================================


async def example_request_processing():
    """Demonstrate request processing workflow.

    Shows how to:
    - Process user requests
    - Handle different response types
    - Route tasks to workers
    """
    print("\n" + "=" * 60)
    print("Example 2: Request Processing")
    print("=" * 60)

    from u_llm_sdk.multi_llm import OrchestratorFactory

    providers = {
        Provider.GEMINI: MockProvider(Provider.GEMINI),
        Provider.CLAUDE: MockProvider(Provider.CLAUDE),
        Provider.CODEX: MockProvider(Provider.CODEX),
    }

    factory = OrchestratorFactory(providers)
    master = factory.create_master()

    # Process a user request
    request = "Build a user authentication system with JWT tokens"
    print(f"\nProcessing request: {request}")

    response = await master.process_request(request)

    # Handle different response types
    if response.needs_brainstorm:
        print(f"\nComplex task detected!")
        print(f"Brainstorm topic: {response.brainstorm_topic}")
        print("Would normally trigger brainstorming session...")

    elif response.needs_clarification:
        print(f"\nClarification needed!")
        print("Questions for user:")
        for i, question in enumerate(response.clarification_questions, 1):
            print(f"  {i}. {question}")

    else:
        print(f"\nTask decomposition complete!")
        print(f"Generated {len(response.tasks)} tasks:")

        for task in response.tasks:
            # Route each task
            routing = await master.route_task(task)
            print(f"\n  Task: {task.objective}")
            print(f"  Route to: {routing.recommended_worker.value}")
            print(f"  Reason: {routing.routing_reason}")
            print(f"  Confidence: {routing.confidence:.0%}")


# =============================================================================
# Example 3: Task Execution
# =============================================================================


async def example_task_execution():
    """Demonstrate task execution with clarity checks.

    Shows how to:
    - Create tasks
    - Run ClarityGate validation
    - Execute tasks with workers
    """
    print("\n" + "=" * 60)
    print("Example 3: Task Execution with ClarityGate")
    print("=" * 60)

    from u_llm_sdk.multi_llm import ClarityGate, OrchestratorFactory
    from u_llm_sdk.types import ClarityLevel

    providers = {
        Provider.GEMINI: MockProvider(Provider.GEMINI),
        Provider.CLAUDE: MockProvider(Provider.CLAUDE),
        Provider.CODEX: MockProvider(Provider.CODEX),
    }

    # Create ClarityGate with Claude
    gate = ClarityGate(providers[Provider.CLAUDE])

    # Create a task
    task = Task(
        task_id="auth-001",
        objective="Implement JWT authentication middleware",
        context="FastAPI backend with PostgreSQL database",
        constraints=["Use python-jose library", "Support token refresh"],
    )

    print(f"\nAssessing task clarity: {task.objective}")

    # Assess clarity
    assessment = await gate.assess(task)

    print(f"\nClarity Assessment:")
    print(f"  Level: {assessment.level.value}")
    print(f"  Score: {assessment.score:.2f}")
    print(f"  Recommendation: {assessment.recommendation}")

    if assessment.level == ClarityLevel.CLEAR:
        print("\nTask is clear - proceeding with execution...")
        # Execute task
        claude = providers[Provider.CLAUDE]
        result = await claude.run(task.objective)
        print(f"Result: {result}")

    elif assessment.level == ClarityLevel.NEEDS_CLARIFICATION:
        print("\nTask needs clarification:")
        for aspect in assessment.unclear_aspects:
            print(f"  - {aspect.aspect_type}: {aspect.description}")
            print(f"    Need: {aspect.clarification_needed}")

    else:
        print("\nTask is ambiguous - would escalate to orchestrator")
        print("Questions for orchestrator:")
        for q in assessment.self_questions:
            print(f"  - {q}")


# =============================================================================
# Example 4: Result Aggregation
# =============================================================================


async def example_result_aggregation():
    """Demonstrate result aggregation from multiple workers.

    Shows how to:
    - Execute tasks with multiple workers
    - Aggregate results
    - Handle partial failures
    """
    print("\n" + "=" * 60)
    print("Example 4: Result Aggregation")
    print("=" * 60)

    from u_llm_sdk.multi_llm import (
        OrchestratorFactory,
        WorkerResult,
        ParallelExecutor,
    )

    providers = {
        Provider.GEMINI: MockProvider(Provider.GEMINI),
        Provider.CLAUDE: MockProvider(Provider.CLAUDE),
        Provider.CODEX: MockProvider(Provider.CODEX),
    }

    factory = OrchestratorFactory(providers)
    master = factory.create_master()

    # Simulate worker results
    worker_results = [
        WorkerResult(
            task_id="task-1",
            provider=Provider.CLAUDE,
            success=True,
            output="Implemented JWT middleware",
            latency_ms=150.0,
        ),
        WorkerResult(
            task_id="task-2",
            provider=Provider.CODEX,
            success=True,
            output="Verified security patterns",
            latency_ms=200.0,
        ),
        WorkerResult(
            task_id="task-3",
            provider=Provider.CLAUDE,
            success=True,
            output="Added unit tests",
            latency_ms=180.0,
        ),
    ]

    print(f"\nAggregating {len(worker_results)} worker results...")

    # Aggregate results
    aggregated = await master.aggregate_results(worker_results)

    print(f"\nAggregated Result:")
    print(f"  Success: {aggregated.success}")
    print(f"  Combined output: {aggregated.combined_output[:100]}...")
    print(f"  Total latency: {aggregated.total_latency_ms:.0f}ms")
    print(f"  Tasks completed: {aggregated.tasks_completed}/{aggregated.tasks_total}")


# =============================================================================
# Example 5: Performance Tracking
# =============================================================================


async def example_performance_tracking():
    """Demonstrate performance tracking and metrics.

    Shows how to:
    - Track latency across providers
    - Use parallel execution
    - Export metrics
    """
    print("\n" + "=" * 60)
    print("Example 5: Performance Tracking")
    print("=" * 60)

    from u_llm_sdk.multi_llm import (
        PerformanceOptimizer,
        MetricExporter,
    )

    providers = {
        Provider.GEMINI: MockProvider(Provider.GEMINI),
        Provider.CLAUDE: MockProvider(Provider.CLAUDE),
        Provider.CODEX: MockProvider(Provider.CODEX),
    }

    # Create optimizer
    optimizer = PerformanceOptimizer(
        max_concurrency=3,
        timeout_seconds=30,
    )

    # Track latency for individual operations
    print("\nTracking individual operations...")

    async with optimizer.track_latency(Provider.CLAUDE, "generate"):
        await providers[Provider.CLAUDE].run("Generate code")

    async with optimizer.track_latency(Provider.GEMINI, "route"):
        await providers[Provider.GEMINI].run("Route task")

    async with optimizer.track_latency(Provider.CODEX, "analyze"):
        await providers[Provider.CODEX].run("Analyze code")

    # Parallel execution
    print("\nExecuting tasks in parallel...")

    tasks = [
        lambda: providers[Provider.CLAUDE].run("Task 1"),
        lambda: providers[Provider.GEMINI].run("Task 2"),
        lambda: providers[Provider.CODEX].run("Task 3"),
    ]

    results = await optimizer.parallel_execute(tasks, "batch_process")

    for r in results:
        status = "success" if r.success else "failed"
        print(f"  Task {r.index}: {status} ({r.latency_ms:.1f}ms)")

    # Get metrics summary
    print("\nMetrics Summary:")
    print(optimizer.get_summary())

    # Export to JSON
    exporter = MetricExporter(
        optimizer.metrics,
        prefix="example",
        labels={"env": "demo"},
    )
    print("\nJSON Export (first 500 chars):")
    print(exporter.to_json()[:500] + "...")


# =============================================================================
# Example 6: Monitoring and Logging
# =============================================================================


async def example_monitoring():
    """Demonstrate monitoring and structured logging.

    Shows how to:
    - Use structured logging
    - Emit and handle events
    - Debug mode
    """
    print("\n" + "=" * 60)
    print("Example 6: Monitoring and Logging")
    print("=" * 60)

    from u_llm_sdk.multi_llm import (
        OrchestrationLogger,
        EventEmitter,
        EventType,
        DebugMode,
    )

    # Structured logging
    print("\nStructured Logging:")
    logger = OrchestrationLogger(
        session_id="demo-session-001",
        json_output=False,  # Human-readable for demo
    )

    correlation_id = logger.new_correlation_id()
    print(f"Correlation ID: {correlation_id}")

    with logger.operation("process_request", Provider.GEMINI):
        logger.info("Starting request processing")
        logger.log_request("Build authentication system")
        logger.log_routing("auth-task", Provider.CLAUDE, "Code implementation")

    # Event emitter
    print("\nEvent Emission:")
    emitter = EventEmitter(max_history=100)

    # Register event handler
    @emitter.on(EventType.TASK_COMPLETED)
    async def on_task_complete(event):
        print(f"  Event received: {event.event_type.value}")
        print(f"  Data: {event.data}")

    # Emit event
    await emitter.emit(
        EventType.TASK_COMPLETED,
        session_id="demo-001",
        provider=Provider.CLAUDE,
        task="auth-implementation",
    )

    # Debug mode
    print("\nDebug Mode:")
    with DebugMode(capture_logs=True, profile=True) as debug:
        # Simulated operations
        await asyncio.sleep(0.05)

    print(f"  Logs captured: {len(debug.get_logs())}")
    print(f"  Profile entries: {len(debug.get_profile())}")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Multi-LLM Orchestration - Basic Examples")
    print("=" * 60)

    examples = [
        ("Factory Usage", example_factory_usage),
        ("Request Processing", example_request_processing),
        ("Task Execution", example_task_execution),
        ("Result Aggregation", example_result_aggregation),
        ("Performance Tracking", example_performance_tracking),
        ("Monitoring", example_monitoring),
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
