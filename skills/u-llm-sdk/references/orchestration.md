# Multi-LLM Orchestration Reference

Components for coordinating multiple LLM providers.

## Import

```python
from u_llm_sdk.multi_llm import (
    # Core Components
    ClarityGate,
    EscalationProtocol,
    ConsensusLoop,
    BrainstormModule,

    # Orchestrators
    GeminiOrchestrator,
    ClaudeOrchestrator,
    CodexOrchestrator,
    OrchestratorFactory,

    # Hybrid Integration
    HybridOrchestrator,
    ExecutionMode,
    MigrationHelper,
    StateRecoveryManager,

    # RAG Integration
    MultiLLMRAGHook,
    NoOpMultiLLMRAGHook,

    # Session Management
    SessionManager,
    InMemorySessionStore,
    FileSessionStore,
)
```

---

## ClarityGate

Assesses task clarity before execution. Workers use this to decide if they can proceed autonomously.

### Constructor

```python
gate = ClarityGate(provider: BaseProvider)
```

### Methods

#### `assess(task: Task) -> ClarityAssessment`

```python
from u_llm_sdk.multi_llm import ClarityGate
from u_llm_sdk.types.orchestration import Task, ClarityLevel

gate = ClarityGate(claude_provider)

task = Task(
    task_id="t1",
    objective="Build user authentication",
    context="FastAPI backend, PostgreSQL database",
)

assessment = await gate.assess(task)

print(assessment.level)  # ClarityLevel.CLEAR, UNCLEAR, or AMBIGUOUS
print(assessment.recommendation)  # "proceed", "escalate", or "clarify"
print(assessment.missing_info)  # List of missing information
print(assessment.confidence)  # 0.0 - 1.0
```

### ClarityLevel Enum

```python
from u_llm_sdk.types.orchestration import ClarityLevel

ClarityLevel.CLEAR      # Task is clear, can proceed
ClarityLevel.UNCLEAR    # Needs clarification
ClarityLevel.AMBIGUOUS  # Multiple interpretations possible
```

---

## EscalationProtocol

Handles upward queries when clarity is insufficient.

### Constructor

```python
protocol = EscalationProtocol(orchestrator: BaseOrchestrator)
```

### Methods

#### `escalate(request: EscalationRequest) -> EscalationResponse`

```python
from u_llm_sdk.multi_llm import EscalationProtocol
from u_llm_sdk.types.orchestration import EscalationRequest

protocol = EscalationProtocol(gemini_orchestrator)

request = EscalationRequest(
    source_worker=Provider.CLAUDE,
    original_task=task,
    clarity_assessment=assessment,
    specific_questions=[
        "Which OAuth provider should I use?",
        "Should I support social login?",
    ],
)

response = await protocol.escalate(request)

print(response.refined_task)  # Clarified task
print(response.clarifications)  # Answers to questions
print(response.additional_context)  # Extra guidance
```

---

## ConsensusLoop

Multi-provider voting for decisions (3-round majority).

### Constructor

```python
from u_llm_sdk.multi_llm import ConsensusLoop
from u_llm_sdk.types.orchestration import BrainstormConfig

config = BrainstormConfig(
    max_rounds=3,                    # Maximum voting rounds
    consensus_threshold=0.67,        # 2/3 majority required
    low_agreement_threshold=0.4,     # Below this → ask user
    timeout_per_round=60,            # Seconds per round
)

loop = ConsensusLoop(providers, config)
```

### Methods

#### `run(question: str) -> ConsensusResult`

```python
providers = {
    Provider.GEMINI: gemini_provider,
    Provider.CLAUDE: claude_provider,
    Provider.CODEX: codex_provider,
}

loop = ConsensusLoop(providers, config)
result = await loop.run("Should we use microservices or monolith?")

if result.success:
    print(f"Consensus: {result.final_decision}")
    print(f"Agreement: {result.agreement_score}")  # 0.0 - 1.0
else:
    print(f"No consensus after {result.rounds_completed} rounds")
    print(f"Questions for user: {result.user_questions}")
```

### ConsensusResult

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether consensus was reached |
| `final_decision` | `Optional[str]` | The consensus decision |
| `agreement_score` | `float` | Agreement level (0.0-1.0) |
| `rounds_completed` | `int` | Number of rounds executed |
| `user_questions` | `list[str]` | Questions for user if no consensus |
| `votes` | `dict` | Provider votes per round |

---

## BrainstormModule

Full 3-round brainstorming session with complete record preservation.

### Constructor

```python
from u_llm_sdk.multi_llm import BrainstormModule

module = BrainstormModule(
    providers: dict[Provider, BaseProvider],
    config: Optional[BrainstormConfig] = None,
)
```

### Methods

#### `run_session(topic: str) -> BrainstormResult`

```python
module = BrainstormModule(providers)
result = await module.run_session("Architecture for real-time chat app")

# Full 3-round record (no summarization!)
for i, round_data in enumerate(result.rounds, 1):
    print(f"=== Round {i} ===")
    for entry in round_data.entries:
        print(f"{entry.provider.value}: {entry.content[:100]}...")

# Final consensus
print(f"Consensus: {result.consensus}")

# All discussion entries
for entry in result.all_discussion_entries:
    print(f"[{entry.provider.value}] {entry.content}")
```

### BrainstormResult

| Field | Type | Description |
|-------|------|-------------|
| `rounds` | `list[RoundData]` | Full 3-round records |
| `consensus` | `Optional[str]` | Final consensus |
| `all_discussion_entries` | `list[DiscussionEntry]` | Complete discussion |
| `metadata` | `dict` | Session metadata |

**Key Principle**: All discussion content is preserved without summarization.

---

## Orchestrators

Master-Worker pattern orchestrators for each provider.

### OrchestratorFactory

```python
from u_llm_sdk.multi_llm import OrchestratorFactory

factory = OrchestratorFactory(providers)

# Create master (default: Gemini)
master = factory.create_master()

# Create specific orchestrator
claude_orch = factory.create_sub(Provider.CLAUDE)
codex_orch = factory.create_sub(Provider.CODEX)
```

### Orchestrator Types

| Orchestrator | Role | Specialization |
|--------------|------|----------------|
| `GeminiOrchestrator` | Master | Strategy, intent parsing, routing |
| `ClaudeOrchestrator` | Sub | Code implementation, ClarityGate required |
| `CodexOrchestrator` | Sub | Analysis, verification, scope-limited |

### Processing Requests

```python
from u_llm_sdk.multi_llm import GeminiOrchestrator

master = GeminiOrchestrator(providers)

response = await master.process_request("Build auth system")

if response.needs_brainstorm:
    # Facilitate multi-LLM brainstorm
    consensus = await master.facilitate_brainstorm(response.brainstorm_topic)

elif response.needs_clarification:
    # Ask user for clarification
    questions = response.clarification_questions

else:
    # Route to worker
    for task in response.tasks:
        routing = await master.route_task(task)
        print(f"Task {task.task_id} -> {routing.assigned_provider.value}")
```

---

## HybridOrchestrator

Combines Multi-LLM orchestration with MergeExecutor for code changes.

### Constructor

```python
from u_llm_sdk.multi_llm import HybridOrchestrator, ExecutionMode
from u_llm_sdk.llm.orchestration import MergeExecutorConfig

merge_config = MergeExecutorConfig(
    base_branch="main",
    integration_branch="llm/feature",
    create_pr=True,
    require_tests=True,
)

orchestrator = HybridOrchestrator(
    providers=providers,
    merge_config=merge_config,
    brainstorm_threshold=0.6,  # Trigger brainstorm below this clarity
)
```

### Execution Modes

```python
from u_llm_sdk.multi_llm import ExecutionMode

ExecutionMode.HYBRID             # Multi-LLM decision + MergeExecutor
ExecutionMode.MERGE_EXECUTOR_ONLY  # MergeExecutor only (Claude)
ExecutionMode.MULTI_LLM          # Multi-LLM only (no code execution)
ExecutionMode.FALLBACK           # Fallback mode if providers unavailable
```

### Running

```python
result = await orchestrator.run(
    request="Implement OAuth2 authentication",
    cwd="/project",
    mode=ExecutionMode.HYBRID,
    session_id="auth-001",  # For recovery
)

print(f"Success: {result.success}")
print(f"Mode used: {result.execution_mode.value}")

if result.merge_result:
    print(f"PR URL: {result.merge_result.pr_url}")
    print(f"Branch: {result.merge_result.branch}")
```

---

## StateRecoveryManager

Saves and recovers orchestration state for session continuity.

```python
from u_llm_sdk.multi_llm import StateRecoveryManager
from pathlib import Path

recovery = StateRecoveryManager(Path("~/.cache/state").expanduser())

# Save state
await recovery.save_state("session-001", {
    "in_progress": True,
    "phase": "brainstorming",
    "tasks_completed": ["t1", "t2"],
})

# Load state
state = await recovery.load_state("session-001")
if state and state.get("in_progress"):
    # Resume from saved state
    pass

# Cleanup old sessions
await recovery.cleanup_old_sessions(max_age_hours=24)
```

---

## SessionManager

Manages orchestration sessions.

### Session Stores

```python
from u_llm_sdk.multi_llm import (
    SessionManager,
    InMemorySessionStore,  # For development/testing
    FileSessionStore,      # For persistence
)

# In-memory (lost on restart)
store = InMemorySessionStore()

# File-based (persistent)
store = FileSessionStore(Path("~/.cache/sessions").expanduser())

manager = SessionManager(store)
```

### Usage

```python
# Create session
state = await manager.create_session(
    session_id="session-001",
    orchestrator=Provider.GEMINI,
)

# Switch orchestrator (preserves state)
await manager.switch_orchestrator("session-001", Provider.CLAUDE)

# Add task
task = Task(task_id="t1", objective="Build auth", context="")
await manager.add_task("session-001", task)

# Complete task
await manager.complete_task("session-001", "t1")

# Get summary
summary = await manager.get_session_summary("session-001")
print(f"Active tasks: {summary['active_tasks_count']}")

# Event handling
manager.on("orchestrator_switched", lambda data: print(f"Switched: {data}"))
```

---

## MultiLLMRAGHook

Learns orchestration patterns for future routing decisions.

```python
from u_llm_sdk.multi_llm import MultiLLMRAGHook, RAGTaskType
from u_llm_sdk.rag_client import RAGClientConfig

config = RAGClientConfig(base_url="http://localhost:8000")
hook = MultiLLMRAGHook(config)

# Save routing decision
await hook.save_routing_decision(
    request="Implement auth",
    task_type=RAGTaskType.CODE_IMPLEMENTATION,
    assigned_provider=Provider.CLAUDE,
    routing_reason="Code task -> Claude",
)

# Get hints for new request
hints = await hook.get_orchestration_hints("Build login system")
if hints.suggested_provider:
    print(f"Suggested: {hints.suggested_provider.value}")

# For testing (no API calls)
noop = NoOpMultiLLMRAGHook()
```
