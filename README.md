# U-llm-sdk

Unified LLM execution layer with multi-provider support and orchestration.

## Overview

U-llm-sdk provides a unified interface for multiple LLM providers (Claude, Codex, Gemini) with:

- **Multi-Provider Support**: Claude, Codex, Gemini with consistent API
- **Quick Utilities**: `quick_run()`, `quick_text()` for one-shot executions
- **Orchestration**: Contract-based parallel execution with WorkOrders
- **RAG Integration**: MV-rag HTTP client with fail-open design

## Installation

```bash
# 기본 설치 (llm-types 자동 포함)
pip install u-llm-sdk
```

> **Note**: `llm-types` 패키지가 자동으로 설치됩니다. 별도 설치 불필요.

### Optional: Experience Learning (MV-rag)

경험 학습 기능을 사용하려면 MV-rag 서버를 별도로 설치하고 실행하세요:

```bash
pip install mv-rag
python -m uvicorn mv_rag.api.server:app --port 8000
```

**자동 감지**: MV-rag가 설치되고 서버가 실행 중이면, U-llm-sdk가 자동으로 연결합니다.

```python
# MV-rag 상태 확인
from u_llm_sdk import is_mv_rag_installed, is_mv_rag_running

print(is_mv_rag_installed())  # True if pip install mv-rag
print(is_mv_rag_running())    # True if server is up

# 자동 감지 비활성화
async with LLM(config, auto_rag=False) as llm:
    result = await llm.run("...")  # RAG 연동 없이 실행
```

MV-rag 없이도 U-llm-sdk는 완전히 동작합니다 (`fail_open=True` 기본값).

## Quick Start

### Simple LLM Call

```python
from u_llm_sdk import LLM
from llm_types import Provider

async def main():
    async with LLM(provider=Provider.CLAUDE) as llm:
        result = await llm.run("Explain Python generators in 3 sentences")
        print(result.text)

import asyncio
asyncio.run(main())
```

### One-Shot Execution

```python
from u_llm_sdk.core.utils import quick_run, quick_text
from llm_types import Provider, ModelTier

# Get full LLMResult
result = await quick_run(
    "Fix the bug in auth.py",
    provider=Provider.CLAUDE,
    tier=ModelTier.HIGH,
)

# Get just the text
text = await quick_text("Summarize this code", provider=Provider.GEMINI)
```

### Synchronous API

```python
from u_llm_sdk import LLMSync
from llm_types import Provider

llm = LLMSync(provider=Provider.CLAUDE)
result = llm.run("Hello, world!")
print(result.text)
```

### Streaming

```python
from u_llm_sdk import LLM
from llm_types import Provider

async with LLM(provider=Provider.CLAUDE) as llm:
    async for chunk in llm.stream("Write a haiku about coding"):
        print(chunk.get("text", ""), end="", flush=True)
```

## Providers

### Claude (Anthropic)

```python
from u_llm_sdk.llm.providers import ClaudeProvider
from u_llm_sdk.config import LLMConfig

config = LLMConfig(provider=Provider.CLAUDE, tier=ModelTier.HIGH)
provider = ClaudeProvider(config)
result = await provider.run("Your prompt here")
```

### Codex (OpenAI)

```python
from u_llm_sdk.llm.providers import CodexProvider

config = LLMConfig(provider=Provider.CODEX, tier=ModelTier.LOW)
provider = CodexProvider(config)
result = await provider.run("Your prompt here")
```

### Gemini (Google)

```python
from u_llm_sdk.llm.providers import GeminiProvider

config = LLMConfig(provider=Provider.GEMINI, tier=ModelTier.LOW)
provider = GeminiProvider(config)
result = await provider.run("Your prompt here")
```

## Orchestration

U-llm-sdk includes a full orchestration framework for parallel code editing:

### StateMachineOrchestrator

```python
from u_llm_sdk.llm.orchestration import StateMachineOrchestrator

orchestrator = StateMachineOrchestrator(max_parallel_editors=3)
ctx = await orchestrator.run(
    objective="Implement user authentication",
    cwd="/path/to/repo",
)
print(ctx.state)  # ExecutionState.COMPLETED
```

### MergeExecutor

**Parallel code editing pipeline** - NOT a simple "merger", but an algorithm to solve conflicts in parallel editing:

```
Pipeline:
1. Clarify   → Transform instruction into executable contract
2. Split     → Decompose into WorkOrders (file_set, locks)
3. Parallel  → Execute in isolated worktrees (Claude Workers)
4. Validate  → Per-branch Evidence (tests, typecheck, lint)
5. Merge     → Only passing branches → Integration Branch
6. Final     → Final validation on integration branch
7. Push/PR   → Push to remote or create PR
```

```python
from u_llm_sdk.llm.orchestration import MergeExecutor, MergeExecutorConfig

config = MergeExecutorConfig(
    base_branch="main",
    integration_branch="llm/feature-auth",
    max_parallel=3,              # Max parallel WorkOrders
    require_all_pass=True,       # All WorkOrders must pass
    require_tests=True,          # pytest must pass
    require_typecheck=True,      # mypy must pass
    create_pr=True,
)
executor = MergeExecutor(config)
result = await executor.run(
    instruction="Add user login functionality",
    cwd="/path/to/repo",
)
print(result.to_summary())
print(f"Integration commit: {result.integration_commit[:8]}")
```

### WorkOrder Contracts

```python
from u_llm_sdk.llm.orchestration import WorkOrder, Evidence

# Define a WorkOrder contract
wo = WorkOrder(
    id="WO-001",
    objective="Implement login endpoint",
    file_set=["src/auth/**/*.py"],
    constraints=["No external dependencies"],
    resource_locks=["auth:api"],
)
```

## Multi-LLM Orchestration

U-llm-sdk includes a complete multi-LLM orchestration framework for collaborative AI workflows:

### Orchestration Modes (NEW)

| Mode | Description | Use Case |
|------|-------------|----------|
| **ORIGINAL_STRICT** | Master 완전 제어, ClarityGate 필수 | 높은 예측성 필요 시 |
| **SEMI_AUTONOMOUS** | 설계는 Multi-LLM, 구현은 Claude Code 위임 | 효율성과 안정성 균형 |

```python
from u_llm_sdk.multi_llm import EnhancedMasterOrchestrator
from llm_types import OrchestrationMode, BoundaryConstraints

orchestrator = EnhancedMasterOrchestrator(providers)

# SEMI_AUTONOMOUS: Design → Implement → Review
result = await orchestrator.run(
    "Implement user authentication",
    cwd="/project",
    mode=OrchestrationMode.SEMI_AUTONOMOUS,
    boundaries=BoundaryConstraints(
        max_budget_usd=2.0,
        file_scope=["src/**/*.py"],
        require_tests=True,
    ),
)

if result.success:
    print(f"Files modified: {result.delegation_outcome.files_modified}")
    print(f"Cost: ${result.delegation_outcome.budget_used_usd:.2f}")
```

### Available Components

| Component | Purpose |
|-----------|---------|
| `EnhancedMasterOrchestrator` | **Mode-aware orchestrator** (ORIGINAL_STRICT / SEMI_AUTONOMOUS) |
| `ClaudeCodeExecutor` | **Claude Code delegation** with boundary enforcement |
| `BoundaryValidator` | **Constraint enforcement** (budget, files, timeout) |
| `ClarityGate` | Worker self-assessment before task execution |
| `EscalationProtocol` | Upward communication for unclear tasks |
| `ConsensusLoop` | 3-round majority voting (2/3 threshold) |
| `BrainstormModule` | Full 3-round brainstorming sessions |
| `GeminiOrchestrator` | Master orchestrator (human intent, routing) |
| `ClaudeOrchestrator` | Code-focused sub-orchestrator |
| `CodexOrchestrator` | Analysis-focused sub-orchestrator |
| `HybridOrchestrator` | Multi-LLM + MergeExecutor (parallel editing pipeline) |
| `SessionManager` | Session state management |

### Provider Specializations

```
┌─────────────────────────────────────────────────────────────┐
│  GEMINI (Master)     │  CLAUDE (Worker)   │  CODEX (Analyst) │
├──────────────────────┼────────────────────┼──────────────────┤
│  • Human intent      │  • Code impl       │  • Deep analysis │
│  • Session mgmt      │  • Refactoring     │  • Risk analysis │
│  • Task routing      │  • Bug fixes       │  • Debugging     │
│  • Strategy/Design   │  • Tests           │  • Theory        │
└──────────────────────┴────────────────────┴──────────────────┘
```

### Basic Usage

```python
from u_llm_sdk.multi_llm import (
    GeminiOrchestrator,
    OrchestratorFactory,
    BrainstormModule,
    ClarityGate,
)
from llm_types import Provider

# Create orchestrator
providers = {
    Provider.GEMINI: gemini_provider,
    Provider.CLAUDE: claude_provider,
    Provider.CODEX: codex_provider,
}
factory = OrchestratorFactory(providers)
master = factory.create_master()

# Process request with automatic routing
response = await master.process_request("Implement user authentication")
```

### Brainstorming Session

```python
from u_llm_sdk.multi_llm import BrainstormModule
from llm_types import BrainstormConfig

module = BrainstormModule(providers)
result = await module.run_session("Best approach for caching layer")

if result.consensus.success:
    print(f"Decision: {result.consensus.final_decision}")
else:
    print(f"Questions for user: {result.consensus.user_questions}")
```

### Clarity Gate (Worker Self-Assessment)

```python
from u_llm_sdk.multi_llm import ClarityGate
from llm_types import Task, ClarityLevel

gate = ClarityGate(claude_provider)
task = Task(task_id="t1", objective="Implement OAuth2", context="FastAPI")

assessment = await gate.assess(task)
if assessment.level == ClarityLevel.CLEAR:
    # Proceed with autonomous execution
elif assessment.recommendation == "escalate":
    # Query orchestrator for clarification
```

### Session Management

```python
from u_llm_sdk.multi_llm import SessionManager, InMemorySessionStore

store = InMemorySessionStore()
manager = SessionManager(store)

# Create session
state = await manager.create_session("session-001", Provider.GEMINI)

# Switch orchestrator (state preserved)
await manager.switch_orchestrator("session-001", Provider.CLAUDE)
```

### Orchestration Mode Flows

#### ORIGINAL_STRICT Flow

```
Request → GeminiOrchestrator.process_request()
       → [Brainstorm if complex]
       → For each task:
           → ClarityGate.assess() [MANDATORY]
           → If AMBIGUOUS: EscalationProtocol
           → If CLEAR: Worker.run() [single response]
       → aggregate_results()
```

#### SEMI_AUTONOMOUS Flow

```
Request → Phase 1: BrainstormModule.run_session()
                   → ConsensusResult (design_context)
       → Phase 2: ClaudeCodeExecutor.execute()
                   → stream-json monitoring
                   → BoundaryValidator enforcement
                   → DelegationOutcome
       → Phase 3: Codex review [OPTIONAL]
```

### Boundary Constraints

SEMI_AUTONOMOUS mode enforces hard limits that Claude Code cannot exceed:

```python
from llm_types import BoundaryConstraints

boundaries = BoundaryConstraints(
    max_budget_usd=2.0,           # Budget limit (terminates on exceed)
    max_timeout_seconds=600,      # 10 minute timeout
    file_scope=["src/**/*.py"],   # Only these files can be modified
    forbidden_paths=[".env", "secrets/**"],  # Never touch these
    max_files_modified=20,        # File count limit
    require_tests=True,           # Must pass pytest
    require_typecheck=True,       # Must pass mypy
    allow_shell_commands=True,    # Allow Bash tool
    allow_web_access=False,       # Block WebFetch/WebSearch
)
```

Violations result in immediate termination with `BoundaryViolationError`.

## RAG Integration

```python
from u_llm_sdk.rag_client import RAGClient, RAGClientConfig

config = RAGClientConfig(
    base_url="http://localhost:8000",
    timeout_seconds=0.5,
    fail_open=True,
)

async with RAGClient(config) as client:
    # Pre-action context injection
    pre_ctx = await client.on_pre_action(
        prompt="Fix the bug",
        provider="claude",
    )

    # ... LLM execution ...

    # Post-action feedback
    await client.on_post_action(result, pre_ctx)
```

## Package Structure

```
U-llm-sdk/
├── src/u_llm_sdk/
│   ├── config.py              # LLMConfig, presets
│   ├── core/
│   │   ├── discovery.py       # CLI path discovery
│   │   └── utils.py           # quick_run, quick_text
│   ├── llm/
│   │   ├── client.py          # LLM, LLMSync
│   │   ├── providers/
│   │   │   ├── base.py        # BaseProvider
│   │   │   ├── claude.py      # ClaudeProvider
│   │   │   ├── codex.py       # CodexProvider
│   │   │   ├── gemini.py      # GeminiProvider
│   │   │   └── hooks.py       # InterventionHook
│   │   └── orchestration/     # Contract-based orchestration
│   │       ├── contracts.py   # WorkOrder, Evidence, ROLES
│   │       ├── state_machine.py
│   │       ├── merge_executor.py
│   │       └── ...
│   ├── multi_llm/             # Multi-LLM orchestration
│   │   ├── clarity.py         # ClarityGate
│   │   ├── escalation.py      # EscalationProtocol
│   │   ├── consensus.py       # ConsensusLoop
│   │   ├── brainstorm.py      # BrainstormModule
│   │   ├── orchestrator.py    # Gemini/Claude/Codex Orchestrators
│   │   ├── enhanced_orchestrator.py  # EnhancedMasterOrchestrator (NEW)
│   │   ├── claude_executor.py # ClaudeCodeExecutor (NEW)
│   │   ├── boundary_validation.py    # BoundaryValidator (NEW)
│   │   ├── session.py         # SessionStore, SessionManager
│   │   ├── rag_integration.py # MultiLLMRAGHook
│   │   ├── migration.py       # HybridOrchestrator
│   │   ├── performance.py     # ParallelExecutor, LatencyTracker
│   │   └── monitoring.py      # OrchestrationLogger, DebugMode
│   └── rag_client/            # MV-rag HTTP client
```

## Configuration

### LLMConfig

```python
from u_llm_sdk.config import LLMConfig, SAFE_CONFIG, AUTO_CONFIG
from llm_types import Provider, ModelTier, AutoApproval

# Use presets
config = SAFE_CONFIG.with_provider(Provider.CLAUDE)

# Or customize
config = LLMConfig(
    provider=Provider.CLAUDE,
    tier=ModelTier.HIGH,
    auto_approval=AutoApproval.FULL,
    timeout_seconds=120,
    cwd="/path/to/workspace",
)
```

### Presets

- `SAFE_CONFIG`: Read-only, no auto-approval
- `AUTO_CONFIG`: Full auto-approval, workspace write
- `CLAUDE_CONFIG`, `CODEX_CONFIG`, `GEMINI_CONFIG`: Provider-specific defaults

## License

MIT
