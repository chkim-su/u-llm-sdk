---
name: u-llm-sdk
description: Use this skill when the user asks about "LLM SDK", "multi-provider LLM", "Claude/Codex/Gemini integration", "u-llm-sdk", "unified LLM client", "quick_run", "parallel_run", "multi-LLM orchestration", "consensus voting", "brainstorm module", or needs to programmatically call multiple LLM providers.
version: 0.3.0
---

# U-LLM-SDK: Unified LLM SDK

A Python SDK for unified multi-provider LLM access supporting Claude, Codex, and Gemini.

## Installation

```bash
pip install u-llm-sdk
# Or from source
pip install -e /path/to/u-llm-sdk
```

## Quick Start

### One-Shot Execution (Recommended for Simple Tasks)

```python
from u_llm_sdk.core.utils import quick_run, quick_text
from u_llm_sdk.types import Provider

# Get full result
result = await quick_run("What is 2+2?", provider=Provider.CLAUDE)
print(result.text)

# Get just text
answer = await quick_text("What is 2+2?")  # Claude is default
print(answer)  # "4"

# Sync versions
from u_llm_sdk.core.utils import quick_run_sync, quick_text_sync
result = quick_run_sync("Hello!", provider=Provider.GEMINI)
```

### Context Manager (For Multiple Calls)

```python
from u_llm_sdk import LLM, LLMSync, Provider

# Async usage
async with LLM(provider=Provider.CLAUDE) as llm:
    result = await llm.run("Hello!")
    print(result.text)

# Sync usage
llm = LLMSync(provider=Provider.CLAUDE)
result = llm.run("Hello!")
print(result.text)
```

### Auto Provider Selection

```python
# Automatically selects first available provider (Claude > Codex > Gemini)
async with LLM.auto() as llm:
    result = await llm.run("Hello!")
```

### Streaming

```python
async with LLM(provider=Provider.CLAUDE) as llm:
    async for chunk in llm.stream("Tell me a story"):
        if chunk.get("type") == "text":
            print(chunk["content"], end="", flush=True)
```

### Configuration

```python
from u_llm_sdk import LLM, Provider
from u_llm_sdk.config import LLMConfig
from u_llm_sdk.types import AutoApproval, SandboxMode, ModelTier

config = LLMConfig(
    provider=Provider.CLAUDE,
    tier=ModelTier.HIGH,           # HIGH or LOW tier models
    auto_approval=AutoApproval.FULL,
    sandbox=SandboxMode.WORKSPACE_WRITE,
    timeout=120,
    cwd="/workspace",
    system_prompt="You are a helpful assistant.",
)

async with LLM(config=config) as llm:
    result = await llm.run("Help me with this code")
```

## Providers

| Provider | CLI | Features |
|----------|-----|----------|
| `Provider.CLAUDE` | `claude` | Extended thinking (ultrathink), MCP config |
| `Provider.CODEX` | `codex` | Image inputs, web search, output schema |
| `Provider.GEMINI` | `gemini` | Temperature control, extensions, tools |

## Key Types

```python
from u_llm_sdk.types import (
    # Enums
    Provider,           # CLAUDE, CODEX, GEMINI
    AutoApproval,       # NONE, EDITS_ONLY, FULL
    SandboxMode,        # NONE, READ_ONLY, WORKSPACE_WRITE, FULL_ACCESS
    ModelTier,          # HIGH, LOW
    ReasoningLevel,     # NONE, LOW, MEDIUM, HIGH, XHIGH

    # Data models
    LLMResult,          # Unified result from any provider
    FileChange,         # File modification info
    CommandRun,         # Command execution info
    TokenUsage,         # Token usage stats

    # Hook data
    PreActionContext,   # Context before LLM call
    PostActionFeedback, # Feedback after LLM call

    # Exceptions
    UnifiedLLMError,
    ProviderNotFoundError,
    AuthenticationError,
)
```

## LLMResult Structure

```python
@dataclass
class LLMResult:
    success: bool
    result_type: ResultType  # TEXT, CODE, FILE_EDIT, COMMAND, ERROR, MIXED
    summary: str
    text: str
    files_modified: List[FileChange]
    commands_run: List[CommandRun]
    error_message: Optional[str]
    duration_ms: int
```

## Parallel Execution

```python
from u_llm_sdk.core.utils import parallel_run

# Execute multiple prompts in parallel
results = await parallel_run(
    ["Question 1?", "Question 2?", "Question 3?"],
    provider=Provider.CLAUDE,
)
for r in results:
    print(r.text)
```

## Multi-Provider Execution

```python
from u_llm_sdk.core.utils import multi_provider_run

# Get answers from multiple providers for comparison
results = await multi_provider_run(
    "What is the meaning of life?",
    providers=[Provider.CLAUDE, Provider.GEMINI, Provider.CODEX],
)
for provider, result in results.items():
    print(f"{provider.value}: {result.text[:100]}...")
```

## Multi-LLM Orchestration

### ClarityGate (Task Clarity Assessment)

```python
from u_llm_sdk.multi_llm import ClarityGate
from u_llm_sdk.types.orchestration import Task, ClarityLevel

gate = ClarityGate(claude_provider)
task = Task(task_id="t1", objective="Build auth", context="FastAPI")

assessment = await gate.assess(task)
if assessment.level == ClarityLevel.CLEAR:
    # Proceed with autonomous execution
    pass
elif assessment.recommendation == "escalate":
    # Escalate to orchestrator for clarification
    pass
```

### ConsensusLoop (Multi-Provider Voting)

```python
from u_llm_sdk.multi_llm import ConsensusLoop
from u_llm_sdk.types.orchestration import BrainstormConfig

config = BrainstormConfig(
    max_rounds=3,
    consensus_threshold=0.67,  # 2/3 majority
)

loop = ConsensusLoop(providers, config)
result = await loop.run("Should we use microservices?")

if result.success:
    print(f"Decision: {result.final_decision}")
else:
    print(f"User clarification needed: {result.user_questions}")
```

### BrainstormModule (Full Brainstorming Session)

```python
from u_llm_sdk.multi_llm import BrainstormModule

module = BrainstormModule(providers)
result = await module.run_session("Architecture decision for new feature")

# result.rounds: 3-round full record
# result.consensus: Final consensus result
# result.all_discussion_entries: Complete discussion (no summarization)
```

### HybridOrchestrator (Complete Pipeline)

```python
from u_llm_sdk.multi_llm import HybridOrchestrator, ExecutionMode
from u_llm_sdk.llm.orchestration import MergeExecutorConfig

# Combine multi-LLM decision with code execution
orchestrator = HybridOrchestrator(
    providers=providers,
    merge_config=MergeExecutorConfig(
        integration_branch="llm/feature",
        create_pr=True,
    ),
)

result = await orchestrator.run(
    request="Implement OAuth2 authentication",
    cwd="/project",
    mode=ExecutionMode.HYBRID,
)

if result.merge_result:
    print(f"PR created: {result.merge_result.pr_url}")
```

## Session Management

```python
from u_llm_sdk.session import get_session_manager, SessionTemplate

# Create session with template
manager = get_session_manager(Provider.CLAUDE, "/project")
session_id = manager.create_from_template(SessionTemplate.CODE_REVIEWER)

# Resume session
async with LLM(provider=Provider.CLAUDE) as llm:
    result = await llm.run("Continue", session_id=session_id)
```

## Advanced Features

### Web Search

```python
config = LLMConfig(
    provider=Provider.CLAUDE,
    web_search=True,  # Enables WebSearch, WebFetch tools
)
```

### Extended Thinking (Claude)

```python
config = LLMConfig(
    provider=Provider.CLAUDE,
    reasoning_level=ReasoningLevel.XHIGH,  # Triggers ultrathink
)
```

### Provider Options

```python
# Codex with images
config = LLMConfig(
    provider=Provider.CODEX,
    provider_options={
        "images": ["/path/to/image.png"],
        "search": True,
    },
)

# Gemini with temperature
config = LLMConfig(
    provider=Provider.GEMINI,
    provider_options={
        "temperature": 0.7,
        "allowed_tools": ["edit", "bash", "read"],
    },
)
```

## Import Paths

```python
# Main API
from u_llm_sdk import LLM, LLMSync, Provider, LLMResult

# Configuration
from u_llm_sdk.config import LLMConfig, SAFE_CONFIG, AUTO_CONFIG

# Types
from u_llm_sdk.types import AutoApproval, SandboxMode, ModelTier

# Providers
from u_llm_sdk.llm.providers import ClaudeProvider, CodexProvider, GeminiProvider

# Multi-LLM
from u_llm_sdk.multi_llm import ClarityGate, ConsensusLoop, BrainstormModule

# Session
from u_llm_sdk.session import get_session_manager, SessionTemplate

# Utilities
from u_llm_sdk.core.utils import quick_run, quick_text, parallel_run, template_run

# Advanced Client (multi-agent)
from u_llm_sdk.advanced import UnifiedAdvanced, AdvancedConfig, AgentDefinition
```

## Template Execution

```python
from u_llm_sdk.session import SessionTemplate
from u_llm_sdk.core.utils import template_run

# Execute with specialized template
result = await template_run(
    "Review auth.py for security issues",
    SessionTemplate.SECURITY_ANALYST,
    provider=Provider.CLAUDE,
)

# Available templates:
# CODE_REVIEWER, SECURITY_ANALYST, PYTHON_EXPERT, TYPESCRIPT_EXPERT,
# API_DESIGNER, TEST_ENGINEER, PERFORMANCE_ANALYST, DOCUMENTATION_WRITER,
# CRPG_STORYTELLER, CUSTOM_TELLER
```

## Advanced Client (Multi-Agent)

```python
from u_llm_sdk.advanced import UnifiedAdvanced, AgentDefinition
from u_llm_sdk.types import ModelTier

# Define agents
planner = AgentDefinition(
    name="planner",
    description="Plans implementation tasks",
    system_prompt="You are a planning expert.",
    tier=ModelTier.HIGH,
    allowed_tools=["Read", "Grep", "Glob"],
)

coder = AgentDefinition(
    name="coder",
    description="Writes code",
    system_prompt="You are a coding expert.",
    provider=Provider.CODEX,  # Use different provider
)

# Run agents sequentially
async with UnifiedAdvanced(provider=Provider.CLAUDE) as client:
    results = await client.run_with_agents(
        "Implement user authentication",
        [planner, coder],
        sequential=True,
    )
```
