# Advanced Client Reference

Multi-agent orchestration client supporting all providers.

## Import

```python
from u_llm_sdk.advanced import (
    UnifiedAdvanced,      # Async advanced client
    UnifiedAdvancedSync,  # Sync advanced client
    AdvancedConfig,       # Configuration
    AgentDefinition,      # Agent definition
)
```

---

## UnifiedAdvanced

Advanced async client with multi-agent support.

### Constructor

```python
from u_llm_sdk.advanced import UnifiedAdvanced, AdvancedConfig
from u_llm_sdk.types import Provider, ModelTier, AutoApproval

# Simple initialization
async with UnifiedAdvanced(provider=Provider.CLAUDE) as client:
    result = await client.run("Hello!")

# With full config
config = AdvancedConfig(
    provider=Provider.CLAUDE,
    tier=ModelTier.HIGH,
    auto_approval=AutoApproval.FULL,
    max_parallel_agents=3,
    agent_timeout_multiplier=1.5,
)

async with UnifiedAdvanced(config=config) as client:
    result = await client.run("Hello!")
```

### Methods

#### `run(prompt, **kwargs) -> LLMResult`

Basic execution.

```python
result = await client.run("What is 2+2?")
print(result.text)
```

#### `run_with_system_prompt(prompt, system_prompt) -> LLMResult`

Execute with custom system prompt.

```python
result = await client.run_with_system_prompt(
    "Sort [3, 1, 2]",
    "You are a Python expert. Write efficient code.",
)
```

#### `run_with_template(prompt, template) -> LLMResult`

Execute with predefined template.

```python
from u_llm_sdk.session import SessionTemplate

result = await client.run_with_template(
    "Review auth.py",
    SessionTemplate.SECURITY_ANALYST,
)
```

#### `run_with_agent(prompt, agent) -> LLMResult`

Execute with a single agent.

```python
from u_llm_sdk.advanced import AgentDefinition

agent = AgentDefinition(
    name="analyzer",
    description="Code analyzer",
    system_prompt="You analyze code patterns.",
)

result = await client.run_with_agent("Analyze this codebase", agent)
```

#### `run_with_agents(prompt, agents, sequential=False) -> list[LLMResult]`

Execute with multiple agents.

```python
planner = AgentDefinition(
    name="planner",
    description="Plans tasks",
    system_prompt="You plan implementation steps.",
)

coder = AgentDefinition(
    name="coder",
    description="Writes code",
    system_prompt="You write clean code.",
)

# Sequential execution
results = await client.run_with_agents(
    "Build a login feature",
    [planner, coder],
    sequential=True,
)

# Parallel execution
results = await client.run_with_agents(
    "Analyze from different perspectives",
    [agent1, agent2, agent3],
    sequential=False,
)
```

#### `parallel_agents(prompts_and_agents) -> list[LLMResult]`

Run different prompts with different agents in parallel.

```python
results = await client.parallel_agents([
    ("Analyze code", analyzer_agent),
    ("Review tests", reviewer_agent),
    ("Check security", security_agent),
])
```

#### `stream(prompt, **kwargs) -> AsyncIterator[dict]`

Stream responses.

```python
async for chunk in client.stream("Tell me a story"):
    if chunk.get("type") == "text":
        print(chunk["content"], end="")
```

#### `stream_with_agent(prompt, agent) -> AsyncIterator[dict]`

Stream with specific agent.

```python
async for chunk in client.stream_with_agent("Explain step by step", expert_agent):
    if chunk.get("type") == "text":
        print(chunk["content"], end="")
```

#### `get_session_manager() -> BaseSessionManager`

Get session manager for current provider.

```python
manager = client.get_session_manager()
sessions = manager.list_sessions()
```

---

## UnifiedAdvancedSync

Synchronous version of UnifiedAdvanced.

```python
from u_llm_sdk.advanced import UnifiedAdvancedSync

with UnifiedAdvancedSync(provider=Provider.CLAUDE) as client:
    result = client.run("Hello!")
    print(result.text)

    result = client.run_with_system_prompt(
        "Sort [3, 1, 2]",
        "You are a Python expert.",
    )

    result = client.run_with_template(
        "Review this code",
        SessionTemplate.CODE_REVIEWER,
    )

    result = client.run_with_agent("Analyze", analyzer_agent)
```

### Quick Run (Static Method)

```python
result = UnifiedAdvancedSync.quick_run(
    "Hello!",
    provider=Provider.CLAUDE,
)
```

---

## AgentDefinition

Defines an agent's persona and capabilities.

### Constructor

```python
from u_llm_sdk.advanced import AgentDefinition
from u_llm_sdk.types import Provider, ModelTier

agent = AgentDefinition(
    name="my_agent",                  # Required: agent name
    description="What this agent does",  # Required: brief description
    system_prompt="You are...",       # Required: full system prompt
    provider=None,                    # Optional: override client's provider
    tier=ModelTier.HIGH,              # Optional: model tier
    allowed_tools=None,               # Optional: list of allowed tools
    timeout_multiplier=1.0,           # Optional: timeout adjustment
    metadata=None,                    # Optional: custom metadata
)
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | ✅ | Unique agent name |
| `description` | `str` | ✅ | Brief description |
| `system_prompt` | `str` | ✅ | Full system prompt |
| `provider` | `Provider` | ❌ | Override client's provider |
| `tier` | `ModelTier` | ❌ | Model tier (HIGH/LOW) |
| `allowed_tools` | `list[str]` | ❌ | Allowed tool names |
| `timeout_multiplier` | `float` | ❌ | Timeout adjustment |
| `metadata` | `dict` | ❌ | Custom metadata |

### Multi-Provider Agents

Agents can use different providers than the client:

```python
# Client uses Gemini
async with UnifiedAdvanced(provider=Provider.GEMINI) as client:

    # This agent uses Claude
    claude_coder = AgentDefinition(
        name="coder",
        description="Writes code",
        system_prompt="You write excellent code.",
        provider=Provider.CLAUDE,  # Override
    )

    # This agent uses Codex
    codex_analyzer = AgentDefinition(
        name="analyzer",
        description="Analyzes code",
        system_prompt="You analyze code patterns.",
        provider=Provider.CODEX,  # Override
    )

    # Mixed provider workflow
    results = await client.run_with_agents(
        "Build and analyze a feature",
        [claude_coder, codex_analyzer],
        sequential=True,
    )
```

---

## AdvancedConfig

Configuration for advanced client.

### Constructor

```python
from u_llm_sdk.advanced import AdvancedConfig

config = AdvancedConfig(
    provider=Provider.CLAUDE,        # Default provider
    model=None,                      # Specific model
    tier=ModelTier.HIGH,             # Model tier
    auto_approval=AutoApproval.EDITS_ONLY,  # Approval mode
    timeout=1200,                    # Base timeout
    max_parallel_agents=3,           # Max concurrent agents
    agent_timeout_multiplier=1.5,    # Agent timeout adjustment
    system_prompt=None,              # Default system prompt
    cwd=None,                        # Working directory
    provider_options=None,           # Provider-specific options
)
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `Provider` | `CLAUDE` | Default provider |
| `model` | `str` | `None` | Specific model name |
| `tier` | `ModelTier` | `HIGH` | Model tier |
| `auto_approval` | `AutoApproval` | `EDITS_ONLY` | Approval mode |
| `timeout` | `float` | `1200` | Base timeout (seconds) |
| `max_parallel_agents` | `int` | `3` | Max concurrent agents |
| `agent_timeout_multiplier` | `float` | `1.5` | Agent timeout factor |
| `system_prompt` | `str` | `None` | Default system prompt |
| `cwd` | `str` | `None` | Working directory |
| `provider_options` | `dict` | `None` | Provider options |

---

## Complete Example

```python
import asyncio
from u_llm_sdk.advanced import UnifiedAdvanced, AgentDefinition, AdvancedConfig
from u_llm_sdk.types import Provider, ModelTier
from u_llm_sdk.session import SessionTemplate

async def main():
    config = AdvancedConfig(
        provider=Provider.CLAUDE,
        tier=ModelTier.HIGH,
        max_parallel_agents=3,
    )

    async with UnifiedAdvanced(config=config) as client:
        # Define specialized agents
        planner = AgentDefinition(
            name="planner",
            description="Plans implementation",
            system_prompt="You create detailed implementation plans.",
            tier=ModelTier.HIGH,
        )

        coder = AgentDefinition(
            name="coder",
            description="Writes code",
            system_prompt="You write clean, tested code.",
            provider=Provider.CODEX,  # Use Codex for coding
        )

        reviewer = AgentDefinition(
            name="reviewer",
            description="Reviews code",
            system_prompt="You review code for bugs and improvements.",
        )

        # Sequential workflow: plan -> code -> review
        results = await client.run_with_agents(
            "Implement user authentication with JWT",
            [planner, coder, reviewer],
            sequential=True,
        )

        for i, result in enumerate(results):
            print(f"=== Agent {i+1} ===")
            print(result.text[:500])

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Use Cases

### Code Review Pipeline

```python
reviewer = AgentDefinition(
    name="reviewer",
    description="Reviews code",
    system_prompt=get_template_prompt(SessionTemplate.CODE_REVIEWER),
)

security = AgentDefinition(
    name="security",
    description="Security analysis",
    system_prompt=get_template_prompt(SessionTemplate.SECURITY_ANALYST),
)

# Parallel review
results = await client.run_with_agents(
    "Review auth.py",
    [reviewer, security],
    sequential=False,
)
```

### Multi-Provider Comparison

```python
claude_agent = AgentDefinition(
    name="claude",
    description="Claude's view",
    system_prompt="Analyze from Claude's perspective.",
    provider=Provider.CLAUDE,
)

gemini_agent = AgentDefinition(
    name="gemini",
    description="Gemini's view",
    system_prompt="Analyze from Gemini's perspective.",
    provider=Provider.GEMINI,
)

results = await client.run_with_agents(
    "What's the best approach for this architecture?",
    [claude_agent, gemini_agent],
    sequential=False,
)
```
