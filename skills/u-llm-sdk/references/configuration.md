# Configuration Reference

## LLMConfig Parameters

### Basic Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `Provider` | Required | LLM provider (CLAUDE, CODEX, GEMINI) |
| `model` | `str` | `None` | Specific model name (None = provider default) |
| `tier` | `ModelTier` | `ModelTier.HIGH` | Model tier (HIGH or LOW) |
| `timeout` | `int` | `120` | Timeout in seconds |
| `cwd` | `str` | `None` | Working directory |

### Approval & Sandbox

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auto_approval` | `AutoApproval` | `NONE` | Approval mode for actions |
| `sandbox` | `SandboxMode` | `NONE` | Sandbox restrictions |

**AutoApproval modes:**
- `NONE` - Interactive approval required
- `EDITS_ONLY` - Auto-approve file edits only
- `FULL` - Auto-approve all actions

**SandboxMode options:**
- `NONE` - No sandbox
- `READ_ONLY` - Read-only file system
- `WORKSPACE_WRITE` - Can write in workspace only
- `FULL_ACCESS` - Full file system access

### Prompting

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `system_prompt` | `str` | `None` | System prompt (provider support varies) |
| `append_system_prompt` | `str` | `None` | Additional system prompt to append |
| `reasoning_level` | `ReasoningLevel` | `NONE` | Reasoning intensity |

**ReasoningLevel options:**
- `NONE` - No extended thinking
- `LOW` - Basic reasoning
- `MEDIUM` - Moderate reasoning
- `HIGH` - Deep reasoning
- `XHIGH` - Maximum reasoning (Claude ultrathink)

### Session Management

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `str` | `None` | Session ID to resume |
| `continue_session` | `bool` | `False` | Continue existing session |

### Security

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | `None` | API key (usually from env) |
| `env_file` | `str` | `None` | Path to .env file |
| `strict_env_security` | `bool` | `True` | Block on permissive .env permissions |

### Tools & Features

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `allowed_tools` | `List[str]` | `None` | List of allowed tools (None = all) |
| `disallowed_tools` | `List[str]` | `None` | List of disallowed tools |
| `web_search` | `bool` | `False` | Enable web search tools |

### Advanced

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `intervention_hook` | `InterventionHook` | `None` | Hook for pre/post action callbacks |
| `provider_options` | `dict` | `{}` | Provider-specific options |
| `domain_schema` | `DomainSchema` | `None` | Domain-specific output schema |

## Provider-Specific Options

### Claude

```python
config = LLMConfig(
    provider=Provider.CLAUDE,
    provider_options={
        "mcp_config": "/path/to/mcp.json",
        "max_turns": 10,
        "setting_sources": ["path/to/settings"],
    },
)
```

### Codex

```python
config = LLMConfig(
    provider=Provider.CODEX,
    provider_options={
        "images": ["/path/to/image.png"],
        "search": True,
        "full_auto": True,
        "output_schema": "/path/to/schema.json",
        "add_dirs": ["/extra/dir"],
        "features_enable": ["feature1"],
        "features_disable": ["feature2"],
        "config_overrides": {"key": "value"},
    },
)
```

### Gemini

```python
config = LLMConfig(
    provider=Provider.GEMINI,
    provider_options={
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "allowed_tools": ["edit", "bash", "read"],
        "extensions": ["code-review"],
        "include_directories": ["/shared/libs"],
        "allowed_mcp_server_names": ["filesystem"],
    },
)
```

## Presets

### SAFE_CONFIG

```python
from u_llm_sdk.config import SAFE_CONFIG

# Read-only, no auto-approval
config = SAFE_CONFIG
```

### AUTO_CONFIG

```python
from u_llm_sdk.config import AUTO_CONFIG

# Full auto-approval, workspace write
config = AUTO_CONFIG
```

## Builder Pattern

```python
config = LLMConfig(provider=Provider.CLAUDE) \
    .with_web_search() \
    .with_tier(ModelTier.HIGH) \
    .with_reasoning(ReasoningLevel.XHIGH)
```

## Feature Validation

Validate configuration against provider capabilities:

```python
config = LLMConfig(
    provider=Provider.CODEX,
    auto_approval=AutoApproval.EDITS_ONLY,  # Not supported by Codex!
)

# Get structured validation results
results = config.validate_for_provider()
for r in results:
    print(r.to_dict())
    # {
    #   "feature": "auto_approval_edits_only",
    #   "provider": "codex",
    #   "severity": "warning",
    #   "supported": false,
    #   "suggestion": "Use claude or gemini instead...",
    # }
```

## Quick Utility Functions

For simple one-shot executions:

```python
from u_llm_sdk.core.utils import quick_run, quick_text, parallel_run

# One-shot with full result
result = await quick_run("prompt", provider=Provider.CLAUDE)

# One-shot, text only
text = await quick_text("prompt")

# Parallel execution
results = await parallel_run(["p1", "p2", "p3"])
```
