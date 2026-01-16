# Quick Utilities Reference

Convenience functions for one-shot LLM execution without managing client lifecycle.

## Import

```python
from u_llm_sdk.core.utils import (
    # Async
    quick_run,
    quick_text,
    auto_run,
    parallel_run,
    multi_provider_run,
    structured_run,
    template_run,
    # Sync
    quick_run_sync,
    quick_text_sync,
    auto_run_sync,
    parallel_run_sync,
    structured_run_sync,
    template_run_sync,
)
from u_llm_sdk.types import Provider
```

---

## quick_run

Execute a single prompt and get full result.

### Signature

```python
async def quick_run(
    prompt: str,
    *,
    provider: Provider = Provider.CLAUDE,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    config: Optional[LLMConfig] = None,
    intervention_hook: Optional[InterventionHook] = None,
) -> LLMResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | Required | Prompt to execute |
| `provider` | `Provider` | `CLAUDE` | Provider to use |
| `model` | `str` | `None` | Specific model (optional) |
| `timeout` | `float` | `None` | Timeout in seconds |
| `config` | `LLMConfig` | `None` | Full config (overrides provider/model) |
| `intervention_hook` | `InterventionHook` | `None` | Hook for RAG integration |

### Examples

```python
# Basic usage
result = await quick_run("What is Python?")
print(result.text)

# With specific provider
result = await quick_run("Explain async/await", provider=Provider.GEMINI)

# With timeout
result = await quick_run("Complex analysis...", timeout=60)

# With custom config
config = LLMConfig(provider=Provider.CLAUDE, tier=ModelTier.HIGH)
result = await quick_run("Deep analysis", config=config)
```

### Sync Version

```python
result = quick_run_sync("What is Python?", provider=Provider.CLAUDE)
```

---

## quick_text

Execute a prompt and return just the text response.

### Signature

```python
async def quick_text(
    prompt: str,
    *,
    provider: Provider = Provider.CLAUDE,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    config: Optional[LLMConfig] = None,
) -> str
```

### Examples

```python
# Get just the text
answer = await quick_text("What is 2+2?")
print(answer)  # "4"

# With different provider
answer = await quick_text("Hello!", provider=Provider.GEMINI)
```

### Sync Version

```python
answer = quick_text_sync("What is 2+2?")
```

---

## auto_run

Execute using auto-selected provider (first available).

### Signature

```python
async def auto_run(
    prompt: str,
    *,
    timeout: Optional[float] = None,
    config: Optional[LLMConfig] = None,
) -> LLMResult
```

### Examples

```python
# Auto-selects Claude > Codex > Gemini
result = await auto_run("Hello!")
print(f"Used: {result.provider}")
```

### Sync Version

```python
result = auto_run_sync("Hello!")
```

---

## parallel_run

Execute multiple prompts in parallel.

### Signature

```python
async def parallel_run(
    prompts: list[str],
    *,
    provider: Provider = Provider.CLAUDE,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    config: Optional[LLMConfig] = None,
    intervention_hook: Optional[InterventionHook] = None,
) -> list[LLMResult]
```

### Examples

```python
questions = [
    "What is Python?",
    "What is JavaScript?",
    "What is Rust?",
]

results = await parallel_run(questions)

for q, r in zip(questions, results):
    print(f"Q: {q}")
    print(f"A: {r.text[:100]}...")
```

### Sync Version

```python
results = parallel_run_sync(["Q1?", "Q2?", "Q3?"])
```

---

## multi_provider_run

Execute same prompt across multiple providers for comparison.

### Signature

```python
async def multi_provider_run(
    prompt: str,
    *,
    providers: list[Provider] = None,
    timeout: Optional[float] = None,
) -> dict[Provider, LLMResult]
```

### Examples

```python
results = await multi_provider_run(
    "What is the meaning of life?",
    providers=[Provider.CLAUDE, Provider.GEMINI, Provider.CODEX],
)

for provider, result in results.items():
    print(f"{provider.value}: {result.text[:100]}...")
```

---

## template_run

Execute with a predefined template persona.

### Signature

```python
async def template_run(
    prompt: str,
    template: SessionTemplate,
    *,
    provider: Provider = Provider.CLAUDE,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    config: Optional[LLMConfig] = None,
    intervention_hook: Optional[InterventionHook] = None,
) -> LLMResult
```

### Available Templates

```python
from u_llm_sdk.session import SessionTemplate

SessionTemplate.CODE_REVIEWER          # Code review expert
SessionTemplate.SECURITY_ANALYST       # Security analysis
SessionTemplate.PYTHON_EXPERT          # Python specialist
SessionTemplate.TYPESCRIPT_EXPERT      # TypeScript specialist
SessionTemplate.API_DESIGNER           # API design expert
SessionTemplate.TEST_ENGINEER          # Testing expert
SessionTemplate.PERFORMANCE_ANALYST    # Performance optimization
SessionTemplate.DOCUMENTATION_WRITER   # Documentation expert
SessionTemplate.CRPG_STORYTELLER       # Creative storytelling
SessionTemplate.CUSTOM_TELLER          # Custom narratives
```

### Examples

```python
from u_llm_sdk.session import SessionTemplate

# Security review
result = await template_run(
    "Review this code: def login(u, p): return db.check(u, p)",
    SessionTemplate.SECURITY_ANALYST,
)

# Code review
result = await template_run(
    "Review my implementation of a binary search tree",
    SessionTemplate.CODE_REVIEWER,
    provider=Provider.CLAUDE,
)
```

### Sync Version

```python
result = template_run_sync(
    "Review this code",
    SessionTemplate.CODE_REVIEWER,
)
```

---

## structured_run

Execute and parse response into structured format.

### Signature

```python
async def structured_run(
    prompt: str,
    schema: DomainSchema,
    *,
    provider: Provider = Provider.CLAUDE,
    config: Optional[LLMConfig] = None,
) -> tuple[LLMResult, Any]
```

### Examples

```python
from u_llm_sdk.types.schemas import BrainstormSchema

result, parsed = await structured_run(
    "Brainstorm ideas for a new feature",
    BrainstormSchema(),
)
```

---

## Comparison: When to Use What

| Function | Use Case |
|----------|----------|
| `quick_run` | Single prompt, need full result |
| `quick_text` | Single prompt, only need text |
| `auto_run` | Don't care which provider |
| `parallel_run` | Multiple independent prompts |
| `multi_provider_run` | Compare provider responses |
| `template_run` | Need specialized persona |
| `structured_run` | Need structured output |
