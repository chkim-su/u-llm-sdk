# Core Client Reference

## LLM (Async Client)

The main async client for LLM interactions.

### Constructor

```python
from u_llm_sdk import LLM, Provider
from u_llm_sdk.config import LLMConfig

# Basic initialization
llm = LLM(provider=Provider.CLAUDE)

# With full config
config = LLMConfig(
    provider=Provider.CLAUDE,
    tier=ModelTier.HIGH,
    auto_approval=AutoApproval.FULL,
    timeout=120,
)
llm = LLM(config=config)

# With intervention hook (RAG integration)
llm = LLM(config=config, intervention_hook=my_hook, auto_rag=True)
```

### Context Manager Usage

```python
async with LLM(provider=Provider.CLAUDE) as llm:
    result = await llm.run("Hello!")
    print(result.text)
```

### Methods

#### `run(prompt, *, session_id=None, timeout=None) -> LLMResult`

Execute a single prompt.

```python
result = await llm.run("What is 2+2?")
print(result.text)  # "4"
print(result.success)  # True
print(result.result_type)  # ResultType.TEXT
```

#### `stream(prompt, *, session_id=None, timeout=None) -> AsyncIterator[dict]`

Stream execution results.

```python
async for chunk in llm.stream("Tell me a story"):
    if chunk.get("type") == "text":
        print(chunk["content"], end="", flush=True)
```

#### `parallel_run(prompts, *, timeout=None) -> list[LLMResult]`

Execute multiple prompts in parallel.

```python
results = await llm.parallel_run(["Q1?", "Q2?", "Q3?"])
for r in results:
    print(r.text)
```

#### `resume(session_id) -> LLM`

Set session ID for conversation continuity.

```python
llm = LLM(provider=Provider.CLAUDE).resume("session-123")
async with llm as client:
    result = await client.run("Continue our conversation")
```

### Class Methods

#### `LLM.auto(config=None, intervention_hook=None, auto_rag=True) -> LLM`

Auto-select first available provider (Claude > Codex > Gemini).

```python
async with LLM.auto() as llm:
    result = await llm.run("Hello!")
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `session_id` | `Optional[str]` | Current session ID |
| `provider` | `Optional[BaseProvider]` | Underlying provider instance |
| `config` | `LLMConfig` | Current configuration |

---

## LLMSync (Sync Client)

Synchronous wrapper for LLM operations.

### Constructor

```python
from u_llm_sdk import LLMSync, Provider

llm = LLMSync(provider=Provider.CLAUDE)
```

### Context Manager Usage

```python
with LLMSync(provider=Provider.GEMINI) as llm:
    result = llm.run("Hello!")
    print(result.text)
```

### Methods

#### `run(prompt, *, session_id=None, timeout=None) -> LLMResult`

Execute a prompt synchronously.

```python
llm = LLMSync(provider=Provider.CLAUDE)
result = llm.run("What is 2+2?")
print(result.text)
```

#### `parallel_run(prompts, *, timeout=None) -> list[LLMResult]`

Execute multiple prompts in parallel (runs async internally).

```python
results = llm.parallel_run(["Q1?", "Q2?", "Q3?"])
```

#### `resume(session_id) -> LLMSync`

Set session ID for continuation.

```python
llm = LLMSync(provider=Provider.CLAUDE).resume("session-123")
```

### Class Methods

#### `LLMSync.auto() -> LLMSync`

Auto-select first available provider.

```python
llm = LLMSync.auto()
result = llm.run("Hello!")
```

---

## LLMResult

Unified result structure from any provider.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether execution succeeded |
| `result_type` | `ResultType` | Type of result (TEXT, CODE, FILE_EDIT, etc.) |
| `summary` | `str` | Brief summary of the result |
| `text` | `str` | Main text output |
| `files_modified` | `list[FileChange]` | List of file changes |
| `commands_run` | `list[CommandRun]` | List of commands executed |
| `error_message` | `Optional[str]` | Error message if failed |
| `duration_ms` | `int` | Execution duration in milliseconds |
| `session_id` | `Optional[str]` | Session ID for continuation |
| `token_usage` | `Optional[TokenUsage]` | Token usage statistics |

### Methods

```python
result.has_text()  # True if text is non-empty
result.has_file_changes()  # True if files were modified
result.has_commands()  # True if commands were run
result.to_dict()  # Convert to dictionary
LLMResult.from_dict(data)  # Create from dictionary
```

### ResultType Enum

```python
from u_llm_sdk.types import ResultType

ResultType.TEXT        # Plain text response
ResultType.CODE        # Code generation
ResultType.FILE_EDIT   # File modifications
ResultType.COMMAND     # Command execution
ResultType.ERROR       # Error occurred
ResultType.MIXED       # Multiple result types
```
