# Providers Reference

Provider-specific implementations for Claude, Codex, and Gemini CLIs.

## Provider Enum

```python
from u_llm_sdk.types import Provider

Provider.CLAUDE   # Anthropic Claude CLI
Provider.CODEX    # OpenAI Codex CLI
Provider.GEMINI   # Google Gemini CLI
```

---

## BaseProvider

Abstract base class for all providers.

### Key Methods

```python
class BaseProvider(ABC):
    PROVIDER: Provider       # Provider enum value
    CLI_NAME: str           # CLI executable name

    # Core methods
    async def run(prompt, *, session_id=None, timeout=None) -> LLMResult
    async def stream(prompt, *, session_id=None, timeout=None) -> AsyncIterator[dict]
    def resume(session_id: str) -> Self
    async def parallel_run(prompts, *, timeout=None) -> list[LLMResult]

    # Class methods
    @classmethod
    def is_available() -> bool  # Check if CLI is installed
    @classmethod
    def get_cli_path() -> Optional[str]  # Get CLI path
```

### Checking Availability

```python
from u_llm_sdk.llm.providers import ClaudeProvider, CodexProvider, GeminiProvider

if ClaudeProvider.is_available():
    print("Claude CLI is installed")

if GeminiProvider.is_available():
    print(f"Gemini CLI at: {GeminiProvider.get_cli_path()}")
```

---

## ClaudeProvider

### Import

```python
from u_llm_sdk.llm.providers import ClaudeProvider
from u_llm_sdk.config import LLMConfig
from u_llm_sdk.types import Provider, ReasoningLevel
```

### Features

| Feature | Support | Notes |
|---------|---------|-------|
| Extended Thinking | ✅ | `ReasoningLevel.XHIGH` triggers ultrathink |
| System Prompt | ✅ | Native `--system-prompt` flag |
| MCP Config | ✅ | Via `provider_options["mcp_config"]` |
| Web Search | ✅ | WebSearch, WebFetch tools |
| Session Resume | ✅ | `--resume` flag |

### Extended Thinking (Ultrathink)

**IMPORTANT**: Only `ReasoningLevel.XHIGH` triggers extended thinking.

```python
config = LLMConfig(
    provider=Provider.CLAUDE,
    reasoning_level=ReasoningLevel.XHIGH,  # Triggers "ultrathink: " prefix
)

# Or use tier which implies XHIGH
config = LLMConfig(
    provider=Provider.CLAUDE,
    tier=ModelTier.HIGH,
    reasoning_level=ReasoningLevel.XHIGH,
)
```

### Provider Options

```python
config = LLMConfig(
    provider=Provider.CLAUDE,
    provider_options={
        "mcp_config": "/path/to/mcp.json",    # MCP server config
        "max_turns": 10,                       # Max conversation turns
        "setting_sources": ["user", "project"],  # Settings sources
    },
)
```

---

## CodexProvider

### Import

```python
from u_llm_sdk.llm.providers import CodexProvider
from u_llm_sdk.config import LLMConfig
from u_llm_sdk.types import Provider, AutoApproval, SandboxMode
```

### Features

| Feature | Support | Notes |
|---------|---------|-------|
| Image Inputs | ✅ | Multiple `-i` flags |
| Web Search | ✅ | `--search` flag |
| Output Schema | ✅ | Structured JSON output |
| Sandbox | ✅ | Multiple modes |
| Full Auto | ✅ | `--full-auto` flag |

### Approval Mode Mapping

| AutoApproval | Codex Flag |
|--------------|------------|
| `NONE` | `-a untrusted` |
| `EDITS_ONLY` | `-a on-failure` |
| `FULL` | `-a never` |

### Sandbox Mode Mapping

| SandboxMode | Codex Flag |
|-------------|------------|
| `READ_ONLY` | `-s read-only` |
| `WORKSPACE_WRITE` | `-s workspace-write` |
| `FULL_ACCESS` | `-s danger-full-access` |

### Provider Options

```python
config = LLMConfig(
    provider=Provider.CODEX,
    auto_approval=AutoApproval.FULL,
    sandbox=SandboxMode.WORKSPACE_WRITE,
    cwd="/workspace",
    provider_options={
        # Image inputs
        "images": ["/path/to/img1.png", "/path/to/img2.jpg"],

        # Web search
        "search": True,

        # Full auto mode
        "full_auto": True,

        # Output schema for structured output
        "output_schema": "/path/to/schema.json",

        # Additional directories
        "add_dirs": ["/path/to/dir1", "/path/to/dir2"],

        # Feature flags
        "features_enable": ["feature1"],
        "features_disable": ["feature2"],

        # Config overrides
        "config_overrides": {
            "timeout": "120",
            "max_tokens": "8000",
        },

        # Skip git repo check
        "skip_git_repo_check": True,
    },
)
```

---

## GeminiProvider

### Import

```python
from u_llm_sdk.llm.providers import GeminiProvider
from u_llm_sdk.config import LLMConfig
from u_llm_sdk.types import Provider, AutoApproval
```

### Features

| Feature | Support | Notes |
|---------|---------|-------|
| Temperature | ✅ | 0.0-1.0 via provider_options |
| Top-P/Top-K | ✅ | Sampling parameters |
| Extensions | ✅ | Multiple `-e` flags |
| Tool Control | ✅ | Allowed tools list |
| MCP Filter | ✅ | Filter MCP servers |
| GoogleSearch | ✅ | Built-in when available |

### Approval Mode Mapping

| AutoApproval | Gemini Flag |
|--------------|-------------|
| `NONE` | `--approval-mode default` |
| `EDITS_ONLY` | `--approval-mode auto_edit` |
| `FULL` | `-y` (yolo mode) |

### Provider Options

```python
config = LLMConfig(
    provider=Provider.GEMINI,
    auto_approval=AutoApproval.EDITS_ONLY,
    sandbox=SandboxMode.WORKSPACE_WRITE,  # Uses -s flag (boolean)
    provider_options={
        # Sampling parameters
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,

        # Tool control (space-separated in CLI)
        "allowed_tools": ["edit", "bash", "read"],

        # Extensions
        "extensions": ["code-review", "analysis"],

        # Additional directories
        "include_directories": ["/shared/libs", "/config"],

        # MCP server filter
        "allowed_mcp_server_names": ["filesystem", "github"],
    },
)
```

### Session Resume

```python
# Resume latest session
config = LLMConfig(provider=Provider.GEMINI)
llm = GeminiProvider(config)
llm.resume("latest")  # --resume latest

# Resume specific session
llm.resume("session-abc123")  # --resume session-abc123
```

---

## Provider Comparison

| Feature | Claude | Codex | Gemini |
|---------|--------|-------|--------|
| Extended Thinking | ✅ XHIGH only | ❌ | ❌ |
| System Prompt | ✅ Native | ⚠️ Prepended | ⚠️ Prepended |
| Image Input | ❌ | ✅ Multiple | ❌ |
| Temperature | ❌ | ❌ | ✅ |
| Web Search | ✅ | ✅ | ✅ |
| MCP Config | ✅ | ❌ | ✅ Filter |
| Sandbox Modes | ❌ | ✅ | ⚠️ Boolean |
| Output Schema | ❌ | ✅ | ❌ |

---

## Checking Available Providers

```python
from u_llm_sdk.llm.client import available_providers
from u_llm_sdk.types import Provider

# Get list of available providers
available = available_providers()
print(available)  # [Provider.CLAUDE, Provider.GEMINI]

# Check specific provider
if Provider.CLAUDE in available:
    print("Claude is available")
```
