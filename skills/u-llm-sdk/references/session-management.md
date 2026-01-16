# Session Management Reference

File-based session management for all providers.

## Import

```python
from u_llm_sdk.session import (
    # Factory
    get_session_manager,
    inject_system_prompt,

    # Base class
    BaseSessionManager,

    # Provider implementations
    ClaudeSessionManager,
    CodexSessionManager,
    GeminiSessionManager,

    # Message handling
    SessionMessage,
    resolve_prompt,

    # Templates
    SessionTemplate,
    get_template_prompt,
    list_templates,
    get_template_info,
    create_custom_template,
)
```

---

## get_session_manager

Factory function to get provider-specific session manager.

### Signature

```python
def get_session_manager(
    provider: Provider,
    project_path: str,
) -> BaseSessionManager
```

### Usage

```python
from u_llm_sdk.session import get_session_manager
from u_llm_sdk.types import Provider

# Get Claude session manager
manager = get_session_manager(Provider.CLAUDE, "/my/project")

# Get Gemini session manager
manager = get_session_manager(Provider.GEMINI, "/my/project")
```

---

## BaseSessionManager

Abstract base class for session management.

### Methods

#### `create_session(session_id=None) -> str`

Create a new session.

```python
session_id = manager.create_session()
print(f"Created: {session_id}")

# Or with custom ID
session_id = manager.create_session("my-custom-id")
```

#### `create_from_system_prompt(system_prompt, assistant_acknowledgment=None) -> str`

Create session with initial system prompt.

```python
session_id = manager.create_from_system_prompt(
    "You are a security expert.",
    assistant_acknowledgment="Ready to analyze.",
)
```

#### `create_from_template(template: SessionTemplate) -> str`

Create session from a predefined template.

```python
from u_llm_sdk.session import SessionTemplate

session_id = manager.create_from_template(SessionTemplate.SECURITY_ANALYST)
```

#### `get_session(session_id) -> Optional[dict]`

Get session data.

```python
session = manager.get_session(session_id)
if session:
    print(session["messages"])
```

#### `list_sessions() -> list[str]`

List all session IDs.

```python
sessions = manager.list_sessions()
for sid in sessions:
    print(sid)
```

---

## Provider-Specific Storage

### Claude Session Storage

```
~/.claude/projects/<path-key>/
├── session_<uuid>.jsonl    # JSONL format
└── ...
```

### Codex Session Storage

```
~/.codex/sessions/YYYY/MM/DD/
├── <session-id>/
│   └── session_meta.instructions  # Stores system prompt
└── ...
```

### Gemini Session Storage

```
~/.gemini/tmp/<project-hash>/chats/
├── <session-id>.json       # JSON with messages array
└── ...
```

---

## inject_system_prompt

Provider-appropriate system prompt injection.

### Signature

```python
def inject_system_prompt(
    provider: Provider,
    prompt: str,
    system_prompt: str,
) -> tuple[str, dict]
```

### Behavior by Provider

| Provider | Method |
|----------|--------|
| Claude | Uses native `--system-prompt` flag |
| Codex | Prepends to user prompt |
| Gemini | Prepends to user prompt |

### Usage

```python
from u_llm_sdk.session import inject_system_prompt
from u_llm_sdk.types import Provider

# Claude - uses native flag
prompt, config = inject_system_prompt(
    Provider.CLAUDE,
    "Analyze this code",
    "You are a security expert.",
)
# prompt == "Analyze this code"
# config == {"system_prompt": "You are a security expert."}

# Codex/Gemini - prepends
prompt, config = inject_system_prompt(
    Provider.CODEX,
    "Analyze this code",
    "You are a security expert.",
)
# prompt == "You are a security expert.\n\nAnalyze this code"
# config == {}
```

---

## SessionTemplate

Predefined persona templates.

### Available Templates

```python
from u_llm_sdk.session import SessionTemplate

SessionTemplate.CODE_REVIEWER         # Code review specialist
SessionTemplate.SECURITY_ANALYST      # Security vulnerability expert
SessionTemplate.PYTHON_EXPERT         # Python specialist
SessionTemplate.TYPESCRIPT_EXPERT     # TypeScript/JavaScript specialist
SessionTemplate.API_DESIGNER          # API design expert
SessionTemplate.TEST_ENGINEER         # Testing and QA expert
SessionTemplate.PERFORMANCE_ANALYST   # Performance optimization
SessionTemplate.DOCUMENTATION_WRITER  # Documentation specialist
SessionTemplate.CRPG_STORYTELLER      # Creative CRPG storytelling
SessionTemplate.CUSTOM_TELLER         # Custom narrative creation
```

### Template Functions

#### `get_template_prompt(template) -> str`

Get the full system prompt for a template.

```python
from u_llm_sdk.session import get_template_prompt, SessionTemplate

prompt = get_template_prompt(SessionTemplate.SECURITY_ANALYST)
print(prompt)  # Full security analyst persona
```

#### `list_templates() -> list[SessionTemplate]`

List all available templates.

```python
from u_llm_sdk.session import list_templates

for template in list_templates():
    print(template.value)
```

#### `get_template_info(template) -> dict`

Get metadata about a template.

```python
from u_llm_sdk.session import get_template_info, SessionTemplate

info = get_template_info(SessionTemplate.CODE_REVIEWER)
print(info["name"])
print(info["description"])
print(info["focus_areas"])
```

#### `create_custom_template(name, prompt, **kwargs) -> SessionTemplate`

Create a custom template.

```python
from u_llm_sdk.session import create_custom_template

custom = create_custom_template(
    name="my_expert",
    prompt="You are an expert in my domain...",
    description="Custom domain expert",
)
```

---

## SessionMessage

Represents a single message in a session.

### Fields

```python
@dataclass
class SessionMessage:
    role: str           # "user", "assistant", or "system"
    content: str        # Message content
    timestamp: Optional[datetime]
    metadata: Optional[dict]
```

### Usage

```python
from u_llm_sdk.session import SessionMessage

msg = SessionMessage(
    role="user",
    content="Analyze this code",
)
```

---

## resolve_prompt

Resolve a prompt with optional template context.

### Signature

```python
def resolve_prompt(
    prompt: str,
    template: Optional[SessionTemplate] = None,
    context: Optional[dict] = None,
) -> str
```

### Usage

```python
from u_llm_sdk.session import resolve_prompt, SessionTemplate

# Simple prompt
resolved = resolve_prompt("Review this code")

# With template
resolved = resolve_prompt(
    "Review this code",
    template=SessionTemplate.CODE_REVIEWER,
)

# With context variables
resolved = resolve_prompt(
    "Review {filename}",
    context={"filename": "auth.py"},
)
```

---

## Complete Example

```python
from u_llm_sdk import LLM, Provider
from u_llm_sdk.session import (
    get_session_manager,
    SessionTemplate,
    get_template_prompt,
)

# Create session with template
manager = get_session_manager(Provider.CLAUDE, "/my/project")
session_id = manager.create_from_template(SessionTemplate.SECURITY_ANALYST)

# Use session
async with LLM(provider=Provider.CLAUDE) as llm:
    # First interaction
    result = await llm.run("Review auth.py", session_id=session_id)
    print(result.text)

    # Continue in same session
    result = await llm.run("What about SQL injection?", session_id=session_id)
    print(result.text)

# Or use CLI directly
# claude --resume {session_id}
```

---

## CLI Resume Commands

| Provider | Command |
|----------|---------|
| Claude | `claude --resume <session_id>` |
| Codex | `codex resume <session_id>` |
| Gemini | `gemini --resume <session_id>` |
