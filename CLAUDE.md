# U-llm-sdk — Claude Rules

이 패키지는 **LLM 실행 레이어**로, 다양한 LLM 프로바이더를 통합하고 MV-rag와 HTTP로 통신한다.

## Package Structure

```
U-llm-sdk/
├── src/u_llm_sdk/
│   ├── __init__.py            # LLM, LLMSync exports
│   ├── config.py              # LLMConfig, presets
│   ├── types/                 # Shared types (merged from llm-types)
│   │   ├── __init__.py        # Provider, LLMResult, etc.
│   │   ├── config.py          # Provider, ModelTier, AutoApproval
│   │   ├── models.py          # LLMResult, TokenUsage, FileChange
│   │   ├── hooks.py           # PreActionContext, PostActionFeedback
│   │   ├── exceptions.py      # UnifiedLLMError, ProviderNotFoundError
│   │   ├── features.py        # Feature validation
│   │   ├── evidence.py        # Evidence types
│   │   ├── orchestration/     # Multi-LLM orchestration types
│   │   ├── schemas/           # Domain schema types
│   │   └── chronicle/         # Chronicle record types
│   ├── core/
│   │   ├── discovery.py       # CLI path discovery
│   │   └── utils.py           # quick_run, quick_text, template_run
│   ├── llm/
│   │   ├── client.py          # LLM, LLMSync classes
│   │   ├── providers/
│   │   │   ├── base.py        # BaseProvider ABC
│   │   │   ├── claude.py      # ClaudeProvider
│   │   │   ├── codex.py       # CodexProvider
│   │   │   ├── gemini.py      # GeminiProvider
│   │   │   └── hooks.py       # InterventionHook protocol
│   │   └── orchestration/     # Full orchestration framework
│   │       ├── contracts.py   # WorkOrder, Evidence, ROLES
│   │       ├── types.py       # ExecutionState, WorkOrderState
│   │       ├── state_machine.py
│   │       ├── merge_executor.py
│   │       ├── verifier.py
│   │       ├── checkpoint.py
│   │       ├── lock_manager.py
│   │       ├── file_scope.py
│   │       ├── git_integration.py
│   │       ├── context_builder.py
│   │       ├── delta_compliance.py
│   │       ├── patch_submission.py
│   │       ├── evidence_executor.py
│   │       └── agent_executor.py
│   ├── session/               # File-based session management (all providers)
│   │   ├── __init__.py        # get_session_manager, inject_system_prompt
│   │   ├── base.py            # BaseSessionManager ABC
│   │   ├── message.py         # SessionMessage, resolve_prompt
│   │   ├── claude.py          # ClaudeSessionManager
│   │   ├── codex.py           # CodexSessionManager
│   │   ├── gemini.py          # GeminiSessionManager
│   │   ├── templates.py       # SessionTemplate enum
│   │   └── templates/         # Template .md files
│   ├── advanced/              # Advanced orchestration client
│   │   ├── __init__.py        # UnifiedAdvanced, AdvancedConfig
│   │   ├── config.py          # AdvancedConfig, AgentDefinition
│   │   └── client.py          # UnifiedAdvanced, UnifiedAdvancedSync
│   ├── multi_llm/             # Multi-LLM orchestration
│   │   ├── __init__.py        # ClarityGate, ConsensusLoop, Orchestrators, Hybrid
│   │   ├── clarity.py         # ClarityGate - task clarity assessment
│   │   ├── escalation.py      # EscalationProtocol - upward queries
│   │   ├── consensus.py       # ConsensusLoop - 3-round voting
│   │   ├── brainstorm.py      # BrainstormModule - full sessions
│   │   ├── orchestrator.py    # GeminiOrchestrator, ClaudeOrchestrator, etc.
│   │   ├── rag_integration.py # MultiLLMRAGHook - pattern learning
│   │   ├── migration.py       # HybridOrchestrator, StateRecoveryManager
│   │   ├── performance.py     # ParallelExecutor, LatencyTracker, MemoryManager
│   │   ├── monitoring.py      # OrchestrationLogger, DebugMode, MetricExporter
│   │   └── session.py         # SessionStore, SessionManager
│   └── rag_client/            # MV-rag HTTP client
│       ├── config.py
│       ├── cache.py
│       └── client.py
```

## Absolute Rules

1. **MV-rag 직접 Import 금지**
   - MV-rag 코드를 직접 import하지 않음
   - HTTP API를 통해서만 통신 (RAGClient 사용)
   - 이유: 독립 배포, 순환 의존성 방지

2. **타입은 u_llm_sdk.types에서 import**
   - 공유 타입은 내장된 types 서브패키지에서 import
   - `from u_llm_sdk.types import Provider, LLMResult, PreActionContext`
   - 또는 top-level에서 re-export: `from u_llm_sdk import Provider, LLMResult`

3. **Fail-Open 필수**
   - MV-rag 연결 실패 시 LLM 실행은 계속되어야 함
   - RAGClient는 에러 시 None 반환 (예외 발생 금지)
   - `fail_open=True`가 기본값

4. **스키마 호환성 확인**
   - RAGClient 초기화 시 MV-rag 스키마 버전 확인
   - 호환되지 않으면 경고 로그 (crash 금지)

## Core Components

### LLM / LLMSync

메인 클라이언트 인터페이스:

```python
from u_llm_sdk import LLM, LLMSync, Provider

# Async
async with LLM(provider=Provider.CLAUDE) as llm:
    result = await llm.run("prompt")
    async for chunk in llm.stream("prompt"):
        print(chunk)

# Sync
llm = LLMSync(provider=Provider.CLAUDE)
result = llm.run("prompt")
```

### Quick Utilities

```python
from u_llm_sdk.core.utils import quick_run, quick_text, parallel_run

# One-shot execution
result = await quick_run("prompt", provider=Provider.CLAUDE)
text = await quick_text("prompt", provider=Provider.GEMINI)

# Parallel execution
results = await parallel_run(["p1", "p2", "p3"], provider=Provider.CLAUDE)
```

### LLMConfig

```python
from u_llm_sdk.config import LLMConfig, SAFE_CONFIG, AUTO_CONFIG

config = LLMConfig(
    provider=Provider.CLAUDE,
    tier=ModelTier.HIGH,
    auto_approval=AutoApproval.FULL,
    sandbox_mode=SandboxMode.WORKSPACE_WRITE,
    timeout_seconds=120,
    cwd="/workspace",
)

# Presets
SAFE_CONFIG   # Read-only, no approval
AUTO_CONFIG   # Full approval, workspace write

# Web search enabled
research_config = LLMConfig(
    provider=Provider.CLAUDE,
    web_search=True,  # Enables WebSearch, WebFetch tools
)

# Or using builder pattern
config = LLMConfig(provider=Provider.GEMINI).with_web_search()

# Feature validation (LLM-friendly)
results = config.validate_for_provider()
for r in results:
    print(r.to_dict())  # LLM can parse this JSON
```

### Web Search

Provider-specific web search tools (verified 2026-01-08):

| Provider | Tools | Notes |
|----------|-------|-------|
| **Claude** | WebSearch, WebFetch | Requires `web_search=True` to add to allowed_tools |
| **Gemini** | GoogleSearch | Built-in, `web_search=True` ensures availability |
| **Codex** | web_search (native) | `--search` flag in exec mode (enables native web_search tool) |

```python
from u_llm_sdk import LLM, LLMConfig
from u_llm_sdk.types import Provider

# Claude with web search
config = LLMConfig(
    provider=Provider.CLAUDE,
    web_search=True,
)

async with LLM(config=config) as llm:
    result = await llm.run("Search for the latest Python 3.13 release notes")
    # Claude will use WebSearch tool automatically

# Gemini with GoogleSearch
config = LLMConfig(
    provider=Provider.GEMINI,
    web_search=True,
)

async with LLM(config=config) as llm:
    result = await llm.run("Find current weather in Seoul")
```

### Feature Validation

Provider별 CLI 기능 지원 검증:

```python
from u_llm_sdk.config import LLMConfig
from u_llm_sdk.types import Provider, AutoApproval

# 지원되지 않는 기능 사용 시 warning 로그 자동 출력
config = LLMConfig(
    provider=Provider.CODEX,
    auto_approval=AutoApproval.EDITS_ONLY,  # Codex에서 미지원!
)

# 검증 결과 확인
results = config.validate_for_provider()
for r in results:
    print(r.to_dict())
    # {
    #   "feature": "auto_approval_edits_only",
    #   "provider": "codex",
    #   "severity": "warning",
    #   "supported": false,
    #   "suggestion": "Use claude or gemini instead...",
    #   ...
    # }
```

Provider 초기화 시 자동 검증:

```python
# validate_features=True (default) → 초기화 시 자동 검증
provider = CodexProvider(config)
print(provider.validation_results)  # 검증 결과 접근

# validate_features=False → 검증 비활성화 (테스트용)
provider = CodexProvider(config, validate_features=False)
```

## Provider Architecture

### BaseProvider

모든 프로바이더의 기본 클래스:

```python
class BaseProvider(ABC):
    PROVIDER: Provider
    CLI_NAME: str

    async def run(prompt: str, ...) -> LLMResult
    async def stream(prompt: str, ...) -> AsyncIterator[dict]
    def resume(session_id: str) -> Self

    @classmethod
    def is_available() -> bool
    @classmethod
    def get_cli_path() -> Optional[str]
```

### Provider 구현체

| Provider | CLI | 특징 |
|----------|-----|------|
| ClaudeProvider | `claude` | Ultrathink only (extended thinking), MCP config |
| CodexProvider | `codex` | Approval/sandbox modes, image inputs, web search, output schema |
| GeminiProvider | `gemini` | Temperature (settings-based), allowed_tools, extensions, MCP filter |

### Claude Extended Thinking (Verified 2026-01-08)

**IMPORTANT**: Only `ultrathink` triggers extended thinking in Claude CLI v2.0.0+

| ReasoningLevel | Trigger | Thinking Tokens |
|----------------|---------|-----------------|
| NONE | (none) | No |
| LOW | (none) | No |
| MEDIUM | (deprecated) | No - "think" does NOT trigger thinking |
| HIGH | (deprecated) | No - "think hard" does NOT trigger thinking |
| XHIGH | `ultrathink: ` | Yes - allocates up to 31,999 tokens |

The `ReasoningLevel.MEDIUM` and `ReasoningLevel.HIGH` values are deprecated for Claude.
Use `ReasoningLevel.XHIGH` for extended thinking capabilities.

#### CodexProvider CLI Features

CodexProvider now supports extensive CLI features via `provider_options`:

```python
from u_llm_sdk import LLM, LLMConfig
from u_llm_sdk.types import Provider, AutoApproval, SandboxMode

config = LLMConfig(
    provider=Provider.CODEX,
    model="gpt-5.2",
    auto_approval=AutoApproval.FULL,      # -a never
    sandbox=SandboxMode.WORKSPACE_WRITE,  # -s workspace-write
    cwd="/workspace",                      # -C /workspace
    provider_options={
        # Image inputs (can be multiple)
        "images": ["/path/to/image1.png", "/path/to/image2.jpg"],  # -i flags

        # Web search
        "search": True,                    # --search

        # Full auto mode (alternative to auto_approval)
        "full_auto": True,                 # --full-auto

        # Output schema (structured output)
        "output_schema": "/path/to/schema.json",  # --output-schema

        # Additional directories (beyond cwd)
        "add_dirs": ["/path/to/dir1", "/path/to/dir2"],  # --add-dir flags

        # Feature flags
        "features_enable": ["feature1", "feature2"],   # --enable flags
        "features_disable": ["feature3", "feature4"],  # --disable flags

        # Config overrides (key=value pairs)
        "config_overrides": {              # -c key=value
            "timeout": "120",
            "max_tokens": "8000",
        },

        # Legacy options
        "skip_git_repo_check": True,       # --skip-git-repo-check
    },
)

async with LLM(config=config) as llm:
    result = await llm.run("Analyze this image and code")
```

**Approval Mode Mapping:**
- `AutoApproval.NONE` → `-a untrusted` (interactive approval)
- `AutoApproval.EDITS_ONLY` → `-a on-failure` (approve on test failure)
- `AutoApproval.FULL` → `-a never` (no approval needed)

**Sandbox Mode Mapping:**
- `SandboxMode.READ_ONLY` → `-s read-only`
- `SandboxMode.WORKSPACE_WRITE` → `-s workspace-write`
- `SandboxMode.FULL_ACCESS` → `-s danger-full-access`

**Output Format:**
- CodexProvider always uses `--json` flag for JSONL output
- Output is parsed line-by-line as JSON events
- Both `run()` and `stream()` handle JSONL format correctly

#### GeminiProvider CLI Features

GeminiProvider now supports advanced CLI features via `provider_options`:

```python
from u_llm_sdk import LLM, LLMConfig
from u_llm_sdk.types import Provider, AutoApproval, SandboxMode

config = LLMConfig(
    provider=Provider.GEMINI,
    model="gemini-3-pro-preview",
    auto_approval=AutoApproval.EDITS_ONLY,  # --approval-mode auto_edit
    sandbox=SandboxMode.WORKSPACE_WRITE,     # -s (boolean flag)
    provider_options={
        # Sampling parameters
        "temperature": 0.7,                  # --temperature 0.7
        "top_p": 0.9,                        # --top-p 0.9
        "top_k": 40,                         # --top-k 40

        # Tool control (space-separated)
        "allowed_tools": ["edit", "bash", "read"],  # --allowed-tools edit bash read

        # Extensions (multiple -e flags)
        "extensions": ["code-review", "analysis"],  # -e code-review -e analysis

        # Additional directories
        "include_directories": ["/shared/libs", "/config"],  # --include-directories

        # MCP server filter
        "allowed_mcp_server_names": ["filesystem", "github"],  # --allowed-mcp-server-names
    },
)

async with LLM(config=config) as llm:
    # Regular execution
    result = await llm.run("Analyze this codebase")

    # Resume latest session
    result = await llm.run("", session_id="latest")
```

**Approval Mode Mapping:**
- `AutoApproval.NONE` → `--approval-mode default`
- `AutoApproval.EDITS_ONLY` → `--approval-mode auto_edit`
- `AutoApproval.FULL` → `-y` (yolo mode)

**Sandbox Mode:**
- Gemini CLI uses `-s` as a **boolean flag** (not mode-based like Codex)
- Any non-NONE `SandboxMode` enables the `-s` flag

**Session Resume:**
- `session_id="latest"` → `--resume latest` (resume most recent session)
- `session_id="<id>"` → `--resume <id>` (resume specific session)

**Output Format:**
- GeminiProvider always uses `-o stream-json` for parsing
- Output is parsed line-by-line as JSON events
- Both `run()` and `stream()` handle stream-json format correctly

**Non-Interactive Mode:**
- Gemini uses positional argument for prompt (NOT `-p` flag)
- Example: `gemini "prompt" -o stream-json` (not `gemini -p "prompt"`)

## Orchestration Framework

### Contracts

```python
from u_llm_sdk.llm.orchestration import WorkOrder, Evidence, ReviewReport

# WorkOrder: 작업 계약
wo = WorkOrder(
    id="WO-001",
    objective="Implement feature",
    file_set=["src/**/*.py"],       # 수정 가능한 파일
    create_files=["src/new.py"],    # 생성할 파일
    constraints=["No deps"],
    resource_locks=["api:auth"],     # 리소스 잠금
    expected_delta=ExpectedDelta(    # 변경 제약
        max_files_modified=5,
        forbid_new_public_exports=True,
    ),
)

# Evidence: 작업 증거
evidence = Evidence(
    work_order_id="WO-001",
    branch="wo/WO-001",
    files_modified=["src/auth.py"],
    test_results={"pytest": True},
    typecheck_passed=True,
)
```

### State Machine

```python
from u_llm_sdk.llm.orchestration import StateMachineOrchestrator

orchestrator = StateMachineOrchestrator(
    max_parallel_editors=3,
    max_retries=2,
)

# 상태 전이:
# INIT → PLANNING → DISPATCHING → REVIEWING → MERGING → AGGREGATING → COMPLETED
ctx = await orchestrator.run(
    objective="Implement auth",
    cwd="/repo",
)
```

### MergeExecutor

통합 브랜치 패턴으로 전체 파이프라인 실행:

```python
from u_llm_sdk.llm.orchestration import MergeExecutor, MergeExecutorConfig

config = MergeExecutorConfig(
    base_branch="main",
    integration_branch="llm/feature",
    create_pr=True,
    require_tests=True,
    require_typecheck=True,
)

executor = MergeExecutor(config)
result = await executor.run("Add user auth", cwd="/repo")

# Pipeline:
# 1. Clarify: 작업 명확화
# 2. Plan: WorkOrder 분해
# 3. Dispatch: 병렬 실행 (worktree 격리)
# 4. Review: 증거 검토
# 5. Merge: 통합 브랜치에 머지
# 6. Validate: 최종 검증
# 7. Push/PR: 원격 푸시 또는 PR 생성
```

### Supporting Modules

| Module | 역할 |
|--------|------|
| `verifier.py` | LLM/Callback/Composite 검증기 |
| `checkpoint.py` | 감사 가능한 체크포인트 |
| `lock_manager.py` | 리소스 잠금 (In-memory, Redis) |
| `file_scope.py` | 파일셋 확장 및 충돌 검사 |
| `git_integration.py` | 브랜치/worktree/머지 관리 |
| `context_builder.py` | WorkOrder 컨텍스트 구성 |
| `delta_compliance.py` | AST 기반 변경 검증 |
| `patch_submission.py` | Patch 기반 변경 제출 |
| `evidence_executor.py` | 테스트/타입체크/린트 실행 |
| `agent_executor.py` | Role 기반 LLM 실행 |

## Multi-LLM Orchestration

Multi-provider 조율을 위한 컴포넌트:

```python
from u_llm_sdk.multi_llm import (
    # Core components
    ClarityGate,
    EscalationProtocol,
    ConsensusLoop,
    BrainstormModule,
    # Orchestrators
    OrchestratorFactory,
    GeminiOrchestrator,
    ClaudeOrchestrator,
    CodexOrchestrator,
)
```

### ClarityGate

Worker가 작업 수신 시 명확성 자가 평가:

```python
from u_llm_sdk.multi_llm import ClarityGate
from u_llm_sdk.types import Task, ClarityLevel

gate = ClarityGate(claude_provider)
task = Task(task_id="t1", objective="Build auth", context="FastAPI")

assessment = await gate.assess(task)
if assessment.level == ClarityLevel.CLEAR:
    # 자율 실행
elif assessment.recommendation == "escalate":
    # 오케스트레이터에게 질의
```

### EscalationProtocol

불명확시 상위 질의:

```python
from u_llm_sdk.multi_llm import EscalationProtocol
from u_llm_sdk.types import EscalationRequest

protocol = EscalationProtocol(gemini_orchestrator)
request = EscalationRequest(
    source_worker=Provider.CLAUDE,
    original_task=task,
    clarity_assessment=assessment,
    specific_questions=["What API version?"],
)

response = await protocol.escalate(request)
# response.refined_task, response.clarifications
```

### ConsensusLoop

3라운드 다수결 합의:

```python
from u_llm_sdk.multi_llm import ConsensusLoop
from u_llm_sdk.types import BrainstormConfig

providers = {
    Provider.GEMINI: gemini,
    Provider.CLAUDE: claude,
    Provider.CODEX: codex,
}

config = BrainstormConfig(
    max_rounds=3,
    consensus_threshold=0.67,  # 2/3 다수결
    low_agreement_threshold=0.4,  # 이 미만 → 유저 재질의
)

loop = ConsensusLoop(providers, config)
result = await loop.run("Should we use microservices?")

if result.success:
    print(f"Decision: {result.final_decision}")
else:
    print(f"User questions: {result.user_questions}")
```

### BrainstormModule

3라운드 브레인스토밍:

```python
from u_llm_sdk.multi_llm import BrainstormModule

module = BrainstormModule(providers)
result = await module.run_session("Architecture decision")

# result.rounds: 3라운드 전체 기록
# result.consensus: 최종 합의 결과
# result.all_discussion_entries: 전체 토론 기록 (요약 없음)
```

**핵심 원칙**: 전체 기록 보존 (요약 금지)

### Orchestrators

Master-Worker 패턴의 오케스트레이터:

```python
from u_llm_sdk.multi_llm import (
    OrchestratorFactory,
    GeminiOrchestrator,
    ClaudeOrchestrator,
    CodexOrchestrator,
)
from u_llm_sdk.types import Provider

providers = {
    Provider.GEMINI: gemini_provider,
    Provider.CLAUDE: claude_provider,
    Provider.CODEX: codex_provider,
}

# Factory로 생성 (권장)
factory = OrchestratorFactory(providers)
master = factory.create_master()  # Gemini (default)
claude_sub = factory.create_sub(Provider.CLAUDE)

# 직접 생성
master = GeminiOrchestrator(providers)

# 요청 처리
response = await master.process_request("Build auth system")
if response.needs_brainstorm:
    consensus = await master.facilitate_brainstorm(response.brainstorm_topic)
elif response.needs_clarification:
    # 유저에게 질문
    pass
else:
    # 작업 라우팅
    routing = await master.route_task(response.tasks[0])
```

**Provider별 Orchestrator 특성:**

| Orchestrator | 역할 | 특화 행동 |
|--------------|------|----------|
| `GeminiOrchestrator` | Master | 인간 의도 파악, 세션 관리, 작업 라우팅 |
| `ClaudeOrchestrator` | Sub (코드) | ClarityGate 필수 체크 → 불확실시 즉시 에스컬레이션 |
| `CodexOrchestrator` | Sub (분석) | Scope 제한 자동 적용 → 과도한 분석 방지 |

**핵심 원칙**: Gemini가 전략/디자인, Claude가 구현, Codex가 검증

### HybridOrchestrator

Multi-LLM 조율과 MergeExecutor 통합:

```python
from u_llm_sdk.multi_llm import (
    HybridOrchestrator,
    ExecutionMode,
    MigrationHelper,
)
from u_llm_sdk.llm.orchestration import MergeExecutorConfig

# MergeExecutor 설정
merge_config = MergeExecutorConfig(
    integration_branch="llm/auth-feature",
    create_pr=True,
    require_tests=True,
)

# Hybrid orchestrator 생성
orchestrator = HybridOrchestrator(
    providers=providers,
    merge_config=merge_config,
    brainstorm_threshold=0.6,
)

# 실행 (mode: HYBRID, MERGE_EXECUTOR_ONLY, MULTI_LLM, FALLBACK)
result = await orchestrator.run(
    request="Implement OAuth2 authentication",
    cwd="/project",
    mode=ExecutionMode.HYBRID,
    session_id="auth-001",  # 복구용
)

print(f"Success: {result.success}")
print(f"Mode: {result.execution_mode.value}")
if result.merge_result:
    print(f"PR: {result.merge_result.pr_url}")

# Migration helper: 점진적 전환 지원
helper = MigrationHelper()
if helper.should_use_multi_llm("Design system architecture"):
    # Multi-LLM 사용
    pass
else:
    # 단일 provider 사용
    pass
```

**Execution Modes:**

| Mode | 설명 |
|------|------|
| `HYBRID` | Multi-LLM 결정 + MergeExecutor 실행 |
| `MERGE_EXECUTOR_ONLY` | MergeExecutor만 사용 (Claude) |
| `MULTI_LLM` | Multi-LLM 조율만 (코드 실행 없음) |
| `FALLBACK` | Provider 불가시 대체 모드 |

### MultiLLMRAGHook

오케스트레이션 패턴 학습:

```python
from u_llm_sdk.multi_llm import (
    MultiLLMRAGHook,
    NoOpMultiLLMRAGHook,
    RAGTaskType,
    RoutingDecision,
)
from u_llm_sdk.rag_client import RAGClientConfig

# RAG hook 생성
config = RAGClientConfig(base_url="http://localhost:8000")
rag_hook = MultiLLMRAGHook(config)

# 라우팅 결정 저장
await rag_hook.save_routing_decision(
    request="Implement auth",
    task_type=RAGTaskType.CODE_IMPLEMENTATION,
    assigned_provider=Provider.CLAUDE,
    routing_reason="Code task -> Claude",
)

# 유사 패턴 검색
hints = await rag_hook.get_orchestration_hints("Build login system")
if hints.suggested_provider:
    print(f"추천: {hints.suggested_provider.value}")

# 테스트용 NoOp hook
noop = NoOpMultiLLMRAGHook()  # API 호출 없음
```

### StateRecoveryManager

세션 상태 복구:

```python
from u_llm_sdk.multi_llm import StateRecoveryManager
from pathlib import Path

recovery = StateRecoveryManager(Path("~/.cache/state").expanduser())

# 상태 저장
await recovery.save_state("session-001", {
    "in_progress": True,
    "phase": "brainstorming",
})

# 복구
state = await recovery.load_state("session-001")
if state and state.get("in_progress"):
    # 재개...
    pass

# 오래된 세션 정리
await recovery.cleanup_old_sessions(max_age_hours=24)
```

### SessionStore & SessionManager

세션 저장 및 관리:

```python
from u_llm_sdk.multi_llm import (
    SessionManager,
    InMemorySessionStore,
    FileSessionStore,
)
from u_llm_sdk.types import SessionConfig, Provider

# Store 선택 (InMemory 또는 File)
store = InMemorySessionStore()
# store = FileSessionStore(Path("~/.cache/sessions").expanduser())

# Manager 생성
manager = SessionManager(store)

# 세션 생성
state = await manager.create_session(
    session_id="session-001",
    orchestrator=Provider.GEMINI,
)

# 오케스트레이터 전환 (상태 보존)
await manager.switch_orchestrator("session-001", Provider.CLAUDE)

# 태스크 관리
task = Task(task_id="t1", objective="Implement auth", context="")
await manager.add_task("session-001", task)
await manager.complete_task("session-001", "t1")

# 세션 요약
summary = await manager.get_session_summary("session-001")
print(f"Active tasks: {summary['active_tasks_count']}")

# 이벤트 핸들러
manager.on("orchestrator_switched", lambda data: print(f"Switched: {data}"))
```

**SessionStore 구현체:**

| Store | 특징 |
|-------|------|
| `InMemorySessionStore` | 개발/테스트용, 프로세스 종료 시 소실 |
| `FileSessionStore` | 파일 기반 영속성, JSON 직렬화 |

## InterventionHook Protocol

```python
from u_llm_sdk.llm.providers import InterventionHook, NoOpHook

class InterventionHook(Protocol):
    async def on_pre_action(
        self, prompt: str, provider: str, model: Optional[str] = None,
        session_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Optional[PreActionContext]: ...

    async def on_post_action(
        self, result: LLMResult, pre_action_context: Optional[PreActionContext],
        run_id: Optional[str] = None
    ) -> None: ...
```

Provider는 실행 전후에 hook을 호출하여 RAG 컨텍스트 주입/피드백 수집.

## RAGClient

```python
from u_llm_sdk.rag_client import RAGClient, RAGClientConfig

config = RAGClientConfig(
    base_url="http://localhost:8000",
    timeout_seconds=0.5,
    fail_open=True,
    cache_ttl_seconds=300,
)

async with RAGClient(config) as client:
    # Pre-action (캐시 확인 → API 호출)
    context = await client.on_pre_action(prompt, provider, model)

    # Post-action (fire-and-forget)
    await client.on_post_action(result, context)
```

## Session Management (Unified)

File-based session management for all providers (migrated from claude-only-sdk):

```python
from u_llm_sdk import get_session_manager, SessionTemplate, Provider

# Get provider-specific manager
manager = get_session_manager(Provider.CLAUDE, "/project/path")

# Create session with system context
session_id = manager.create_from_system_prompt(
    "You are a security analyst.",
    assistant_acknowledgment="Ready to analyze.",
)

# Resume with CLI:
# claude --resume <session_id>
# codex resume <session_id>
# gemini --resume <session_id>
```

### Provider-Specific Session Storage

| Provider | Session Directory | File Format |
|----------|------------------|-------------|
| Claude | `~/.claude/projects/<path-key>/` | JSONL |
| Codex | `~/.codex/sessions/YYYY/MM/DD/` | JSONL with `session_meta.instructions` |
| Gemini | `~/.gemini/tmp/<project-hash>/chats/` | JSON with messages array |

### System Prompt Injection

Different providers handle system prompts differently:

```python
from u_llm_sdk import inject_system_prompt, Provider

# Claude: uses native --system-prompt flag
prompt, config = inject_system_prompt(Provider.CLAUDE, "user prompt", "system prompt")
# prompt == "user prompt", config == {"system_prompt": "system prompt"}

# Codex/Gemini: prepends to prompt
prompt, config = inject_system_prompt(Provider.CODEX, "user prompt", "system prompt")
# prompt == "system prompt\n\nuser prompt", config == {}
```

### Session Templates

10 built-in templates for specialized personas:

```python
from u_llm_sdk import SessionTemplate, get_template_prompt, template_run

# List available templates
templates = list_templates()
# CODE_REVIEWER, SECURITY_ANALYST, PYTHON_EXPERT, TYPESCRIPT_EXPERT,
# API_DESIGNER, TEST_ENGINEER, PERFORMANCE_ANALYST, DOCUMENTATION_WRITER,
# CRPG_STORYTELLER, CUSTOM_TELLER

# Get template prompt
prompt = get_template_prompt(SessionTemplate.SECURITY_ANALYST)

# Quick template execution
result = await template_run(
    "Review auth.py",
    SessionTemplate.SECURITY_ANALYST,
    provider=Provider.CLAUDE,
)
```

## Advanced Client (Unified)

Advanced orchestration client supporting all providers (migrated from claude-only-sdk):

```python
from u_llm_sdk import UnifiedAdvanced, AdvancedConfig, AgentDefinition, Provider

# Basic usage
async with UnifiedAdvanced(provider=Provider.CLAUDE) as client:
    result = await client.run("Hello!")

# With system prompt
result = await client.run_with_system_prompt(
    "Sort [3,1,2]",
    "You are a Python expert.",
)

# With template
result = await client.run_with_template(
    "Review auth.py",
    SessionTemplate.SECURITY_ANALYST,
)
```

### Agent Definitions

Agents can optionally specify a different provider:

```python
# Basic agent
planner = AgentDefinition(
    name="planner",
    description="Plans tasks",
    system_prompt="You are a planning expert.",
    tier=ModelTier.HIGH,
    allowed_tools=["Read", "Grep", "Glob"],
)

# Agent with different provider
codex_analyzer = AgentDefinition(
    name="analyzer",
    description="Analyzes code",
    system_prompt="You analyze code patterns.",
    provider=Provider.CODEX,  # Override client's provider
)

# Multi-provider workflow
async with UnifiedAdvanced(provider=Provider.GEMINI) as client:
    # planner uses client's provider (Gemini)
    # codex_analyzer uses its own provider (Codex)
    results = await client.run_with_agents(
        "Implement auth",
        [planner, codex_analyzer],
        sequential=True,
    )
```

### AdvancedConfig

```python
config = AdvancedConfig(
    provider=Provider.CLAUDE,
    tier=ModelTier.HIGH,
    auto_approval=AutoApproval.EDITS_ONLY,
    max_parallel_agents=3,
    agent_timeout_multiplier=1.5,
)

async with UnifiedAdvanced(config=config) as client:
    result = await client.run("Hello")
```

## Import Paths

```python
# Types (from internal types subpackage)
from u_llm_sdk.types import (
    # Enums
    Provider, ModelTier, AutoApproval, SandboxMode, ReasoningLevel,
    # Data models
    LLMResult, TokenUsage, FileChange, CommandRun, CodeBlock,
    # Hook data
    PreActionContext, PostActionFeedback,
    # Exceptions
    UnifiedLLMError, ProviderNotFoundError, AuthenticationError,
)

# Convenience re-exports (top-level)
from u_llm_sdk import Provider, LLMResult, PreActionContext

# Main API
from u_llm_sdk import LLM, LLMSync

# Config
from u_llm_sdk.config import LLMConfig, SAFE_CONFIG, AUTO_CONFIG

# Utilities
from u_llm_sdk.core.utils import quick_run, quick_text, parallel_run, template_run

# Providers
from u_llm_sdk.llm.providers import (
    BaseProvider, ClaudeProvider, CodexProvider, GeminiProvider,
    InterventionHook, NoOpHook,
)

# Orchestration
from u_llm_sdk.llm.orchestration import (
    # Contracts
    WorkOrder, Evidence, ReviewReport, ExecutionPlan,
    ROLES, RoleSpec, ExpectedDelta, EvidenceRequirement,
    # Types
    ExecutionState, WorkOrderState, ExecutionContext,
    # Orchestrators
    StateMachineOrchestrator, MergeExecutor, MergeExecutorConfig,
    # Verifiers
    Verifier, LLMVerifier, CompositeVerifier,
    # Git
    GitManager, merge_work_order,
    # Locks
    LockManager, InMemoryLockManager,
)

# RAG Client
from u_llm_sdk.rag_client import RAGClient, RAGClientConfig

# Multi-LLM Orchestration
from u_llm_sdk.multi_llm import (
    # Core components
    ClarityGate, EscalationProtocol, ConsensusLoop, BrainstormModule,
    # Orchestrators
    GeminiOrchestrator, ClaudeOrchestrator, CodexOrchestrator,
    OrchestratorFactory,
    # Hybrid integration
    HybridOrchestrator, ExecutionMode, MigrationHelper,
    StateRecoveryManager, HybridExecutionResult,
    # RAG integration
    MultiLLMRAGHook, NoOpMultiLLMRAGHook, RAGTaskType,
    RoutingDecision, PatternMatch, OrchestrationHint,
    # Performance & Monitoring
    PerformanceOptimizer, ParallelExecutor, LatencyTracker, MemoryManager,
    OrchestrationLogger, DebugMode, MetricExporter, EventEmitter,
    # Session Management
    SessionStore, SessionManager, InMemorySessionStore, FileSessionStore,
    SessionError, SessionExistsError, SessionNotFoundError,
)

# Session Management (File-based, all providers)
from u_llm_sdk.session import (
    # Base/Factory
    BaseSessionManager, get_session_manager, inject_system_prompt,
    # Provider implementations
    ClaudeSessionManager, CodexSessionManager, GeminiSessionManager,
    # Message
    SessionMessage, resolve_prompt,
    # Templates
    SessionTemplate, get_template_prompt, list_templates, create_custom_template,
)

# Advanced Client (Unified)
from u_llm_sdk.advanced import (
    UnifiedAdvanced, UnifiedAdvancedSync,
    AdvancedConfig, AgentDefinition,
)
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_state_machine.py -v
pytest tests/test_merge_executor.py -v
pytest tests/test_e2e_orchestration.py -v  # Phase 4 E2E tests
```

현재 테스트: **909개** 통과
