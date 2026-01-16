# U-LLM-SDK Skill

A Claude Code skill that provides knowledge about the u-llm-sdk Python library for unified multi-provider LLM access.

## What This Skill Provides

This skill helps Claude answer questions about:
- Installing and using u-llm-sdk
- Configuring different providers (Claude, Codex, Gemini)
- Multi-LLM orchestration patterns
- Session management and templates
- Advanced features and provider-specific options

## Activation

This skill activates when users ask about:
- "LLM SDK"
- "multi-provider LLM"
- "Claude/Codex/Gemini integration"
- "u-llm-sdk"
- "unified LLM client"
- Programmatic LLM access across providers

## Structure

```
u-llm-sdk/
├── SKILL.md                    # Main skill content
├── README.md                   # This file
├── examples/
│   ├── basic_usage.py          # Basic async/sync examples
│   └── multi_provider.py       # Multi-provider comparison
└── references/
    └── configuration.md        # Complete configuration reference
```

## Installation of u-llm-sdk Library

The skill documents how to install the actual u-llm-sdk package:

```bash
# From PyPI (when published)
pip install u-llm-sdk

# From wheel
pip install /path/to/u_llm_sdk-0.3.0-py3-none-any.whl

# From source (editable)
pip install -e /path/to/u-llm-sdk
```

## Quick Reference

### Basic Usage

```python
from u_llm_sdk import LLM, Provider

async with LLM(provider=Provider.CLAUDE) as llm:
    result = await llm.run("Hello!")
    print(result.text)
```

### With Configuration

```python
from u_llm_sdk import LLM
from u_llm_sdk.config import LLMConfig
from u_llm_sdk.types import Provider, AutoApproval

config = LLMConfig(
    provider=Provider.CLAUDE,
    auto_approval=AutoApproval.FULL,
    timeout=120,
)

async with LLM(config=config) as llm:
    result = await llm.run("Help me code")
```

## Package Details

- **Location**: `/home/chanhokim/projects/new_project/claude-plugin/u-llm-sdk/`
- **Distribution**: `dist/u_llm_sdk-0.3.0-py3-none-any.whl`
- **Version**: 0.3.0
- **Python**: >=3.9

## Related Files

- **Library source**: `../../../src/u_llm_sdk/`
- **Tests**: `../../../tests/`
- **Documentation**: `../../../CLAUDE.md`
