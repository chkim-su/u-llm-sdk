"""LLM Types - Domain Schema Package.

This package provides the schema layering system for domain-specific LLM outputs.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                     Schema Layering                         │
    ├─────────────────────────────────────────────────────────────┤
    │  Domain Layer (Variable):                                   │
    │    - BrainstormSchema: Multi-LLM discussion outputs         │
    │    - CodeReviewSchema: Code analysis outputs                │
    │    - AnalysisSchema: General analysis outputs               │
    │                                                             │
    │  Global Layer (Always Active):                              │
    │    - MV-RAG Observer: Behavior logging (parallel)           │
    │    - Extracts: tokens, duration, success, tool_calls        │
    └─────────────────────────────────────────────────────────────┘

Key Principles:
    1. Domain schemas define WHAT output structure is expected
    2. MV-RAG observes HOW the LLM behaved (independent of domain)
    3. Domain schema failures are Fail-Open (log warning, continue)
    4. Schemas are injected at Config level (session-wide)

Usage:
    >>> from u_llm_sdk.types.schemas import DomainSchema, BrainstormSchema
    >>> from u_llm_sdk.config import LLMConfig
    >>>
    >>> # Create config with domain schema
    >>> config = LLMConfig(
    ...     provider=Provider.CLAUDE,
    ...     domain_schema=BrainstormSchema(),
    ... )
    >>>
    >>> # Schema validation happens after LLM response
    >>> result = await llm.run(prompt)
    >>> # result.structured_output contains validated domain data
    >>> # MV-RAG independently observes behavior metadata
"""

from .base import (
    # Core types
    SchemaField,
    ValidationSeverity,
    ValidationResult,
    # Protocol
    DomainSchema,
    # Base implementation
    BaseDomainSchema,
    # Registry
    SchemaRegistry,
    get_schema,
    register_schema,
    list_schemas,
)

from .brainstorm import (
    BrainstormSchema,
    # Brainstorm-specific types
    BrainstormOutputType,
)

__all__ = [
    # Core types
    "SchemaField",
    "ValidationSeverity",
    "ValidationResult",
    # Protocol
    "DomainSchema",
    # Base implementation
    "BaseDomainSchema",
    # Registry
    "SchemaRegistry",
    "get_schema",
    "register_schema",
    "list_schemas",
    # Brainstorm
    "BrainstormSchema",
    "BrainstormOutputType",
]
