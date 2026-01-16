"""Domain Schema Base - Protocol and base implementation for domain schemas.

This module defines the core schema infrastructure:
    - DomainSchema: Protocol for all domain schemas
    - BaseDomainSchema: Base implementation with common functionality
    - SchemaRegistry: Global registry for schema discovery
    - Validation types: SchemaField, ValidationResult

Design Philosophy:
    - Schemas define expected OUTPUT structure (not input)
    - Validation is Fail-Open: warnings logged, execution continues
    - Schemas are type-safe via dataclass field definitions
    - MV-RAG is independent: observes behavior regardless of domain schema
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, TypeVar, runtime_checkable


class ValidationSeverity(Enum):
    """Severity level for schema validation results."""

    ERROR = "error"      # Critical: schema structure broken
    WARNING = "warning"  # Non-critical: missing optional fields
    INFO = "info"        # Informational: type coercion applied


@dataclass
class SchemaField:
    """Definition of a field in a domain schema.

    Attributes:
        name: Field name in output
        type_hint: Expected type ("str", "int", "list[str]", etc.)
        required: Whether field must be present
        description: Human-readable description
        default: Default value if missing (only for optional fields)
        nested_schema: For complex types, the nested schema name
    """

    name: str
    type_hint: str
    required: bool = True
    description: str = ""
    default: Any = None
    nested_schema: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "type_hint": self.type_hint,
            "required": self.required,
            "description": self.description,
            "default": self.default,
            "nested_schema": self.nested_schema,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SchemaField":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            type_hint=data.get("type_hint", "str"),
            required=data.get("required", True),
            description=data.get("description", ""),
            default=data.get("default"),
            nested_schema=data.get("nested_schema"),
        )


@dataclass
class ValidationResult:
    """Result of schema validation.

    Attributes:
        valid: Whether validation passed (no ERROR severity issues)
        issues: List of validation issues found
        coerced_data: Data after type coercion (if applicable)
        original_data: Original data before validation
    """

    valid: bool
    issues: list[dict] = field(default_factory=list)
    coerced_data: Optional[dict] = None
    original_data: Optional[dict] = None

    def add_issue(
        self,
        severity: ValidationSeverity,
        field_name: str,
        message: str,
    ) -> None:
        """Add a validation issue."""
        self.issues.append({
            "severity": severity.value,
            "field": field_name,
            "message": message,
        })
        if severity == ValidationSeverity.ERROR:
            self.valid = False

    def has_errors(self) -> bool:
        """Check if any ERROR severity issues exist."""
        return any(i["severity"] == "error" for i in self.issues)

    def has_warnings(self) -> bool:
        """Check if any WARNING severity issues exist."""
        return any(i["severity"] == "warning" for i in self.issues)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "valid": self.valid,
            "issues": self.issues,
            "coerced_data": self.coerced_data,
            "original_data": self.original_data,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ValidationResult":
        """Create from dictionary."""
        return cls(
            valid=data["valid"],
            issues=data.get("issues", []),
            coerced_data=data.get("coerced_data"),
            original_data=data.get("original_data"),
        )


@runtime_checkable
class DomainSchema(Protocol):
    """Protocol for domain-specific output schemas.

    Domain schemas define the expected structure of LLM outputs
    for specific use cases (brainstorming, code review, etc.).

    The schema is used to:
        1. Guide prompt construction (optional)
        2. Validate LLM output structure
        3. Parse output into typed data structures

    MV-RAG Relationship:
        Domain schemas are INDEPENDENT of MV-RAG logging.
        MV-RAG observes behavior (tokens, duration, success) regardless
        of whether a domain schema is active.
    """

    @property
    def name(self) -> str:
        """Unique schema name (e.g., 'brainstorm', 'code_review')."""
        ...

    @property
    def version(self) -> str:
        """Schema version for compatibility tracking."""
        ...

    @property
    def fields(self) -> list[SchemaField]:
        """List of expected output fields."""
        ...

    def validate(self, data: dict) -> ValidationResult:
        """Validate data against schema.

        Args:
            data: Raw output data to validate

        Returns:
            ValidationResult with validation status and issues
        """
        ...

    def get_prompt_guidance(self) -> Optional[str]:
        """Get optional prompt guidance for LLM.

        Returns guidance text that can be appended to prompts
        to help LLM produce schema-compliant output.

        Returns:
            Guidance text or None if not needed
        """
        ...

    def to_dict(self) -> dict:
        """Convert schema definition to dictionary."""
        ...


T = TypeVar("T", bound="BaseDomainSchema")


class BaseDomainSchema(ABC):
    """Base implementation of DomainSchema protocol.

    Provides common functionality for domain schemas:
        - Field definition via _define_fields()
        - Default validation logic
        - Prompt guidance generation
        - JSON serialization

    Subclasses should:
        1. Set _name and _version class attributes
        2. Implement _define_fields() to return field list
        3. Optionally override validate() for custom logic
    """

    _name: str = "base"
    _version: str = "1.0.0"

    def __init__(self):
        """Initialize schema with field definitions."""
        self._fields = self._define_fields()

    @property
    def name(self) -> str:
        """Unique schema name."""
        return self._name

    @property
    def version(self) -> str:
        """Schema version."""
        return self._version

    @property
    def fields(self) -> list[SchemaField]:
        """List of expected output fields."""
        return self._fields

    @abstractmethod
    def _define_fields(self) -> list[SchemaField]:
        """Define schema fields. Subclasses must implement.

        Returns:
            List of SchemaField definitions
        """
        ...

    def validate(self, data: dict) -> ValidationResult:
        """Validate data against schema.

        Default implementation checks:
            1. Required fields are present
            2. Field types match (basic type checking)
            3. Applies defaults for missing optional fields

        Args:
            data: Raw output data to validate

        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult(valid=True, original_data=data)
        coerced = dict(data) if data else {}

        for field_def in self._fields:
            # Check required fields
            if field_def.name not in data:
                if field_def.required:
                    result.add_issue(
                        ValidationSeverity.ERROR,
                        field_def.name,
                        f"Required field '{field_def.name}' is missing",
                    )
                else:
                    # Apply default for optional fields
                    coerced[field_def.name] = field_def.default
                    result.add_issue(
                        ValidationSeverity.INFO,
                        field_def.name,
                        f"Optional field '{field_def.name}' missing, using default",
                    )
                continue

            # Basic type validation
            value = data[field_def.name]
            if not self._check_type(value, field_def.type_hint):
                result.add_issue(
                    ValidationSeverity.WARNING,
                    field_def.name,
                    f"Field '{field_def.name}' has unexpected type "
                    f"(expected {field_def.type_hint}, got {type(value).__name__})",
                )

        result.coerced_data = coerced
        return result

    def _check_type(self, value: Any, type_hint: str) -> bool:
        """Basic type checking against type hint string.

        Supports: str, int, float, bool, list, dict, list[T], dict[K,V]
        """
        type_map = {
            "str": str,
            "int": int,
            "float": (int, float),
            "bool": bool,
            "list": list,
            "dict": dict,
            "any": object,
        }

        # Handle generic types like list[str]
        if "[" in type_hint:
            base_type = type_hint.split("[")[0]
            if base_type in type_map:
                return isinstance(value, type_map[base_type])

        # Handle simple types
        if type_hint.lower() in type_map:
            return isinstance(value, type_map[type_hint.lower()])

        # Unknown type: accept anything (fail-open)
        return True

    def get_prompt_guidance(self) -> Optional[str]:
        """Generate prompt guidance from field definitions.

        Creates a JSON schema-like description for LLM.
        """
        if not self._fields:
            return None

        lines = [
            f"Please format your response as JSON with the following structure:",
            "```json",
            "{",
        ]

        for i, field_def in enumerate(self._fields):
            optional = "" if field_def.required else " (optional)"
            comma = "," if i < len(self._fields) - 1 else ""
            lines.append(
                f'  "{field_def.name}": <{field_def.type_hint}>{optional}{comma}  '
                f'// {field_def.description}'
            )

        lines.extend([
            "}",
            "```",
        ])

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert schema definition to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "fields": [f.to_dict() for f in self._fields],
        }

    @classmethod
    def from_dict(cls: type[T], data: dict) -> T:
        """Create schema instance from dictionary.

        Note: This creates an instance with the same name/version
        but the actual field definitions come from _define_fields().
        """
        instance = cls()
        # Verify compatibility
        if instance.name != data.get("name"):
            raise ValueError(
                f"Schema name mismatch: expected '{instance.name}', "
                f"got '{data.get('name')}'"
            )
        return instance


class SchemaRegistry:
    """Global registry for domain schema discovery.

    Allows dynamic schema registration and lookup.
    Used by LLMConfig to resolve schema names to instances.

    Example:
        >>> register_schema(BrainstormSchema())
        >>> schema = get_schema("brainstorm")
        >>> print(schema.fields)
    """

    _schemas: dict[str, DomainSchema] = {}

    @classmethod
    def register(cls, schema: DomainSchema) -> None:
        """Register a schema instance.

        Args:
            schema: Schema instance to register

        Raises:
            ValueError: If schema with same name already registered
        """
        if schema.name in cls._schemas:
            raise ValueError(f"Schema '{schema.name}' already registered")
        cls._schemas[schema.name] = schema

    @classmethod
    def get(cls, name: str) -> Optional[DomainSchema]:
        """Get schema by name.

        Args:
            name: Schema name to look up

        Returns:
            Schema instance or None if not found
        """
        return cls._schemas.get(name)

    @classmethod
    def list_names(cls) -> list[str]:
        """List all registered schema names."""
        return list(cls._schemas.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered schemas (for testing)."""
        cls._schemas.clear()


# Module-level convenience functions
def register_schema(schema: DomainSchema) -> None:
    """Register a schema globally."""
    SchemaRegistry.register(schema)


def get_schema(name: str) -> Optional[DomainSchema]:
    """Get a schema by name."""
    return SchemaRegistry.get(name)


def list_schemas() -> list[str]:
    """List all registered schema names."""
    return SchemaRegistry.list_names()
