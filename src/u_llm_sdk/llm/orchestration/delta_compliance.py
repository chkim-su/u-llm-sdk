"""Delta Compliance Verification for Contract-Based Parallel Execution.

Ensures that code changes comply with expected_delta contract by analyzing:
1. Public Exports - Functions/classes exposed in module interface
2. Dependency Edges - Import statements (internal and external)
3. Symbol Changes - Modified/added/removed symbols

Key Insight:
    "Contract violation" is not just "did you modify allowed files?"
    It's "did you ONLY make the EXPECTED changes?"

    WorkOrder says: "Add PaymentProcessor class"
    Actual change: Adds PaymentProcessor AND modifies User class
    → Delta violation (scope creep)

Supported Languages:
    - Python: AST parsing for __all__, top-level defs, imports
    - TypeScript/JavaScript: regex-based extraction (TODO: proper TS parser)

Usage:
    >>> analyzer = DeltaAnalyzer("/repo")
    >>> before = analyzer.analyze_snapshot(commit_before)
    >>> after = analyzer.analyze_snapshot(commit_after)
    >>> violations = analyzer.check_compliance(before, after, expected_delta)
"""

from __future__ import annotations

import ast
import hashlib
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from u_llm_sdk.types import LLMResult

    from .contracts import ExpectedDelta
    from .verifier import VerificationResult


# =============================================================================
# Data Structures
# =============================================================================


class SymbolType(Enum):
    """Type of exported symbol."""

    FUNCTION = "function"
    CLASS = "class"
    VARIABLE = "variable"
    CONSTANT = "constant"
    TYPE = "type"  # TypeScript type/interface


@dataclass
class ExportedSymbol:
    """A publicly exported symbol.

    Attributes:
        name: Symbol name
        symbol_type: Type of symbol (function, class, etc.)
        module_path: Module where defined (e.g., "src/payments/processor.py")
        line_number: Line number in source
        signature: Function/method signature (if applicable)
    """

    name: str
    symbol_type: SymbolType
    module_path: str
    line_number: int = 0
    signature: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.symbol_type.value,
            "module": self.module_path,
            "line": self.line_number,
            "signature": self.signature,
        }


@dataclass
class DependencyEdge:
    """An import/dependency relationship.

    Attributes:
        source_module: Module containing the import
        target_module: Module being imported
        imported_names: Specific names imported (empty for "import X")
        is_external: Whether target is external package
    """

    source_module: str
    target_module: str
    imported_names: List[str] = field(default_factory=list)
    is_external: bool = False

    def to_dict(self) -> dict:
        return {
            "source": self.source_module,
            "target": self.target_module,
            "names": self.imported_names,
            "external": self.is_external,
        }


@dataclass
class ModuleSnapshot:
    """Snapshot of a module's public interface.

    Attributes:
        path: Module file path
        exports: Set of exported symbol names
        symbols: Detailed symbol information
        dependencies: Import statements
        content_hash: Hash of file content
    """

    path: str
    exports: Set[str] = field(default_factory=set)
    symbols: List[ExportedSymbol] = field(default_factory=list)
    dependencies: List[DependencyEdge] = field(default_factory=list)
    content_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "exports": list(self.exports),
            "symbols": [s.to_dict() for s in self.symbols],
            "dependencies": [d.to_dict() for d in self.dependencies],
        }


@dataclass
class RepoSnapshot:
    """Snapshot of entire repository's public interface.

    Attributes:
        commit: Git commit ref used to create snapshot
        commit_sha: Actual resolved commit SHA (for reproducibility)
        base_commit_sha: Base commit SHA (for delta tracking)
        is_working_tree: Whether snapshot includes uncommitted changes
        modules: Dict of module path -> ModuleSnapshot
        worktree_path: Path to worktree (if using worktree isolation)
    """

    commit: str
    modules: Dict[str, ModuleSnapshot] = field(default_factory=dict)
    commit_sha: str = ""
    base_commit_sha: str = ""
    is_working_tree: bool = False
    worktree_path: str = ""

    @property
    def all_exports(self) -> Set[str]:
        """Get all exported symbol names."""
        exports = set()
        for module in self.modules.values():
            for symbol in module.symbols:
                exports.add(f"{module.path}:{symbol.name}")
        return exports

    @property
    def all_dependencies(self) -> List[DependencyEdge]:
        """Get all dependency edges."""
        deps = []
        for module in self.modules.values():
            deps.extend(module.dependencies)
        return deps


@dataclass
class DeltaViolation:
    """A violation of expected_delta contract.

    Attributes:
        violation_type: Type of violation
        description: Human-readable description
        file_path: File where violation occurred
        symbol_name: Symbol involved (if applicable)
        severity: "error" or "warning"
    """

    violation_type: str
    description: str
    file_path: str = ""
    symbol_name: str = ""
    severity: str = "error"

    def to_dict(self) -> dict:
        return {
            "type": self.violation_type,
            "description": self.description,
            "file": self.file_path,
            "symbol": self.symbol_name,
            "severity": self.severity,
        }


@dataclass
class ComplianceResult:
    """Result of delta compliance check.

    Attributes:
        compliant: Whether all checks passed
        violations: List of violations found
        new_exports: Symbols added
        removed_exports: Symbols removed
        modified_symbols: Symbols changed
        new_dependencies: Dependencies added
        removed_dependencies: Dependencies removed
    """

    compliant: bool
    violations: List[DeltaViolation] = field(default_factory=list)
    new_exports: Set[str] = field(default_factory=set)
    removed_exports: Set[str] = field(default_factory=set)
    modified_symbols: Set[str] = field(default_factory=set)
    new_dependencies: List[DependencyEdge] = field(default_factory=list)
    removed_dependencies: List[DependencyEdge] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "compliant": self.compliant,
            "violations": [v.to_dict() for v in self.violations],
            "new_exports": list(self.new_exports),
            "removed_exports": list(self.removed_exports),
            "modified_symbols": list(self.modified_symbols),
            "new_dependencies": [d.to_dict() for d in self.new_dependencies],
        }


# =============================================================================
# Python AST Analyzer
# =============================================================================


class PythonAnalyzer:
    """Analyzes Python modules for exports and dependencies.

    Uses AST parsing for accurate extraction.
    """

    def analyze_file(self, file_path: str, content: str) -> ModuleSnapshot:
        """Analyze a Python file.

        Args:
            file_path: Path to the file
            content: File content

        Returns:
            ModuleSnapshot with extracted information
        """
        snapshot = ModuleSnapshot(path=file_path)

        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError:
            # Can't parse - return empty snapshot
            return snapshot

        # Extract exports, symbols, and dependencies
        self._extract_exports(tree, snapshot)
        self._extract_symbols(tree, snapshot)
        self._extract_dependencies(tree, snapshot, file_path)

        return snapshot

    def _extract_exports(self, tree: ast.AST, snapshot: ModuleSnapshot):
        """Extract __all__ exports."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(
                                    elt.value, str
                                ):
                                    snapshot.exports.add(elt.value)

    def _extract_symbols(self, tree: ast.AST, snapshot: ModuleSnapshot):
        """Extract top-level function and class definitions."""
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(
                node, ast.AsyncFunctionDef
            ):
                # Public function (not starting with _)
                if not node.name.startswith("_") or node.name in snapshot.exports:
                    signature = self._get_function_signature(node)
                    snapshot.symbols.append(
                        ExportedSymbol(
                            name=node.name,
                            symbol_type=SymbolType.FUNCTION,
                            module_path=snapshot.path,
                            line_number=node.lineno,
                            signature=signature,
                        )
                    )

            elif isinstance(node, ast.ClassDef):
                # Public class
                if not node.name.startswith("_") or node.name in snapshot.exports:
                    snapshot.symbols.append(
                        ExportedSymbol(
                            name=node.name,
                            symbol_type=SymbolType.CLASS,
                            module_path=snapshot.path,
                            line_number=node.lineno,
                        )
                    )

            elif isinstance(node, ast.Assign):
                # Module-level constants (ALL_CAPS)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id.isupper() or target.id in snapshot.exports:
                            snapshot.symbols.append(
                                ExportedSymbol(
                                    name=target.id,
                                    symbol_type=SymbolType.CONSTANT,
                                    module_path=snapshot.path,
                                    line_number=node.lineno,
                                )
                            )

    def _extract_dependencies(
        self,
        tree: ast.AST,
        snapshot: ModuleSnapshot,
        file_path: str,
    ):
        """Extract import statements."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    snapshot.dependencies.append(
                        DependencyEdge(
                            source_module=file_path,
                            target_module=alias.name,
                            is_external=self._is_external(alias.name),
                        )
                    )

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_names = [alias.name for alias in node.names]
                    snapshot.dependencies.append(
                        DependencyEdge(
                            source_module=file_path,
                            target_module=node.module,
                            imported_names=imported_names,
                            is_external=self._is_external(node.module),
                        )
                    )

    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature."""
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)

        sig = f"({', '.join(args)})"
        if node.returns:
            sig += f" -> {ast.unparse(node.returns)}"

        return sig

    def _is_external(self, module_name: str) -> bool:
        """Check if module is external (not in project)."""
        # Simple heuristic: standard library and site-packages
        stdlib_modules = {
            "os",
            "sys",
            "re",
            "json",
            "typing",
            "dataclasses",
            "pathlib",
            "subprocess",
            "asyncio",
            "collections",
            "functools",
            "itertools",
            "time",
            "datetime",
            "enum",
            "abc",
            "hashlib",
            "uuid",
            "logging",
            "contextlib",
        }

        root_module = module_name.split(".")[0]
        return root_module in stdlib_modules or not module_name.startswith(".")


# =============================================================================
# TypeScript Analyzer (Tree-sitter with Regex fallback)
# =============================================================================


# Try to import tree-sitter (optional dependency)
_TREE_SITTER_AVAILABLE = False
_ts_parser = None
_ts_language = None

try:
    from tree_sitter import Parser
    from tree_sitter_languages import get_language, get_parser

    _ts_language = get_language("typescript")
    _ts_parser = get_parser("typescript")
    _TREE_SITTER_AVAILABLE = True
except ImportError:
    pass


class TypeScriptAnalyzer:
    """Analyzes TypeScript/JavaScript modules.

    Uses tree-sitter for accurate AST parsing when available,
    falls back to regex patterns when tree-sitter is not installed.

    Install tree-sitter support with:
        pip install u-llm-sdk[treesitter]
    """

    # Patterns for exports (fallback regex)
    EXPORT_PATTERNS = [
        # export function name
        r"export\s+(?:async\s+)?function\s+(\w+)",
        # export class name
        r"export\s+class\s+(\w+)",
        # export const/let/var name
        r"export\s+(?:const|let|var)\s+(\w+)",
        # export type/interface name
        r"export\s+(?:type|interface)\s+(\w+)",
        # export { name }
        r"export\s+\{\s*([^}]+)\s*\}",
        # export default
        r"export\s+default\s+(?:function|class)?\s*(\w+)?",
    ]

    # Patterns for imports (fallback regex)
    IMPORT_PATTERNS = [
        # import { x } from "y"
        r'import\s+\{([^}]+)\}\s+from\s+["\']([^"\']+)["\']',
        # import x from "y"
        r'import\s+(\w+)\s+from\s+["\']([^"\']+)["\']',
        # import * as x from "y"
        r'import\s+\*\s+as\s+(\w+)\s+from\s+["\']([^"\']+)["\']',
    ]

    def __init__(self):
        """Initialize analyzer with tree-sitter if available."""
        self.use_tree_sitter = _TREE_SITTER_AVAILABLE
        self._parser = _ts_parser

    @property
    def parser_type(self) -> str:
        """Return the parser type being used."""
        return "tree-sitter" if self.use_tree_sitter else "regex"

    def analyze_file(self, file_path: str, content: str) -> ModuleSnapshot:
        """Analyze a TypeScript/JavaScript file.

        Args:
            file_path: Path to the file
            content: File content

        Returns:
            ModuleSnapshot with extracted information
        """
        if self.use_tree_sitter:
            try:
                return self._analyze_with_tree_sitter(file_path, content)
            except Exception:
                # Fall back to regex on any tree-sitter error
                pass

        return self._analyze_with_regex(file_path, content)

    def _analyze_with_tree_sitter(
        self, file_path: str, content: str
    ) -> ModuleSnapshot:
        """Analyze using tree-sitter AST parsing.

        Provides accurate parsing of:
        - Nested generics (Map<string, Array<number>>)
        - Multiline exports/imports
        - Comments within declarations
        - Complex type expressions
        """
        snapshot = ModuleSnapshot(path=file_path)

        tree = self._parser.parse(content.encode("utf-8"))
        root = tree.root_node

        # Walk tree for exports and imports
        self._extract_exports_tree_sitter(root, content, file_path, snapshot)
        self._extract_imports_tree_sitter(root, content, file_path, snapshot)

        return snapshot

    def _extract_exports_tree_sitter(
        self, node, content: str, file_path: str, snapshot: ModuleSnapshot
    ):
        """Extract exports using tree-sitter AST."""
        # Export statement types in tree-sitter TypeScript grammar
        export_types = {
            "export_statement",
            "export_clause",
        }

        for child in self._walk_tree(node):
            if child.type == "export_statement":
                self._process_export_node(child, content, file_path, snapshot)

    def _process_export_node(
        self, node, content: str, file_path: str, snapshot: ModuleSnapshot
    ):
        """Process a single export node."""
        # Get the declaration child
        for child in node.children:
            if child.type == "function_declaration":
                name = self._get_identifier(child)
                if name:
                    snapshot.exports.add(name)
                    snapshot.symbols.append(
                        ExportedSymbol(
                            name=name,
                            symbol_type=SymbolType.FUNCTION,
                            module_path=file_path,
                            line_number=child.start_point[0] + 1,
                            signature=self._get_function_signature(child, content),
                        )
                    )

            elif child.type == "class_declaration":
                name = self._get_identifier(child)
                if name:
                    snapshot.exports.add(name)
                    snapshot.symbols.append(
                        ExportedSymbol(
                            name=name,
                            symbol_type=SymbolType.CLASS,
                            module_path=file_path,
                            line_number=child.start_point[0] + 1,
                        )
                    )

            elif child.type in ("lexical_declaration", "variable_declaration"):
                # export const/let/var
                for decl in child.children:
                    if decl.type == "variable_declarator":
                        name = self._get_identifier(decl)
                        if name:
                            snapshot.exports.add(name)
                            snapshot.symbols.append(
                                ExportedSymbol(
                                    name=name,
                                    symbol_type=SymbolType.VARIABLE,
                                    module_path=file_path,
                                    line_number=decl.start_point[0] + 1,
                                )
                            )

            elif child.type in ("type_alias_declaration", "interface_declaration"):
                name = self._get_identifier(child)
                if name:
                    snapshot.exports.add(name)
                    snapshot.symbols.append(
                        ExportedSymbol(
                            name=name,
                            symbol_type=SymbolType.TYPE,
                            module_path=file_path,
                            line_number=child.start_point[0] + 1,
                        )
                    )

            elif child.type == "export_clause":
                # export { a, b, c }
                for spec in child.children:
                    if spec.type == "export_specifier":
                        name = self._get_identifier(spec)
                        if name:
                            snapshot.exports.add(name)
                            snapshot.symbols.append(
                                ExportedSymbol(
                                    name=name,
                                    symbol_type=SymbolType.VARIABLE,
                                    module_path=file_path,
                                    line_number=spec.start_point[0] + 1,
                                )
                            )

    def _extract_imports_tree_sitter(
        self, node, content: str, file_path: str, snapshot: ModuleSnapshot
    ):
        """Extract imports using tree-sitter AST."""
        for child in self._walk_tree(node):
            if child.type == "import_statement":
                self._process_import_node(child, content, file_path, snapshot)

    def _process_import_node(
        self, node, content: str, file_path: str, snapshot: ModuleSnapshot
    ):
        """Process a single import node."""
        imported_names = []
        target_module = ""

        for child in node.children:
            if child.type == "import_clause":
                # Get imported names
                for spec in self._walk_tree(child):
                    if spec.type == "identifier":
                        imported_names.append(
                            content[spec.start_byte : spec.end_byte]
                        )
                    elif spec.type == "import_specifier":
                        name = self._get_identifier(spec)
                        if name:
                            imported_names.append(name)

            elif child.type == "string":
                # Module path (remove quotes)
                target_module = content[child.start_byte + 1 : child.end_byte - 1]

        if target_module:
            snapshot.dependencies.append(
                DependencyEdge(
                    source_module=file_path,
                    target_module=target_module,
                    imported_names=imported_names,
                    is_external=not target_module.startswith("."),
                )
            )

    def _walk_tree(self, node):
        """Generator that yields all nodes in the tree."""
        cursor = node.walk()

        visited_children = False
        while True:
            if not visited_children:
                yield cursor.node
                if cursor.goto_first_child():
                    continue

            if cursor.goto_next_sibling():
                visited_children = False
            elif cursor.goto_parent():
                visited_children = True
            else:
                break

    def _get_identifier(self, node) -> Optional[str]:
        """Get identifier name from a node."""
        for child in node.children:
            if child.type in ("identifier", "type_identifier"):
                return child.text.decode("utf-8") if child.text else None
        return None

    def _get_function_signature(self, node, content: str) -> str:
        """Extract function signature from tree-sitter node."""
        for child in node.children:
            if child.type == "formal_parameters":
                return content[child.start_byte : child.end_byte]
        return "()"

    def _analyze_with_regex(self, file_path: str, content: str) -> ModuleSnapshot:
        """Analyze using regex patterns (fallback).

        Less accurate than tree-sitter but has no dependencies.
        May miss complex patterns like multiline exports or nested generics.
        """
        snapshot = ModuleSnapshot(path=file_path)

        # Extract exports
        for pattern in self.EXPORT_PATTERNS:
            for match in re.finditer(pattern, content):
                names = match.group(1)
                if names:
                    # Handle { a, b, c } style exports
                    if "," in names:
                        for name in names.split(","):
                            name = name.strip().split(" as ")[0].strip()
                            if name:
                                snapshot.exports.add(name)
                                snapshot.symbols.append(
                                    ExportedSymbol(
                                        name=name,
                                        symbol_type=SymbolType.VARIABLE,
                                        module_path=file_path,
                                    )
                                )
                    else:
                        name = names.strip()
                        snapshot.exports.add(name)
                        # Determine type from pattern
                        symbol_type = SymbolType.VARIABLE
                        if "function" in pattern:
                            symbol_type = SymbolType.FUNCTION
                        elif "class" in pattern:
                            symbol_type = SymbolType.CLASS
                        elif "type|interface" in pattern:
                            symbol_type = SymbolType.TYPE

                        snapshot.symbols.append(
                            ExportedSymbol(
                                name=name,
                                symbol_type=symbol_type,
                                module_path=file_path,
                            )
                        )

        # Extract imports
        for pattern in self.IMPORT_PATTERNS:
            for match in re.finditer(pattern, content):
                names_str = match.group(1)
                target = match.group(2)

                imported_names = []
                if "," in names_str or "{" in pattern:
                    for name in names_str.split(","):
                        name = name.strip().split(" as ")[0].strip()
                        if name:
                            imported_names.append(name)
                else:
                    imported_names = [names_str.strip()]

                snapshot.dependencies.append(
                    DependencyEdge(
                        source_module=file_path,
                        target_module=target,
                        imported_names=imported_names,
                        is_external=not target.startswith("."),
                    )
                )

        return snapshot


# =============================================================================
# Delta Analyzer
# =============================================================================


class DeltaAnalyzer:
    """Analyzes code changes and checks delta compliance.

    Main entry point for delta compliance verification.

    Example:
        >>> analyzer = DeltaAnalyzer("/repo")
        >>> before = analyzer.analyze_current()
        >>> # ... make changes ...
        >>> after = analyzer.analyze_current()
        >>> result = analyzer.check_compliance(before, after, expected_delta)
    """

    def __init__(self, repo_path: str):
        """Initialize delta analyzer.

        Args:
            repo_path: Path to git repository root
        """
        self.repo_path = Path(repo_path).resolve()
        self.python_analyzer = PythonAnalyzer()
        self.ts_analyzer = TypeScriptAnalyzer()

    def analyze_current(self) -> RepoSnapshot:
        """Analyze current working tree.

        Returns:
            RepoSnapshot of current state (includes uncommitted changes)
        """
        snapshot = self.analyze_snapshot("HEAD")
        snapshot.is_working_tree = True
        return snapshot

    def analyze_snapshot(self, ref: str = "HEAD") -> RepoSnapshot:
        """Analyze repository at a specific ref.

        Args:
            ref: Git ref (commit hash, branch, tag)

        Returns:
            RepoSnapshot at that ref
        """
        snapshot = RepoSnapshot(commit=ref)

        # Resolve commit SHA for reproducibility
        snapshot.commit_sha = self._resolve_commit_sha(ref)
        snapshot.worktree_path = str(self.repo_path)

        # Get list of tracked files
        files = self._get_tracked_files(ref)

        for file_path in files:
            if self._should_analyze(file_path):
                try:
                    content = self._get_file_content(file_path, ref)
                    module_snapshot = self._analyze_file(file_path, content)
                    # Compute content hash for change detection
                    module_snapshot.content_hash = self._compute_content_hash(content)
                    snapshot.modules[file_path] = module_snapshot
                except Exception:
                    # Skip files that can't be analyzed
                    continue

        return snapshot

    def analyze_at_commit(
        self,
        commit_sha: str,
        base_commit_sha: Optional[str] = None,
    ) -> RepoSnapshot:
        """Analyze repository at a specific commit (for reproducible snapshots).

        This is for commit-based delta compliance.
        Unlike analyze_snapshot, this:
        - Uses exact commit SHA (not ref)
        - Records base_commit for delta tracking
        - Is fully reproducible

        Args:
            commit_sha: Exact commit SHA to analyze
            base_commit_sha: Base commit SHA (for delta tracking)

        Returns:
            RepoSnapshot at that exact commit
        """
        snapshot = self.analyze_snapshot(commit_sha)
        snapshot.commit_sha = commit_sha
        snapshot.base_commit_sha = base_commit_sha or ""
        snapshot.is_working_tree = False
        return snapshot

    def check_compliance(
        self,
        before: RepoSnapshot,
        after: RepoSnapshot,
        expected_delta: Optional["ExpectedDelta"] = None,
    ) -> ComplianceResult:
        """Check if changes comply with expected_delta.

        Args:
            before: Snapshot before changes
            after: Snapshot after changes
            expected_delta: Expected delta contract (None = no restrictions)

        Returns:
            ComplianceResult with violations and diff info
        """
        from .contracts import ExpectedDelta

        result = ComplianceResult(compliant=True)

        # Compute diffs
        result.new_exports = after.all_exports - before.all_exports
        result.removed_exports = before.all_exports - after.all_exports
        result.modified_symbols = self._compute_modified_symbols(before, after)
        result.new_dependencies = self._compute_new_dependencies(before, after)
        result.removed_dependencies = self._compute_removed_dependencies(before, after)

        # If no expected_delta, just return diff info
        if expected_delta is None:
            return result

        # Check constraints
        violations = []

        # 1. Check forbid_new_public_exports
        if expected_delta.forbid_new_public_exports and result.new_exports:
            for export in result.new_exports:
                violations.append(
                    DeltaViolation(
                        violation_type="new_public_export",
                        description=f"New public export not allowed: {export}",
                        symbol_name=export,
                    )
                )

        # 2. Check forbid_new_deps
        if expected_delta.forbid_new_deps:
            forbidden_patterns = set(expected_delta.forbid_new_deps)
            for dep in result.new_dependencies:
                if dep.target_module in forbidden_patterns:
                    violations.append(
                        DeltaViolation(
                            violation_type="forbidden_dependency",
                            description=f"Forbidden dependency added: {dep.target_module}",
                            file_path=dep.source_module,
                        )
                    )
                # Also check for pattern matching
                for pattern in forbidden_patterns:
                    if "*" in pattern:
                        regex = pattern.replace("*", ".*")
                        if re.match(regex, dep.target_module):
                            violations.append(
                                DeltaViolation(
                                    violation_type="forbidden_dependency",
                                    description=f"Dependency matches forbidden pattern: {dep.target_module} matches {pattern}",
                                    file_path=dep.source_module,
                                )
                            )

        # 3. Check allow_symbol_changes
        if expected_delta.allow_symbol_changes is not None:
            allowed = set(expected_delta.allow_symbol_changes)
            for symbol in result.modified_symbols:
                symbol_name = symbol.split(":")[-1]  # Extract just the name
                if symbol_name not in allowed and symbol not in allowed:
                    violations.append(
                        DeltaViolation(
                            violation_type="unauthorized_symbol_change",
                            description=f"Symbol changed but not in allow list: {symbol}",
                            symbol_name=symbol,
                        )
                    )

        # 4. Check max_files_modified
        if expected_delta.max_files_modified:
            modified_files = set()
            for module in after.modules:
                if module not in before.modules:
                    modified_files.add(module)
                elif (
                    before.modules[module].content_hash
                    != after.modules[module].content_hash
                ):
                    modified_files.add(module)

            for module in before.modules:
                if module not in after.modules:
                    modified_files.add(module)

            if len(modified_files) > expected_delta.max_files_modified:
                violations.append(
                    DeltaViolation(
                        violation_type="too_many_files_modified",
                        description=f"Modified {len(modified_files)} files, max allowed: {expected_delta.max_files_modified}",
                    )
                )

        # 5. Check max_files_created
        if expected_delta.max_files_created is not None:
            created_files = set()
            for module in after.modules:
                if module not in before.modules:
                    created_files.add(module)

            if len(created_files) > expected_delta.max_files_created:
                violations.append(
                    DeltaViolation(
                        violation_type="too_many_files_created",
                        description=f"Created {len(created_files)} files, max allowed: {expected_delta.max_files_created}",
                    )
                )

        result.violations = violations
        result.compliant = len(violations) == 0

        return result

    def _resolve_commit_sha(self, ref: str) -> str:
        """Resolve a git ref to its actual commit SHA."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", ref],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip() if result.returncode == 0 else ref
        except Exception:
            return ref

    def _compute_content_hash(self, content: str) -> str:
        """Compute hash of file content for change detection."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_tracked_files(self, ref: str) -> List[str]:
        """Get list of tracked files at ref."""
        try:
            if ref == "HEAD":
                result = subprocess.run(
                    ["git", "ls-files"],
                    cwd=str(self.repo_path),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            else:
                result = subprocess.run(
                    ["git", "ls-tree", "-r", "--name-only", ref],
                    cwd=str(self.repo_path),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

            if result.returncode != 0:
                return []
            return [f for f in result.stdout.strip().split("\n") if f]
        except Exception:
            return []

    def _get_file_content(self, file_path: str, ref: str) -> str:
        """Get file content at ref."""
        if ref == "HEAD":
            # Read from working tree
            full_path = self.repo_path / file_path
            if full_path.exists():
                return full_path.read_text()
            return ""

        try:
            result = subprocess.run(
                ["git", "show", f"{ref}:{file_path}"],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout if result.returncode == 0 else ""
        except Exception:
            return ""

    def _should_analyze(self, file_path: str) -> bool:
        """Check if file should be analyzed."""
        # Python files
        if file_path.endswith(".py"):
            return True
        # TypeScript/JavaScript files
        if file_path.endswith((".ts", ".tsx", ".js", ".jsx")):
            return True
        return False

    def _analyze_file(self, file_path: str, content: str) -> ModuleSnapshot:
        """Analyze a file based on its extension."""
        if file_path.endswith(".py"):
            return self.python_analyzer.analyze_file(file_path, content)
        elif file_path.endswith((".ts", ".tsx", ".js", ".jsx")):
            return self.ts_analyzer.analyze_file(file_path, content)
        return ModuleSnapshot(path=file_path)

    def _compute_modified_symbols(
        self,
        before: RepoSnapshot,
        after: RepoSnapshot,
    ) -> Set[str]:
        """Compute symbols that were modified."""
        modified = set()

        # Compare symbols in modules that exist in both
        for path, after_module in after.modules.items():
            if path in before.modules:
                before_module = before.modules[path]

                # Build symbol maps
                before_symbols = {s.name: s for s in before_module.symbols}
                after_symbols = {s.name: s for s in after_module.symbols}

                # Check for modifications
                for name, after_sym in after_symbols.items():
                    if name in before_symbols:
                        before_sym = before_symbols[name]
                        # Compare signatures for functions
                        if (
                            after_sym.symbol_type == SymbolType.FUNCTION
                            and after_sym.signature != before_sym.signature
                        ):
                            modified.add(f"{path}:{name}")
                        # Compare line numbers (moved)
                        elif after_sym.line_number != before_sym.line_number:
                            # Only count as modified if significantly moved
                            if abs(after_sym.line_number - before_sym.line_number) > 5:
                                modified.add(f"{path}:{name}")

        return modified

    def _compute_new_dependencies(
        self,
        before: RepoSnapshot,
        after: RepoSnapshot,
    ) -> List[DependencyEdge]:
        """Compute new dependencies added."""
        before_deps = set()
        for module in before.modules.values():
            for dep in module.dependencies:
                before_deps.add((dep.source_module, dep.target_module))

        new_deps = []
        for module in after.modules.values():
            for dep in module.dependencies:
                key = (dep.source_module, dep.target_module)
                if key not in before_deps:
                    new_deps.append(dep)

        return new_deps

    def _compute_removed_dependencies(
        self,
        before: RepoSnapshot,
        after: RepoSnapshot,
    ) -> List[DependencyEdge]:
        """Compute dependencies that were removed."""
        after_deps = set()
        for module in after.modules.values():
            for dep in module.dependencies:
                after_deps.add((dep.source_module, dep.target_module))

        removed_deps = []
        for module in before.modules.values():
            for dep in module.dependencies:
                key = (dep.source_module, dep.target_module)
                if key not in after_deps:
                    removed_deps.append(dep)

        return removed_deps


# =============================================================================
# Verifier Integration
# =============================================================================


class DeltaComplianceVerifier:
    """Verifier that checks delta compliance.

    Integrates with the verifier framework for WorkOrder compliance checking.
    """

    def __init__(
        self,
        repo_path: str,
        expected_delta: Optional["ExpectedDelta"] = None,
    ):
        """Initialize verifier.

        Args:
            repo_path: Path to repository
            expected_delta: Expected delta contract
        """
        self.analyzer = DeltaAnalyzer(repo_path)
        self.expected_delta = expected_delta
        self._before_snapshot: Optional[RepoSnapshot] = None

    def capture_before(self):
        """Capture snapshot before changes.

        Call this before WorkOrder execution.
        """
        self._before_snapshot = self.analyzer.analyze_current()

    async def verify(
        self,
        result: "LLMResult",
        context: Any = None,
    ) -> "VerificationResult":
        """Verify delta compliance after execution.

        Args:
            result: LLM execution result
            context: Additional context (WorkOrder)

        Returns:
            VerificationResult with compliance status
        """
        from .contracts import WorkOrder
        from .verifier import VerificationResult

        # Use WorkOrder's expected_delta if available
        expected_delta = self.expected_delta
        if isinstance(context, WorkOrder) and context.expected_delta:
            expected_delta = context.expected_delta

        # Capture after snapshot
        after_snapshot = self.analyzer.analyze_current()

        # If no before snapshot, can't compare
        if self._before_snapshot is None:
            return VerificationResult(
                passed=True,
                message="No before snapshot captured - skipping delta compliance",
                details={"warning": "capture_before() was not called"},
            )

        # Check compliance
        compliance = self.analyzer.check_compliance(
            self._before_snapshot,
            after_snapshot,
            expected_delta,
        )

        if compliance.compliant:
            return VerificationResult(
                passed=True,
                message="Delta compliance check passed",
                details={
                    "new_exports": list(compliance.new_exports),
                    "modified_symbols": list(compliance.modified_symbols),
                    "new_dependencies": [
                        d.to_dict() for d in compliance.new_dependencies
                    ],
                },
            )
        else:
            violation_messages = [v.description for v in compliance.violations]
            return VerificationResult(
                passed=False,
                message=f"Delta compliance violations: {len(compliance.violations)}",
                details={
                    "violations": [v.to_dict() for v in compliance.violations],
                    "violation_messages": violation_messages,
                },
            )


# =============================================================================
# Utility Functions
# =============================================================================


def quick_delta_check(
    repo_path: str,
    before_ref: str,
    after_ref: str = "HEAD",
    expected_delta: Optional["ExpectedDelta"] = None,
) -> ComplianceResult:
    """Quick delta compliance check between two refs.

    Args:
        repo_path: Path to repository
        before_ref: Git ref before changes
        after_ref: Git ref after changes (default: HEAD)
        expected_delta: Expected delta contract

    Returns:
        ComplianceResult with compliance status

    Example:
        >>> result = quick_delta_check("/repo", "main", "feature-branch")
        >>> if not result.compliant:
        ...     print(result.violations)
    """
    analyzer = DeltaAnalyzer(repo_path)
    before = analyzer.analyze_snapshot(before_ref)
    after = analyzer.analyze_snapshot(after_ref)
    return analyzer.check_compliance(before, after, expected_delta)


def get_public_exports(file_path: str) -> Set[str]:
    """Get public exports from a single file.

    Args:
        file_path: Path to Python/TS file

    Returns:
        Set of exported symbol names

    Example:
        >>> exports = get_public_exports("src/payments/processor.py")
        >>> print(exports)  # {'PaymentProcessor', 'process_payment', ...}
    """
    path = Path(file_path)
    if not path.exists():
        return set()

    content = path.read_text()

    if file_path.endswith(".py"):
        analyzer = PythonAnalyzer()
    else:
        analyzer = TypeScriptAnalyzer()

    snapshot = analyzer.analyze_file(file_path, content)
    return snapshot.exports


__all__ = [
    # Enums
    "SymbolType",
    # Data structures
    "ExportedSymbol",
    "DependencyEdge",
    "ModuleSnapshot",
    "RepoSnapshot",
    "DeltaViolation",
    "ComplianceResult",
    # Analyzers
    "PythonAnalyzer",
    "TypeScriptAnalyzer",
    "DeltaAnalyzer",
    # Verifier
    "DeltaComplianceVerifier",
    # Utility functions
    "quick_delta_check",
    "get_public_exports",
]
