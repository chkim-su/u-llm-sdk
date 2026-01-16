"""CLI Discovery and Path Management.

Robust discovery of LLM CLI executables with persistent caching.
Ensures CLI paths are found once and reused across all library calls.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from glob import glob
from pathlib import Path
from typing import Optional

from u_llm_sdk.types import Provider


# =============================================================================
# Configuration
# =============================================================================

# Library config directory
CONFIG_DIR = Path.home() / ".u-llm-sdk"
CONFIG_FILE = CONFIG_DIR / "cli_paths.json"

# CLI binary names per provider
CLI_NAMES: dict[Provider, str] = {
    Provider.CLAUDE: "claude",
    Provider.CODEX: "codex",
    Provider.GEMINI: "gemini",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CLIInfo:
    """Information about a discovered CLI."""

    provider: str
    path: Optional[str] = None
    version: Optional[str] = None
    available: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CLIInfo:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class DiscoveryResult:
    """Result of CLI discovery process."""

    claude: CLIInfo = field(default_factory=lambda: CLIInfo(provider="claude"))
    codex: CLIInfo = field(default_factory=lambda: CLIInfo(provider="codex"))
    gemini: CLIInfo = field(default_factory=lambda: CLIInfo(provider="gemini"))

    def get(self, provider: Provider) -> CLIInfo:
        """Get CLI info for a provider."""
        return getattr(self, provider.value)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "claude": self.claude.to_dict(),
            "codex": self.codex.to_dict(),
            "gemini": self.gemini.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> DiscoveryResult:
        """Create from dictionary."""
        return cls(
            claude=CLIInfo.from_dict(data.get("claude", {"provider": "claude"})),
            codex=CLIInfo.from_dict(data.get("codex", {"provider": "codex"})),
            gemini=CLIInfo.from_dict(data.get("gemini", {"provider": "gemini"})),
        )

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = ["CLI Discovery Results:", "=" * 40]
        for provider in [self.claude, self.codex, self.gemini]:
            status = "+" if provider.available else "-"
            version = f" (v{provider.version})" if provider.version else ""
            path = provider.path or "not found"
            lines.append(f"  {status} {provider.provider}: {path}{version}")
        return "\n".join(lines)


# =============================================================================
# Path Search Functions
# =============================================================================

def _get_search_paths(cli_name: str) -> list[str]:
    """Get comprehensive list of paths to search for a CLI.

    Searches in order of priority:
    1. User local installations
    2. System locations
    3. npm global installations
    4. nvm node versions
    5. Homebrew (macOS)
    6. Snap (Linux)
    """
    home = Path.home()
    paths: list[str] = []

    # 1. User local bin (highest priority for user installs)
    paths.extend([
        str(home / ".local" / "bin" / cli_name),
        str(home / "bin" / cli_name),
    ])

    # 2. System locations
    paths.extend([
        f"/usr/local/bin/{cli_name}",
        f"/usr/bin/{cli_name}",
        f"/opt/bin/{cli_name}",
    ])

    # 3. npm global locations
    npm_prefixes = [
        home / ".npm-global" / "bin",
        home / ".npm" / "bin",
        home / "node_modules" / ".bin",
        Path("/usr/local/lib/node_modules/.bin"),
    ]
    for prefix in npm_prefixes:
        paths.append(str(prefix / cli_name))

    # 4. Try to get npm global bin dynamically
    try:
        result = subprocess.run(
            ["npm", "root", "-g"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            npm_root = Path(result.stdout.strip())
            # npm root -g returns node_modules, bin is sibling
            npm_bin = npm_root.parent / "bin" / cli_name
            paths.insert(0, str(npm_bin))
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass

    # 5. nvm installations (search all node versions)
    nvm_base = home / ".nvm" / "versions" / "node"
    if nvm_base.exists():
        for node_dir in nvm_base.iterdir():
            if node_dir.is_dir():
                paths.append(str(node_dir / "bin" / cli_name))

    # 6. fnm installations (Fast Node Manager)
    fnm_base = home / ".fnm" / "node-versions"
    if fnm_base.exists():
        for node_dir in fnm_base.iterdir():
            if node_dir.is_dir():
                paths.append(str(node_dir / "installation" / "bin" / cli_name))

    # 7. Homebrew (macOS)
    brew_paths = [
        f"/opt/homebrew/bin/{cli_name}",  # Apple Silicon
        f"/usr/local/Cellar/**/bin/{cli_name}",  # Intel
    ]
    paths.extend(brew_paths)

    # 8. Snap (Linux)
    paths.append(f"/snap/bin/{cli_name}")

    # 9. pipx installations (for Python-based CLIs)
    paths.append(
        str(home / ".local" / "pipx" / "venvs" / cli_name / "bin" / cli_name)
    )

    # 10. Windows specific (WSL compatibility)
    if os.name != "nt":
        # Check Windows paths from WSL
        win_paths = [
            f"/mnt/c/Users/*/AppData/Roaming/npm/{cli_name}",
            f"/mnt/c/Program Files/nodejs/{cli_name}",
        ]
        paths.extend(win_paths)

    return paths


def _find_cli_path(cli_name: str) -> Optional[str]:
    """Find CLI executable path using comprehensive search.

    Args:
        cli_name: Name of the CLI executable

    Returns:
        Full path to executable or None if not found
    """
    # First, try shutil.which (searches PATH)
    path = shutil.which(cli_name)
    if path:
        return path

    # Search all known locations
    for potential_path in _get_search_paths(cli_name):
        # Handle glob patterns
        if "*" in potential_path:
            matches = glob(potential_path)
            for match in matches:
                if os.path.isfile(match) and os.access(match, os.X_OK):
                    return match
        elif os.path.isfile(potential_path) and os.access(potential_path, os.X_OK):
            return potential_path

    return None


def _get_cli_version(cli_path: str) -> Optional[str]:
    """Get version string from CLI.

    Args:
        cli_path: Path to CLI executable

    Returns:
        Version string or None
    """
    try:
        result = subprocess.run(
            [cli_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # Parse version from output (usually first line)
            version_line = result.stdout.strip().split("\n")[0]
            # Extract version number (handle various formats)
            # e.g., "claude v1.2.3", "codex 1.2.3", "gemini version 1.2.3"
            match = re.search(r"[\d]+\.[\d]+\.[\d]+(?:[-.\w]*)?", version_line)
            if match:
                return match.group()
            return version_line[:50]  # Truncate if no version pattern found
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass
    return None


# =============================================================================
# Discovery Functions
# =============================================================================

def discover_cli(provider: Provider) -> CLIInfo:
    """Discover a single CLI.

    Args:
        provider: Provider to discover CLI for

    Returns:
        CLIInfo with discovery results
    """
    cli_name = CLI_NAMES.get(provider)
    if not cli_name:
        return CLIInfo(
            provider=provider.value,
            available=False,
            error=f"Unknown provider: {provider}",
        )

    path = _find_cli_path(cli_name)

    if path:
        version = _get_cli_version(path)
        return CLIInfo(
            provider=provider.value,
            path=path,
            version=version,
            available=True,
        )
    else:
        return CLIInfo(
            provider=provider.value,
            available=False,
            error=f"{cli_name} CLI not found in PATH or common locations",
        )


def discover_all() -> DiscoveryResult:
    """Discover all LLM CLIs.

    Returns:
        DiscoveryResult with all CLI information
    """
    return DiscoveryResult(
        claude=discover_cli(Provider.CLAUDE),
        codex=discover_cli(Provider.CODEX),
        gemini=discover_cli(Provider.GEMINI),
    )


# =============================================================================
# Persistence Functions
# =============================================================================

def _ensure_config_dir() -> None:
    """Ensure config directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def save_discovery(result: DiscoveryResult) -> None:
    """Save discovery results to config file.

    Args:
        result: Discovery results to save
    """
    _ensure_config_dir()
    with open(CONFIG_FILE, "w") as f:
        json.dump(result.to_dict(), f, indent=2)


def load_discovery() -> Optional[DiscoveryResult]:
    """Load discovery results from config file.

    Returns:
        DiscoveryResult if file exists and is valid, None otherwise
    """
    if not CONFIG_FILE.exists():
        return None

    try:
        with open(CONFIG_FILE) as f:
            data = json.load(f)
        return DiscoveryResult.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def clear_discovery() -> None:
    """Clear saved discovery results."""
    if CONFIG_FILE.exists():
        CONFIG_FILE.unlink()


# =============================================================================
# Global State Management
# =============================================================================

# Global cached discovery result
_cached_result: Optional[DiscoveryResult] = None


def get_discovery(force_refresh: bool = False) -> DiscoveryResult:
    """Get CLI discovery results (cached).

    First checks memory cache, then file cache, then performs discovery.

    Args:
        force_refresh: If True, ignore caches and rediscover

    Returns:
        DiscoveryResult with CLI paths
    """
    global _cached_result

    # Return memory cache if available and not forcing refresh
    if _cached_result is not None and not force_refresh:
        return _cached_result

    # Try loading from file cache
    if not force_refresh:
        loaded = load_discovery()
        if loaded is not None:
            # Validate paths still exist
            valid = True
            for provider in Provider:
                info = loaded.get(provider)
                if info.available and info.path:
                    if not os.path.isfile(info.path):
                        valid = False
                        break

            if valid:
                _cached_result = loaded
                return _cached_result

    # Perform fresh discovery
    _cached_result = discover_all()
    save_discovery(_cached_result)
    return _cached_result


def get_cli_path(provider: Provider) -> Optional[str]:
    """Get CLI path for a provider (convenience function).

    Args:
        provider: Provider to get path for

    Returns:
        Path to CLI or None if not available
    """
    result = get_discovery()
    info = result.get(provider)
    return info.path if info.available else None


def is_available(provider: Provider) -> bool:
    """Check if a provider CLI is available.

    Args:
        provider: Provider to check

    Returns:
        True if CLI is available
    """
    result = get_discovery()
    return result.get(provider).available


def available_providers() -> list[Provider]:
    """Get list of available providers.

    Returns:
        List of providers with available CLIs
    """
    result = get_discovery()
    return [p for p in Provider if result.get(p).available]


# =============================================================================
# Setup and Diagnostic Functions
# =============================================================================

def setup() -> DiscoveryResult:
    """Run CLI discovery and display results.

    Use this function for initial setup or troubleshooting.

    Returns:
        DiscoveryResult with all CLI information

    Example:
        >>> from u_llm_sdk.core.discovery import setup
        >>> result = setup()
        CLI Discovery Results:
        ========================================
          + claude: /home/user/.local/bin/claude (v1.0.0)
          + codex: /usr/local/bin/codex (v2.0.0)
          - gemini: not found
    """
    result = get_discovery(force_refresh=True)
    print(result.summary())
    return result


def refresh() -> DiscoveryResult:
    """Force refresh CLI discovery.

    Use when CLIs are installed/uninstalled during runtime.

    Returns:
        Fresh DiscoveryResult
    """
    return get_discovery(force_refresh=True)


def diagnose() -> dict:
    """Run diagnostic checks and return detailed information.

    Returns:
        Dictionary with diagnostic information
    """
    result = get_discovery(force_refresh=True)

    diagnostics: dict = {
        "config_dir": str(CONFIG_DIR),
        "config_file": str(CONFIG_FILE),
        "config_exists": CONFIG_FILE.exists(),
        "providers": {},
    }

    for provider in Provider:
        info = result.get(provider)
        cli_name = CLI_NAMES[provider]

        prov_diag = {
            "cli_name": cli_name,
            "available": info.available,
            "path": info.path,
            "version": info.version,
            "error": info.error,
            "searched_paths": _get_search_paths(cli_name)[:10],  # First 10
        }

        # Check if in PATH
        prov_diag["in_path"] = shutil.which(cli_name) is not None

        diagnostics["providers"][provider.value] = prov_diag

    return diagnostics


# =============================================================================
# Environment Variable Export
# =============================================================================

def export_to_env() -> dict[str, str]:
    """Export CLI paths to environment variables.

    Sets environment variables:
    - U_LLM_CLAUDE_PATH
    - U_LLM_CODEX_PATH
    - U_LLM_GEMINI_PATH

    Returns:
        Dictionary of set environment variables
    """
    result = get_discovery()
    env_vars: dict[str, str] = {}

    env_names = {
        Provider.CLAUDE: "U_LLM_CLAUDE_PATH",
        Provider.CODEX: "U_LLM_CODEX_PATH",
        Provider.GEMINI: "U_LLM_GEMINI_PATH",
    }

    for provider, env_name in env_names.items():
        info = result.get(provider)
        if info.available and info.path:
            os.environ[env_name] = info.path
            env_vars[env_name] = info.path

    return env_vars


def get_from_env(provider: Provider) -> Optional[str]:
    """Get CLI path from environment variable.

    Args:
        provider: Provider to get path for

    Returns:
        Path from environment or None
    """
    env_names = {
        Provider.CLAUDE: "U_LLM_CLAUDE_PATH",
        Provider.CODEX: "U_LLM_CODEX_PATH",
        Provider.GEMINI: "U_LLM_GEMINI_PATH",
    }
    return os.environ.get(env_names.get(provider, ""))


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Data classes
    "CLIInfo",
    "DiscoveryResult",
    # Discovery functions
    "discover_cli",
    "discover_all",
    "get_discovery",
    "get_cli_path",
    "is_available",
    "available_providers",
    # Setup and diagnostic
    "setup",
    "refresh",
    "diagnose",
    # Persistence
    "save_discovery",
    "load_discovery",
    "clear_discovery",
    # Environment
    "export_to_env",
    "get_from_env",
    # Constants
    "CONFIG_DIR",
    "CONFIG_FILE",
    "CLI_NAMES",
]
