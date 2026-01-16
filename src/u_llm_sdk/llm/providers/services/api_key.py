"""API Key Resolution Service.

Extracts API key management from BaseProvider (SRP compliance).
Handles resolution priority, .env file loading, and security validation.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from u_llm_sdk.types import API_KEY_ENV_VARS, Provider

if TYPE_CHECKING:
    from u_llm_sdk.config import LLMConfig

logger = logging.getLogger(__name__)


class ApiKeyResolver:
    """Resolves API keys for LLM provider subprocess execution.

    Resolution priority:
    1. config.api_key (explicit)
    2. config.env_file (.env file)
    3. Current environment variable (passthrough)

    Security features:
    - Symlink attack detection (always blocks)
    - File ownership validation (always blocks)
    - Permission validation (warns or blocks based on strict mode)
    """

    def __init__(self, provider: Provider, config: LLMConfig) -> None:
        """Initialize resolver.

        Args:
            provider: Provider type (determines env var name)
            config: LLMConfig with api_key, env_file settings
        """
        self._provider = provider
        self._config = config
        self._env_var_name = API_KEY_ENV_VARS.get(provider)

    @property
    def env_var_name(self) -> Optional[str]:
        """Environment variable name for this provider's API key."""
        return self._env_var_name

    def resolve(self) -> Optional[str]:
        """Resolve API key from config or environment.

        Returns:
            API key string or None (None means use existing environment)
        """
        # Priority 1: Explicit API key
        if self._config.api_key:
            return self._config.api_key

        # Priority 2: Load from .env file
        if self._config.env_file:
            env_key = self._load_from_env_file(self._config.env_file)
            if env_key:
                return env_key

        # Priority 3: Already in environment (return None to use existing)
        return None

    def get_env_with_api_key(self) -> dict[str, str]:
        """Get environment variables with API key for subprocess execution.

        Returns:
            Environment dict with API key set (if resolved)
        """
        env = os.environ.copy()
        api_key = self.resolve()

        if api_key and self._env_var_name:
            env[self._env_var_name] = api_key

        return env

    def _load_from_env_file(self, env_file: str) -> Optional[str]:
        """Load API key from .env file.

        Args:
            env_file: Path to .env file

        Returns:
            API key if found and valid, None otherwise
        """
        env_path = Path(env_file)
        if not env_path.exists():
            return None

        # Security validation
        strict = getattr(self._config, "strict_env_security", False)
        security_error = self._validate_security(env_path, strict=strict)
        if security_error:
            logger.error(security_error)
            return None

        if not self._env_var_name:
            return None

        try:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key == self._env_var_name:
                            return value
        except (IOError, OSError) as e:
            logger.warning(f"Failed to read .env file {env_file}: {e}")

        return None

    def _validate_security(
        self, env_path: Path, strict: bool = False
    ) -> Optional[str]:
        """Validate .env file security properties.

        Args:
            env_path: Path to .env file
            strict: If True, block on permissive permissions (not just warn)

        Returns:
            Error message if validation fails, None if OK

        Security checks:
            1. Symlink attack detection - always blocks
            2. File ownership (must be current user) - always blocks
            3. Permissive permissions (group/world readable) - warns or blocks
        """
        # Check for symlink attack
        if env_path.is_symlink():
            return f"Security: .env file is a symlink (potential attack): {env_path}"

        try:
            file_stat = env_path.stat()

            # Check file ownership (must be current user)
            if hasattr(os, "getuid") and file_stat.st_uid != os.getuid():
                return f"Security: .env file not owned by current user: {env_path}"

            # Check permissions (warn or block if group/world readable)
            mode = file_stat.st_mode
            if mode & (stat.S_IRGRP | stat.S_IROTH):
                msg = (
                    f"Security: .env file {env_path} has permissive permissions "
                    f"(mode={oct(mode)}). Use chmod 600 to fix."
                )
                if strict:
                    return msg
                else:
                    logger.warning(msg + " Set strict_env_security=True to block.")
        except OSError as e:
            return f"Cannot stat .env file: {e}"

        return None


__all__ = ["ApiKeyResolver"]
