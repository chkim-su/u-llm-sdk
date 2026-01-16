"""Test configuration for u-llm-sdk."""
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from u_llm_sdk.types import Provider


@pytest.fixture
def sample_prompt():
    """Sample prompt for testing."""
    return "Hello, world!"


@pytest.fixture
def sample_provider():
    """Sample provider for testing."""
    return Provider.CLAUDE


@pytest.fixture
def sample_model():
    """Sample model for testing."""
    return "claude-sonnet-4-20250514"
