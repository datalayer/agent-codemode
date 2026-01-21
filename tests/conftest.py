# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Pytest configuration and fixtures for agent-codemode tests."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    """Create a temporary skills directory."""
    skills_path = tmp_path / "skills"
    skills_path.mkdir()
    (skills_path / "__init__.py").write_text('"""Test skills."""\n')
    return skills_path


@pytest.fixture
def sample_skill_file(skills_dir: Path) -> Path:
    """Create a sample skill file."""
    skill_code = '''"""Sample skill for testing."""

async def sample_skill(value: int) -> int:
    """Double the value."""
    return value * 2
'''
    skill_file = skills_dir / "sample_skill.py"
    skill_file.write_text(skill_code)
    return skill_file
