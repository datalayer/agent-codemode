# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""End-to-end tests for skill binding generation and routing.

These tests verify the full pipeline:
1. PythonCodeGenerator.generate_skill_bindings() creates the correct files
2. CodeModeExecutor routes skills__* calls to the skill tool caller
3. CodemodeToolset post-init callbacks fire after lazy initialisation
"""

import asyncio
import importlib
import json
import sys
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_codemode.discovery.codegen import PythonCodeGenerator
from agent_codemode.discovery.registry import ToolRegistry
from agent_codemode.types import CodeModeConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SKILLS_METADATA: list[dict[str, Any]] = [
    {
        "name": "pdf-extractor",
        "description": "Extract content from PDF documents.",
        "scripts": [
            {"name": "extract", "description": "Extract text from a PDF."},
            {"name": "summarize", "description": "Summarize a PDF."},
        ],
        "resources": [
            {"name": "template.txt"},
        ],
    },
    {
        "name": "csv-analyzer",
        "description": "Analyze CSV files.",
        "scripts": [
            {"name": "analyze", "description": "Analyze a CSV."},
        ],
    },
]


@pytest.fixture()
def generated_dir(tmp_path: Path) -> Path:
    """Return a temporary generated/ directory."""
    d = tmp_path / "generated"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# 1. Codegen: generate_skill_bindings produces correct file hierarchy
# ---------------------------------------------------------------------------

class TestGenerateSkillBindings:
    """Test that PythonCodeGenerator.generate_skill_bindings creates
    the expected files under servers/skills/."""

    def test_creates_skill_files(self, generated_dir: Path):
        codegen = PythonCodeGenerator(str(generated_dir))
        codegen.generate_skill_bindings(SAMPLE_SKILLS_METADATA)

        skills_dir = generated_dir / "servers" / "skills"
        assert skills_dir.is_dir()

        expected_files = [
            "__init__.py",
            "list_skills.py",
            "load_skill.py",
            "read_skill_resource.py",
            "run_skill.py",
        ]
        for fname in expected_files:
            assert (skills_dir / fname).is_file(), f"Missing {fname}"

    def test_list_skills_contains_catalog(self, generated_dir: Path):
        codegen = PythonCodeGenerator(str(generated_dir))
        codegen.generate_skill_bindings(SAMPLE_SKILLS_METADATA)

        source = (generated_dir / "servers" / "skills" / "list_skills.py").read_text()
        # The catalog JSON should be embedded
        assert "pdf-extractor" in source
        assert "csv-analyzer" in source

    def test_bindings_import_call_tool(self, generated_dir: Path):
        codegen = PythonCodeGenerator(str(generated_dir))
        codegen.generate_skill_bindings(SAMPLE_SKILLS_METADATA)

        for fname in ["load_skill.py", "read_skill_resource.py", "run_skill.py"]:
            source = (generated_dir / "servers" / "skills" / fname).read_text()
            assert "from ...client import call_tool" in source

    def test_init_exports_all_functions(self, generated_dir: Path):
        codegen = PythonCodeGenerator(str(generated_dir))
        codegen.generate_skill_bindings(SAMPLE_SKILLS_METADATA)

        init_src = (generated_dir / "servers" / "skills" / "__init__.py").read_text()
        for fn in ["list_skills", "load_skill", "read_skill_resource", "run_skill"]:
            assert fn in init_src


# ---------------------------------------------------------------------------
# 2. Executor: skills__* routing through set_skill_tool_caller
# ---------------------------------------------------------------------------

class TestExecutorSkillRouting:
    """Test that CodeModeExecutor routes skills__* calls to the caller."""

    @pytest.mark.asyncio
    async def test_skills_prefix_routes_to_caller(self, tmp_path: Path):
        from agent_codemode.composition.executor import CodeModeExecutor

        registry = ToolRegistry()
        config = CodeModeConfig(
            generated_path=str(tmp_path / "generated"),
            workspace_path=str(tmp_path / "workspace"),
            skills_path=str(tmp_path / "skills"),
        )
        executor = CodeModeExecutor(registry=registry, config=config)

        # Mock skill caller
        mock_caller = AsyncMock(return_value={"result": "ok"})
        executor.set_skill_tool_caller(mock_caller)

        result = await executor.call_tool(
            "skills__list_skills", {}
        )

        mock_caller.assert_awaited_once_with("skills__list_skills", {})
        assert result == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_non_skills_prefix_goes_to_registry(self, tmp_path: Path):
        from agent_codemode.composition.executor import CodeModeExecutor

        registry = ToolRegistry()
        registry.call_tool = AsyncMock(return_value={"mcp": True})

        config = CodeModeConfig(
            generated_path=str(tmp_path / "generated"),
            workspace_path=str(tmp_path / "workspace"),
            skills_path=str(tmp_path / "skills"),
        )
        executor = CodeModeExecutor(registry=registry, config=config)
        executor.set_skill_tool_caller(AsyncMock())

        result = await executor.call_tool("github__star_repo", {"owner": "x"})

        registry.call_tool.assert_awaited_once_with("github__star_repo", {"owner": "x"})
        assert result == {"mcp": True}

    @pytest.mark.asyncio
    async def test_skills_metadata_stored(self, tmp_path: Path):
        from agent_codemode.composition.executor import CodeModeExecutor

        registry = ToolRegistry()
        config = CodeModeConfig(
            generated_path=str(tmp_path / "generated"),
            workspace_path=str(tmp_path / "workspace"),
            skills_path=str(tmp_path / "skills"),
        )
        executor = CodeModeExecutor(registry=registry, config=config)

        assert executor._skills_metadata == []

        executor.set_skills_metadata(SAMPLE_SKILLS_METADATA)
        assert len(executor._skills_metadata) == 2
        assert executor._skills_metadata[0]["name"] == "pdf-extractor"


# ---------------------------------------------------------------------------
# 3. CodemodeToolset post-init callbacks
# ---------------------------------------------------------------------------

class TestCodemodeToolsetPostInit:
    """Test the post_init_callback mechanism on CodemodeToolset."""

    @pytest.mark.asyncio
    async def test_callback_fires_on_ensure_initialized(self, tmp_path: Path):
        """Post-init callback should fire after _ensure_initialized."""
        from agent_codemode.toolset import CodemodeToolset, PYDANTIC_AI_AVAILABLE

        if not PYDANTIC_AI_AVAILABLE:
            pytest.skip("pydantic-ai not installed")

        registry = ToolRegistry()
        config = CodeModeConfig(
            generated_path=str(tmp_path / "generated"),
            workspace_path=str(tmp_path / "workspace"),
            skills_path=str(tmp_path / "skills"),
        )

        toolset = CodemodeToolset(registry=registry, config=config)

        callback_called = []
        toolset.add_post_init_callback(lambda ts: callback_called.append(ts))

        # Trigger lazy init
        await toolset.start()

        assert len(callback_called) == 1
        assert callback_called[0] is toolset

    @pytest.mark.asyncio
    async def test_callback_not_called_twice(self, tmp_path: Path):
        """Callbacks only fire once even if start() is called twice."""
        from agent_codemode.toolset import CodemodeToolset, PYDANTIC_AI_AVAILABLE

        if not PYDANTIC_AI_AVAILABLE:
            pytest.skip("pydantic-ai not installed")

        registry = ToolRegistry()
        config = CodeModeConfig(
            generated_path=str(tmp_path / "generated"),
            workspace_path=str(tmp_path / "workspace"),
            skills_path=str(tmp_path / "skills"),
        )

        toolset = CodemodeToolset(registry=registry, config=config)

        counter = []
        toolset.add_post_init_callback(lambda ts: counter.append(1))

        await toolset.start()
        await toolset.start()  # second call should be a no-op

        assert len(counter) == 1

    @pytest.mark.asyncio
    async def test_callback_error_does_not_prevent_init(self, tmp_path: Path):
        """A failing callback should log an error but not crash init."""
        from agent_codemode.toolset import CodemodeToolset, PYDANTIC_AI_AVAILABLE

        if not PYDANTIC_AI_AVAILABLE:
            pytest.skip("pydantic-ai not installed")

        registry = ToolRegistry()
        config = CodeModeConfig(
            generated_path=str(tmp_path / "generated"),
            workspace_path=str(tmp_path / "workspace"),
            skills_path=str(tmp_path / "skills"),
        )

        toolset = CodemodeToolset(registry=registry, config=config)

        def bad_callback(ts):
            raise RuntimeError("boom")

        ok_called = []
        toolset.add_post_init_callback(bad_callback)
        toolset.add_post_init_callback(lambda ts: ok_called.append(True))

        await toolset.start()  # should not raise

        assert toolset._initialized
        assert ok_called == [True]


# ---------------------------------------------------------------------------
# 4. Full codegen + import round-trip (no sandbox needed)
# ---------------------------------------------------------------------------

class TestCodegenImportRoundTrip:
    """Verify that generated skill binding files can be imported and
    that calling the functions triggers call_tool with the correct
    prefixed tool names."""

    @pytest.mark.asyncio
    async def test_import_and_call_list_skills(self, generated_dir: Path):
        """generate_skill_bindings → import list_skills → call returns catalog."""
        codegen = PythonCodeGenerator(str(generated_dir))
        # Also generate the client module so imports work
        codegen.generate_from_tools({})
        codegen.generate_skill_bindings(SAMPLE_SKILLS_METADATA)

        # Temporarily add generated_dir's parent to sys.path
        parent = str(generated_dir.parent)
        sys.path.insert(0, parent)
        # Clear any cached 'generated' modules
        for mod in list(sys.modules):
            if mod == "generated" or mod.startswith("generated."):
                del sys.modules[mod]

        try:
            from generated.servers.skills.list_skills import list_skills

            result = await list_skills()
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]["name"] == "pdf-extractor"
        finally:
            sys.path.remove(parent)
            for mod in list(sys.modules):
                if mod == "generated" or mod.startswith("generated."):
                    del sys.modules[mod]

    @pytest.mark.asyncio
    async def test_load_skill_calls_call_tool(self, generated_dir: Path):
        """load_skill should invoke call_tool('skills__load_skill', …)."""
        codegen = PythonCodeGenerator(str(generated_dir))
        codegen.generate_from_tools({})
        codegen.generate_skill_bindings(SAMPLE_SKILLS_METADATA)

        parent = str(generated_dir.parent)
        sys.path.insert(0, parent)
        for mod in list(sys.modules):
            if mod == "generated" or mod.startswith("generated."):
                del sys.modules[mod]

        try:
            from generated.client import set_tool_caller
            from generated.servers.skills.load_skill import load_skill

            calls = []

            async def mock_caller(name: str, args: dict) -> Any:
                calls.append((name, args))
                return "# SKILL.md content"

            set_tool_caller(mock_caller)

            result = await load_skill("pdf-extractor")

            assert len(calls) == 1
            assert calls[0][0] == "skills__load_skill"
            assert calls[0][1] == {"skill_name": "pdf-extractor"}
            assert result == "# SKILL.md content"
        finally:
            sys.path.remove(parent)
            for mod in list(sys.modules):
                if mod == "generated" or mod.startswith("generated."):
                    del sys.modules[mod]

    @pytest.mark.asyncio
    async def test_run_skill_calls_call_tool(self, generated_dir: Path):
        """run_skill should invoke call_tool('skills__run_skill_script', …)."""
        codegen = PythonCodeGenerator(str(generated_dir))
        codegen.generate_from_tools({})
        codegen.generate_skill_bindings(SAMPLE_SKILLS_METADATA)

        parent = str(generated_dir.parent)
        sys.path.insert(0, parent)
        for mod in list(sys.modules):
            if mod == "generated" or mod.startswith("generated."):
                del sys.modules[mod]

        try:
            from generated.client import set_tool_caller
            from generated.servers.skills.run_skill import run_skill

            calls = []

            async def mock_caller(name: str, args: dict) -> Any:
                calls.append((name, args))
                return {"output": "done", "exit_code": 0, "success": True}

            set_tool_caller(mock_caller)

            result = await run_skill("pdf-extractor", "extract", ["report.pdf"])

            assert len(calls) == 1
            assert calls[0][0] == "skills__run_skill_script"
            assert calls[0][1]["skill_name"] == "pdf-extractor"
            assert calls[0][1]["script_name"] == "extract"
            assert calls[0][1]["args"] == ["report.pdf"]
        finally:
            sys.path.remove(parent)
            for mod in list(sys.modules):
                if mod == "generated" or mod.startswith("generated."):
                    del sys.modules[mod]
