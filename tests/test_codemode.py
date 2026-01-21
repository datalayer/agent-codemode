# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Unit tests for agent-codemode package."""

import asyncio
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_codemode import CodemodeToolset, CodeModeConfig
from agent_codemode.models import ToolDefinition, SearchResult

from agent_skills.helpers import (
    wait_for,
    retry,
    run_with_timeout,
    parallel,
    RateLimiter,
)
from agent_skills.files import (
    SkillFile,
    SkillDirectory,
    setup_skills_directory,
)


# =============================================================================
# wait_for Tests
# =============================================================================

class TestWaitFor:
    """Tests for wait_for helper."""

    @pytest.mark.asyncio
    async def test_wait_for_immediate_true(self):
        """Test wait_for when condition is immediately true."""
        condition = lambda: True
        await wait_for(condition, interval_seconds=0.1, timeout_seconds=1.0)
        # Should complete without error

    @pytest.mark.asyncio
    async def test_wait_for_becomes_true(self):
        """Test wait_for when condition becomes true after a few checks."""
        counter = [0]

        def condition():
            counter[0] += 1
            return counter[0] >= 3

        await wait_for(condition, interval_seconds=0.05, timeout_seconds=1.0)

        assert counter[0] >= 3

    @pytest.mark.asyncio
    async def test_wait_for_async_condition(self):
        """Test wait_for with async condition."""
        counter = [0]

        async def async_condition():
            counter[0] += 1
            await asyncio.sleep(0.01)
            return counter[0] >= 2

        await wait_for(async_condition, interval_seconds=0.05, timeout_seconds=1.0)

        assert counter[0] >= 2

    @pytest.mark.asyncio
    async def test_wait_for_timeout(self):
        """Test wait_for timeout."""
        condition = lambda: False

        with pytest.raises(TimeoutError):
            await wait_for(
                condition,
                interval_seconds=0.05,
                timeout_seconds=0.2,
                description="test condition",
            )

    @pytest.mark.asyncio
    async def test_wait_for_no_timeout(self):
        """Test wait_for without timeout (should complete when true)."""
        counter = [0]

        def condition():
            counter[0] += 1
            return counter[0] >= 5

        # No timeout - should still work
        await wait_for(condition, interval_seconds=0.01)

        assert counter[0] >= 5


# =============================================================================
# retry Tests
# =============================================================================

class TestRetry:
    """Tests for retry helper."""

    @pytest.mark.asyncio
    async def test_retry_success_first_try(self):
        """Test retry when function succeeds on first try."""
        async def succeed():
            return "success"

        result = await retry(succeed, max_attempts=3)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self):
        """Test retry when function fails then succeeds."""
        attempts = [0]

        async def fail_then_succeed():
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("Not yet")
            return "success"

        result = await retry(
            fail_then_succeed,
            max_attempts=5,
            delay_seconds=0.01,
        )

        assert result == "success"
        assert attempts[0] == 3

    @pytest.mark.asyncio
    async def test_retry_all_failures(self):
        """Test retry when function always fails."""
        async def always_fail():
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            await retry(always_fail, max_attempts=3, delay_seconds=0.01)

    @pytest.mark.asyncio
    async def test_retry_with_backoff(self):
        """Test retry with exponential backoff."""
        import time

        attempts = [0]
        times = []

        async def fail_with_timing():
            times.append(time.time())
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("Not yet")
            return "done"

        result = await retry(
            fail_with_timing,
            max_attempts=3,
            delay_seconds=0.1,
            backoff_factor=2.0,
        )

        assert result == "done"
        # Check that delays increased (with some tolerance)
        if len(times) >= 2:
            first_delay = times[1] - times[0]
            assert first_delay >= 0.09  # ~0.1s

    @pytest.mark.asyncio
    async def test_retry_with_on_retry_callback(self):
        """Test retry with on_retry callback."""
        attempts = [0]
        callbacks = []

        async def fail_then_succeed():
            attempts[0] += 1
            if attempts[0] < 2:
                raise ValueError("Fail")
            return "done"

        def on_retry(exc, attempt):
            callbacks.append((str(exc), attempt))

        await retry(
            fail_then_succeed,
            max_attempts=3,
            delay_seconds=0.01,
            on_retry=on_retry,
        )

        assert len(callbacks) == 1
        assert "Fail" in callbacks[0][0]

    @pytest.mark.asyncio
    async def test_retry_specific_exceptions(self):
        """Test retry only catches specific exceptions."""
        attempts = [0]

        async def raise_type_error():
            attempts[0] += 1
            if attempts[0] < 2:
                raise TypeError("Wrong type")
            return "done"

        # Should not retry TypeError if we only catch ValueError
        with pytest.raises(TypeError):
            await retry(
                raise_type_error,
                max_attempts=3,
                delay_seconds=0.01,
                exceptions=(ValueError,),
            )


# =============================================================================
# run_with_timeout Tests
# =============================================================================

class TestRunWithTimeout:
    """Tests for run_with_timeout helper."""

    @pytest.mark.asyncio
    async def test_run_with_timeout_completes(self):
        """Test run_with_timeout when function completes in time."""
        async def fast_func():
            await asyncio.sleep(0.05)
            return "done"

        result = await run_with_timeout(fast_func, timeout_seconds=1.0)
        assert result == "done"

    @pytest.mark.asyncio
    async def test_run_with_timeout_exceeds(self):
        """Test run_with_timeout when function exceeds timeout."""
        async def slow_func():
            await asyncio.sleep(10.0)
            return "done"

        with pytest.raises(TimeoutError):
            await run_with_timeout(slow_func, timeout_seconds=0.1)


# =============================================================================
# parallel Tests
# =============================================================================

class TestParallel:
    """Tests for parallel helper."""

    @pytest.mark.asyncio
    async def test_parallel_basic(self):
        """Test running coroutines in parallel."""
        async def task(n):
            await asyncio.sleep(0.01)
            return n * 2

        results = await parallel(
            lambda: task(1),
            lambda: task(2),
            lambda: task(3),
        )

        assert results == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_parallel_empty(self):
        """Test parallel with no coroutines."""
        results = await parallel()
        assert results == []

    @pytest.mark.asyncio
    async def test_parallel_preserves_order(self):
        """Test that parallel preserves result order."""
        async def task(n, delay):
            await asyncio.sleep(delay)
            return n

        # First task is slower but should still be first in results
        results = await parallel(
            lambda: task(1, 0.1),
            lambda: task(2, 0.01),
            lambda: task(3, 0.05),
        )

        assert results == [1, 2, 3]


# =============================================================================
# RateLimiter Tests
# =============================================================================

class TestRateLimiter:
    """Tests for RateLimiter class."""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_within_limit(self):
        """Test that rate limiter allows requests within limit."""
        limiter = RateLimiter(calls_per_second=100)

        # Should not block for a few requests
        for _ in range(5):
            await limiter.acquire()
        # If we got here without hanging, test passes

    @pytest.mark.asyncio
    async def test_rate_limiter_throttles(self):
        """Test that rate limiter throttles excessive requests."""
        import time

        limiter = RateLimiter(calls_per_second=10)

        start = time.time()

        # Make enough requests to trigger throttling
        for _ in range(12):
            await limiter.acquire()

        elapsed = time.time() - start

        # Should have taken at least 0.1s (10 requests per second)
        assert elapsed >= 0.1
        assert elapsed >= 0.1


# =============================================================================
# SkillFile Tests (agent_codemode version)
# =============================================================================

class TestMcpCodemodeSkillFile:
    """Tests for SkillFile in agent_codemode."""

    def test_from_file(self, tmp_path: Path):
        """Test creating SkillFile from a file."""
        skill_code = '''"""Process data."""

async def process():
    return "processed"
'''
        skill_file = tmp_path / "process.py"
        skill_file.write_text(skill_code)

        skill = SkillFile.from_file(skill_file, tmp_path)

        assert skill.name == "process"
        assert "process" in skill.functions


# =============================================================================
# SkillDirectory Tests (agent_codemode version)
# =============================================================================

class TestMcpCodemodeSkillDirectory:
    """Tests for SkillDirectory in agent_codemode."""

    def test_create_and_list(self, tmp_path: Path):
        """Test creating and listing skills."""
        skills = SkillDirectory(str(tmp_path))

        skills.create(
            name="test_skill",
            code='async def test_skill(): return "test"',
            description="A test",
        )

        listed = skills.list()
        assert len(listed) == 1
        assert listed[0].name == "test_skill"

    def test_search(self, tmp_path: Path):
        """Test searching skills."""
        skills = SkillDirectory(str(tmp_path))

        skills.create(name="csv_reader", code='async def csv_reader(): pass')
        skills.create(name="json_writer", code='async def json_writer(): pass')

        results = skills.search("csv")
        assert any(s.name == "csv_reader" for s in results)


# =============================================================================
# Integration Tests
# =============================================================================

class TestMcpCodemodeIntegration:
    """Integration tests for agent-codemode."""

    @pytest.mark.asyncio
    async def test_skill_with_helpers(self, tmp_path: Path):
        """Test a skill that uses helper utilities."""
        skills = SkillDirectory(str(tmp_path))

        # Create a skill that uses wait_for pattern internally
        skills.create(
            name="polling_skill",
            code='''
async def polling_skill():
    counter = [0]
    
    def check():
        counter[0] += 1
        return counter[0] >= 3
    
    # Simple polling without importing wait_for
    import asyncio
    while not check():
        await asyncio.sleep(0.01)
    
    return {"checks": counter[0]}
''',
        )

        skill = skills.get("polling_skill")
        func = skill.get_function()
        result = await func()

        assert result["checks"] >= 3

    @pytest.mark.asyncio
    async def test_retry_in_skill(self, tmp_path: Path):
        """Test a skill that implements retry logic."""
        skills = SkillDirectory(str(tmp_path))

        skills.create(
            name="retry_skill",
            code='''
async def retry_skill():
    attempts = 0
    
    async def flaky_operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError("Temporary failure")
        return "success"
    
    # Simple retry without importing retry helper
    for i in range(5):
        try:
            result = await flaky_operation()
            return {"result": result, "attempts": attempts}
        except ValueError:
            import asyncio
            await asyncio.sleep(0.01)
    
    raise RuntimeError("All retries failed")
''',
        )

        skill = skills.get("retry_skill")
        func = skill.get_function()
        result = await func()

        assert result["result"] == "success"
        assert result["attempts"] == 3


# =============================================================================
# CodemodeToolset Policy and Reranker Tests
# =============================================================================


class _FakeRegistry:
    def __init__(self):
        self.tools = [
            ToolDefinition(
                name="a__one",
                description="first",
                server_name="a",
                output_schema={"type": "object", "properties": {"value": {"type": "string"}}},
                input_examples=[{"value": "example"}],
                defer_loading=False,
            ),
            ToolDefinition(
                name="b__two",
                description="second",
                server_name="b",
                output_schema={"type": "string"},
                input_examples=[],
                defer_loading=True,
            ),
        ]

    def list_tools(
        self,
        server: str | None = None,
        limit: int | None = None,
        include_deferred: bool = False,
    ):
        tools = [t for t in self.tools if not server or t.server_name == server]
        if not include_deferred:
            tools = [t for t in tools if not t.defer_loading]
        return tools[:limit] if limit else tools

    async def search_tools(
        self,
        query: str,
        server: str | None = None,
        limit: int = 10,
        include_deferred: bool = True,
    ):
        tools = [t for t in self.tools if not server or t.server_name == server]
        if not include_deferred:
            tools = [t for t in tools if not t.defer_loading]
        return SearchResult(tools=tools, total=len(tools), query=query)

    def get_tool(self, name: str):
        return next((t for t in self.tools if t.name == name), None)

    async def list_servers(self):
        return []

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]):
        return {"called": tool_name, "args": arguments}


@pytest.mark.asyncio
async def test_call_tool_hidden_when_disabled():
    toolset = CodemodeToolset(
        registry=_FakeRegistry(),
        config=CodeModeConfig(allow_direct_tool_calls=False),
        allow_direct_tool_calls=False,
    )

    tools = await toolset.get_tools(ctx=MagicMock())
    assert "call_tool" not in tools

    with pytest.raises(ValueError):
        await toolset.call_tool(
            name="call_tool",
            tool_args={"tool_name": "a__one", "arguments": {}},
            ctx=MagicMock(),
            tool=MagicMock(),
        )


@pytest.mark.asyncio
async def test_list_tool_names_keywords_and_limit():
    toolset = CodemodeToolset(registry=_FakeRegistry())

    result = await toolset._list_tool_names(keywords=["first"], limit=1)

    assert result["tools"] == ["a__one"]
    assert result["total"] == 1
    assert result["returned"] == 1
    assert result["truncated"] is False


@pytest.mark.asyncio
async def test_list_tool_names_excludes_deferred_by_default():
    toolset = CodemodeToolset(registry=_FakeRegistry())

    result = await toolset._list_tool_names()

    assert result["tools"] == ["a__one"]
    assert result["returned"] == 1


@pytest.mark.asyncio
async def test_search_tools_includes_deferred_by_default():
    toolset = CodemodeToolset(registry=_FakeRegistry())

    result = await toolset._search_tools(query="anything")

    names = [t["name"] for t in result["tools"]]
    assert names == ["a__one", "b__two"]
    assert result["tools"][1]["defer_loading"] is True


@pytest.mark.asyncio
async def test_search_tools_reranker_applied():
    async def reranker(tools, query, server):
        return list(reversed(tools))

    toolset = CodemodeToolset(
        registry=_FakeRegistry(),
        tool_reranker=reranker,
    )

    result = await toolset._search_tools(query="anything", server=None, limit=10)

    names = [t["name"] for t in result["tools"]]
    assert names == ["b__two", "a__one"]
    assert result["tools"][0]["output_schema"] == {"type": "string"}
    assert result["tools"][1]["input_examples"] == [{"value": "example"}]


@pytest.mark.asyncio
async def test_get_tool_details_includes_examples():
    toolset = CodemodeToolset(registry=_FakeRegistry())

    result = await toolset._get_tool_details("a__one")

    assert result["output_schema"] == {"type": "object", "properties": {"value": {"type": "string"}}}
    assert result["input_examples"] == [{"value": "example"}]
