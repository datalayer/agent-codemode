# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Tests for streaming execution behavior in CodeModeExecutor."""

from __future__ import annotations

import pytest
from code_sandboxes import ExecutionResult, Logs, OutputMessage
from code_sandboxes.models import Result

from agent_codemode.composition.executor import CodeModeExecutor
from agent_codemode.discovery.registry import ToolRegistry


class _StreamingSandbox:
    def __init__(self) -> None:
        self.run_code_calls = 0
        self.streaming_called = False

    def run_code(self, code: str, timeout=None) -> ExecutionResult:
        _ = (code, timeout)
        self.run_code_calls += 1
        return ExecutionResult(logs=Logs())

    def run_code_streaming(self, code: str, timeout=None):
        _ = (code, timeout)
        self.streaming_called = True
        yield OutputMessage(line="status: RUNNING", timestamp=0.0, error=False)
        yield OutputMessage(line="hello", timestamp=0.0, error=False)
        yield Result(data={"text/plain": "42"}, is_main_result=True, extra={})


class _NonStreamingSandbox:
    def __init__(self) -> None:
        self.run_code_calls = 0

    def run_code(self, code: str, timeout=None) -> ExecutionResult:
        _ = (code, timeout)
        self.run_code_calls += 1
        if self.run_code_calls >= 3:
            return ExecutionResult(
                logs=Logs(stdout=[OutputMessage(line="fallback", timestamp=0.0, error=False)]),
            )
        return ExecutionResult(logs=Logs())


@pytest.mark.asyncio
async def test_execute_uses_streaming_when_supported():
    executor = CodeModeExecutor(registry=ToolRegistry())
    sandbox = _StreamingSandbox()
    executor._sandbox = sandbox
    executor._setup_done = True

    result = await executor.execute("print('hi')")

    assert sandbox.streaming_called is True
    assert "status: RUNNING" in result.logs.stdout_text
    assert "hello" in result.logs.stdout_text
    assert result.results and result.results[0].data["text/plain"] == "42"


@pytest.mark.asyncio
async def test_execute_falls_back_to_run_code_without_streaming():
    executor = CodeModeExecutor(registry=ToolRegistry())
    sandbox = _NonStreamingSandbox()
    executor._sandbox = sandbox
    executor._setup_done = True

    result = await executor.execute("print('hi')")

    assert sandbox.run_code_calls >= 3
    assert result.logs.stdout_text == "fallback"
