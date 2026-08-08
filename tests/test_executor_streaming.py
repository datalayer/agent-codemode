# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Tests for streaming execution behavior in CodeModeExecutor."""

from __future__ import annotations

import pytest
from code_sandboxes import ExecutionResult, Logs, OutputMessage
from code_sandboxes.models import Result

from agent_codemode.composition.executor import CodeModeExecutor
from agent_codemode.composition import executor as executor_module
from agent_codemode.discovery.registry import ToolRegistry


class _StreamingClient:
    variant = "jupyter"

    def __init__(self) -> None:
        self.run_code_calls = 0
        self.streaming_called = False

    def execute_code(self, code: str, **kwargs) -> ExecutionResult:
        _ = (code, kwargs.get("timeout"), kwargs.get("language"), kwargs.get("envs"))
        self.run_code_calls += 1
        return ExecutionResult(logs=Logs())

    def execute_code_streaming(self, code: str, **kwargs):
        _ = (code, kwargs.get("timeout"), kwargs.get("language"), kwargs.get("envs"))
        self.streaming_called = True
        yield OutputMessage(line="status: RUNNING", timestamp=0.0, error=False)
        yield OutputMessage(line="hello", timestamp=0.0, error=False)
        yield Result(data={"text/plain": "42"}, is_main_result=True, extra={})


class _FailingStreamingClient:
    variant = "jupyter"

    def __init__(self) -> None:
        self.run_code_calls = 0

    def execute_code(self, code: str, **kwargs) -> ExecutionResult:
        _ = (code, kwargs.get("timeout"), kwargs.get("language"), kwargs.get("envs"))
        self.run_code_calls += 1
        return ExecutionResult(logs=Logs())

    def execute_code_streaming(self, code: str, **kwargs):
        _ = (code, kwargs)
        raise RuntimeError("sandbox unavailable")
        yield


@pytest.mark.asyncio
async def test_execute_uses_streaming_when_supported(monkeypatch):
    monkeypatch.setattr(executor_module, "_get_identity_env", lambda: {})
    executor = CodeModeExecutor(registry=ToolRegistry())
    client = _StreamingClient()
    executor._sandbox_client = client
    executor._setup_done = True

    result = await executor.execute("print('hi')")

    assert client.streaming_called is True
    assert "status: RUNNING" in result.logs.stdout_text
    assert "hello" in result.logs.stdout_text
    assert result.results and result.results[0].data["text/plain"] == "42"


@pytest.mark.asyncio
async def test_execute_reports_streaming_infrastructure_failure(monkeypatch):
    monkeypatch.setattr(executor_module, "_get_identity_env", lambda: {})
    executor = CodeModeExecutor(registry=ToolRegistry())
    client = _FailingStreamingClient()
    executor._sandbox_client = client
    executor._setup_done = True

    result = await executor.execute("print('hi')")

    assert client.run_code_calls >= 2
    assert result.execution_ok is False
    assert result.execution_error == "sandbox unavailable"
