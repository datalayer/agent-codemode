"""Test script to verify stdout capture in async code."""

import asyncio
from agent_codemode.composition.executor import CodeModeExecutor
from agent_codemode.discovery.registry import ToolRegistry


async def test_async_stdout():
    """Test that async code stdout is captured."""
    registry = ToolRegistry()
    executor = CodeModeExecutor(registry=registry)
    await executor.setup()
    
    # Test code that prints to stdout in async context
    code = """
import random
import asyncio

async def generate_text():
    text = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=10))
    print(f"Random text: {text}")
    return text

result = await generate_text()
"""
    
    result = await executor.execute(code)
    
    print(f"Execution ok: {result.execution_ok}")
    print(f"Execution error: {result.execution_error}")
    print(f"Code error: {result.code_error}")
    print(f"Stdout text: '{result.logs.stdout_text}'")
    print(f"Stderr text: '{result.logs.stderr_text}'")
    print(f"Stdout messages: {result.logs.stdout}")
    print(f"Stderr messages: {result.logs.stderr}")
    
    # Verify stdout was captured
    assert result.success, f"Execution failed: {result.code_error}"
    assert "Random text:" in result.logs.stdout_text, f"stdout not captured: {result.logs.stdout_text}"
    print("\n✅ SUCCESS: stdout was captured correctly!")


if __name__ == "__main__":
    asyncio.run(test_async_stdout())
