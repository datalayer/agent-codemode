#!/usr/bin/env python3
"""Minimal test for agent with fixed sandbox."""

import asyncio
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from agent_codemode import CodeModeExecutor, ToolRegistry
from agent_codemode.models import MCPServerConfig, CodeModeConfig

async def main():
    print("=== Testing CodeModeExecutor with fixed sandbox ===\n")
    
    # Set up registry with example MCP server
    registry = ToolRegistry()
    server_config = MCPServerConfig(
        name="example_mcp",
        command="mcp-run",
        args=["./example_server.py"],
        cwd=str(Path(__file__).parent / "examples" / "agent")
    )
    registry.add_server(server_config)
    await registry.discover_all()
    
    print(f"Registered tools: {[tool.name for tool in registry.list_tools()]}\n")
    
    # Create executor
    config = CodeModeConfig(
        workspace_path=Path(__file__).parent / "examples" / "agent",
        sandbox_variant="local-eval"
    )
    executor = CodeModeExecutor(registry=registry, config=config)
    await executor.setup()
    
    # Test code execution
    code = """
from generated.servers.example_mcp import generate_random_text

try:
    result = await generate_random_text({'word_count': 50})
    print(f"Generated {result['word_count']} words")
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
"""
    
    print("=== Executing code ===")
    print(code)
    print("\n=== Output ===")
    
    execution = await executor.execute(code)
    
    print(f"Stdout: {execution.stdout}")
    print(f"Stderr: {execution.stderr}")
    print(f"Error: {execution.error}")
    print(f"Success: {not execution.error}")
    
    await executor.cleanup()
    print("\n=== Test completed ===")

if __name__ == "__main__":
    asyncio.run(main())
