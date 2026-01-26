#!/usr/bin/env python3
"""Direct test of executor without agent."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent_codemode import CodeModeExecutor, ToolRegistry
from agent_codemode.types import MCPServerConfig, CodeModeConfig

async def main():
    print("=== Setting up registry ===")
    registry = ToolRegistry()
    
    # Add example MCP server
    server_path = Path(__file__).parent / "examples" / "agent" / "example_mcp_server.py"
    server_config = MCPServerConfig(
        name="example_mcp",
        command="mcp-run",
        args=[str(server_path)],
    )
    registry.add_server(server_config)
    await registry.discover_all()
    
    tools = registry.list_tools()
    print(f"Found {len(tools)} tools: {[t.name for t in tools]}")
    
    print("\n=== Setting up executor ===")
    config = CodeModeConfig(
        workspace_path=Path(__file__).parent / "examples" / "agent",
        sandbox_variant="local-eval"
    )
    executor = CodeModeExecutor(registry=registry, config=config)
    await executor.setup()
    print("Executor setup complete")
    
    print("\n=== Executing code ===")
    code = """
from generated.servers.example_mcp import generate_random_text

try:
    result = await generate_random_text({'word_count': 50})
    print(f"SUCCESS: Generated {result['word_count']} words")
except Exception as e:
    print(f"ERROR: {e}")
"""
    
    print(f"Code to execute:\n{code}\n")
    execution = await executor.execute(code, timeout=10.0)
    
    print(f"\n=== Results ===")
    print(f"Error: {execution.error}")
    print(f"Stdout:\n{execution.stdout}")
    print(f"Stderr:\n{execution.stderr}")
    
    await executor.cleanup()
    print("\n=== Done ===")

if __name__ == "__main__":
    asyncio.run(main())
