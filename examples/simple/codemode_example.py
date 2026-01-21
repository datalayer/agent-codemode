#!/usr/bin/env python
# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

"""Example: Using agent-codemode for Code-First Tool Composition.

This example demonstrates how to use agent-codemode to:
1. Discover tools from MCP servers
2. Generate Python bindings for tools
3. Execute code that composes multiple tools
4. Save reusable tool compositions as skills

Key Concept: Code Mode
Instead of calling tools one-by-one through LLM inference, Code Mode
allows agents to write Python code that orchestrates multiple tool calls.
This is more efficient and allows for complex logic, error handling,
and parallel execution.

Based on:
- Cloudflare Code Mode: https://blog.cloudflare.com/introducing-code-mode
- Anthropic Programmatic Tool Calling
"""

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


async def example_tool_discovery():
    """Example 1: Progressive Tool Discovery.
    
    Instead of loading all tools upfront, use the Tool Search Tool
    to discover relevant tools based on the task at hand.
    """
    from agent_codemode import ToolRegistry, MCPServerConfig
    
    logger.debug("=" * 60)
    logger.debug("Example 1: Progressive Tool Discovery")
    logger.debug("=" * 60)
    
    # Create a registry and add MCP servers
    registry = ToolRegistry()
    
    # Add an example server (filesystem operations)
    # In production, this would be a real MCP server
    registry.add_server(MCPServerConfig(
        name="filesystem",
        command="npx",
        args=["-y", "@anthropic/mcp-server-filesystem", "/tmp"],
    ))
    
    # Discover all tools from configured servers
    logger.debug("\nDiscovering tools from MCP servers...")
    await registry.discover_all()
    
    # List all available tools
    all_tools = registry.list_tools()
    logger.debug("Discovered %s tools", len(all_tools))
    
    # Search for specific tools (progressive discovery)
    logger.debug("\nSearching for 'file' operations...")
    result = await registry.search_tools("file operations", limit=5)
    
    for tool in result.tools:
        logger.debug("  - %s: %s", tool.name, tool.description)
    
    return registry


async def example_code_execution():
    """Example 2: Code-Based Tool Composition.
    
    Execute Python code that calls multiple tools. The code runs
    in an isolated sandbox with generated Python bindings for all tools.
    """
    from agent_codemode import ToolRegistry, CodeModeExecutor, CodeModeConfig
    
    logger.debug("\n" + "=" * 60)
    logger.debug("Example 2: Code-Based Tool Composition")
    logger.debug("=" * 60)
    
    # Set up the registry
    registry = ToolRegistry()
    
    # Configure the executor
    config = CodeModeConfig(
        sandbox_variant="local-eval",  # For development
        generated_path="./generated",
        skills_path="./skills",
    )
    
    # Use the executor as an async context manager
    async with CodeModeExecutor(registry, config) as executor:
        
        # Example: Execute code that would call tools
        # (This is a simplified example - real code would import generated bindings)
        code = '''
# This code runs in an isolated sandbox
import os

# In production, you would import generated tool bindings:
# from generated.servers.filesystem import read_file, write_file

# Create a sample data processing workflow
data = {"files_processed": 0, "total_size": 0}

# Simulated file processing
for filename in ["file1.txt", "file2.txt", "file3.txt"]:
    data["files_processed"] += 1
    data["total_size"] += 100  # Simulated file size

# Return the result
result = f"Processed {data['files_processed']} files, total size: {data['total_size']} bytes"
print(result)
'''
        
        logger.debug("\nExecuting code in sandbox...")
        execution = await executor.execute(code)
        
        if execution.error:
            logger.debug("Error: %s", execution.error)
        else:
            logger.debug(
                "Output:\n%s",
                execution.logs.stdout if execution.logs else "No output",
            )
        
        # Show tool call history
        logger.debug("\nTool calls made: %s", len(executor.tool_call_history))


async def _server():
    """Example 4: Running the Codemode MCP Server.
    
    Shows how to configure and run the Codemode MCP server
    which exposes code execution capabilities to AI agents.
    """
    logger.debug("\n" + "=" * 60)
    logger.debug("Example 4: Codemode MCP Server")
    logger.debug("=" * 60)
    
    logger.debug("""
The Codemode MCP Server exposes these tools to AI agents:

1. search_tools(query) - Progressive tool discovery
   Find relevant tools based on natural language description.

2. execute_code(code) - Code-based tool composition
   Run Python code that can call multiple tools.

To start the server:

    from agent_codemode import codemode_server, configure_server
    from agent_codemode import MCPServerConfig
    
    # Configure with MCP servers
    configure_server()
    
    # Run the MCP server (uses FastMCP under the hood)
    codemode_server.run()

Or from the command line:

    python -m codemode.server
""")


async def main():
    """Run all examples."""
    logger.debug("\n" + "=" * 60)
    logger.debug("Agent Codemode Examples")
    logger.debug("=" * 60)
    
    # Run examples
    await example_tool_discovery()
    await example_code_execution()
    await _server()
    
    logger.debug("\n" + "=" * 60)
    logger.debug("Examples Complete!")
    logger.debug("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
