#!/usr/bin/env python
# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

"""Example: Code Mode Patterns.

This example demonstrates the key patterns from agent-codemode-claude-poc:

1. Progressive Tool Discovery
   - list_tool_names: Fast listing when simple filtering works
   - search_tools: AI-powered tool discovery with full definitions
   - get_tool_definition: Get schema for a specific tool

2. Programmatic Tool Composition
   - Write Python code that imports generated tool bindings
   - Execute code in sandbox to call tools directly
   - Avoid LLM inference overhead for multi-step operations

3. Helper Utilities
    - wait_for: Wait for async conditions with polling
    - retry: Retry operations on failure with backoff
    - parallel: Run multiple async operations concurrently

Key Insight from Code Mode:
Instead of calling many tools one-by-one through LLM inference, agents
write code that orchestrates multiple tool calls. This is:
- More efficient (fewer LLM calls)
- More reliable (exact data handling, no LLM "fixing" mistakes)
- More powerful (loops, conditionals, error handling)

Based on:
- agent-codemode-claude-poc TypeScript implementation
- Cloudflare Code Mode: https://blog.cloudflare.com/introducing-code-mode
- Anthropic Programmatic Tool Calling
"""

import asyncio
import logging


# =============================================================================
# Example 1: Progressive Tool Discovery (Meta-Tool Pattern)
# =============================================================================

async def example_meta_tools():
    """Demonstrate the meta-tool proxy pattern.
    
    The agent uses 4 meta-tools:
    1. list_tool_names - Fast listing of tool names
    2. search_tools - AI-powered tool discovery
    3. get_tool_definition - Get full schema for a tool
    4. execute_code - Run Python code in sandbox
    
    All actual tool execution goes through execute_code, which runs
    Python code using the generated tool bindings.
    """
    from agent_codemode import ToolRegistry
    from agent_codemode.proxy.meta_tools import MetaToolProvider

    logger = logging.getLogger(__name__)
    
    logger.debug("=" * 70)
    logger.debug("Example 1: Meta-Tool Proxy Pattern")
    logger.debug("=" * 70)
    
    # Create registry with mock tools for demonstration
    registry = ToolRegistry()
    
    # In production, you'd add real MCP servers:
    # registry.add_server(MCPServerConfig(name="filesystem", ...))
    # await registry.discover_all()
    
    # Create the meta-tool provider
    provider = MetaToolProvider(registry)
    
    # Get the meta-tool schemas (these are what the agent sees)
    meta_tools = provider.get_meta_tools()
    logger.debug("\nMeta-tools available to agent:")
    for tool in meta_tools:
        logger.debug("  - %s: %s...", tool["name"], tool["description"][:60])
    
    # Example: Fast tool name listing
    logger.debug("\n1. list_tool_names (fast, no full schemas):")
    result = provider.list_tool_names(keywords=["file", "read"], limit=10)
    logger.debug(
        "   Found %s tools, returned %s",
        result["total"],
        result["returned"],
    )
    
    # Example: AI-powered search (if AI selector configured)
    logger.debug("\n2. search_tools (with full schemas):")
    result = await provider.search_tools("read CSV files and analyze data", limit=5)
    logger.debug("   Found %s tools", result["total"])
    for tool in result.get("tools", []):
        logger.debug(
            "   - %s: %s...",
            tool["name"],
            tool.get("description", "")[:50],
        )
    
    # Example: Get specific tool definition
    logger.debug("\n3. get_tool_definition:")
    # result = await provider.get_tool_definition("filesystem__read_file")
    # print(f"   {result}")


# =============================================================================
# Example 2: Code Execution (Tools as Code Pattern)
# =============================================================================

async def example_code_execution():
    """Demonstrate code-based tool composition.
    
    The agent writes Python code that:
    1. Imports from generated tool bindings
    2. Calls multiple tools with regular Python
    3. Uses loops, conditionals, error handling
    4. Returns results
    
    This avoids LLM inference for each tool call!
    """
    logger.debug("\n" + "=" * 70)
    logger.debug("Example 2: Code-Based Tool Composition")
    logger.debug("=" * 70)
    
    # Example code the agent would write and execute
    example_code = '''
# The agent writes code like this:
from generated.mcp.filesystem import read_file, write_file, list_directory

async def process_files():
    """Process multiple files efficiently."""
    
    # List files in a directory
    entries = await list_directory({"path": "/tmp/data"})
    
    results = []
    for entry in entries.get("entries", []):
        if entry.endswith(".csv"):
            # Read each CSV file
            content = await read_file({"path": f"/tmp/data/{entry}"})
            
            # Process it (in code, not LLM!)
            lines = content.split("\\n")
            row_count = len(lines)
            
            # Save result
            results.append({
                "file": entry,
                "rows": row_count,
            })
    
    # Write summary
    import json
    await write_file({
        "path": "/tmp/summary.json",
        "content": json.dumps(results, indent=2)
    })
    
    return results

# Execute
await process_files()
'''
    
    logger.debug("\nExample code the agent would write:")
    logger.debug("-" * 60)
    logger.debug(example_code)
    logger.debug("-" * 60)
    
    logger.debug("\nBenefits of Code Mode:")
    logger.debug("  ✓ One LLM call generates code that does many tool calls")
    logger.debug("  ✓ No LLM inference between each tool call")
    logger.debug("  ✓ Exact data handling (no LLM 'fixing' mistakes")
    logger.debug("  ✓ Complex logic with loops, conditionals, error handling")
    logger.debug("  ✓ Parallel execution with asyncio.gather")


async def main():
    """Run all examples."""
    logger.debug("\n" + "=" * 70)
    logger.debug("Agent Codemode Patterns")
    logger.debug("=" * 70)

    await example_meta_tools()
    await example_code_execution()

    logger.debug("\n" + "=" * 70)
    logger.debug("Examples Complete!")
    logger.debug("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
