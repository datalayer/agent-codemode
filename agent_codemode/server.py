# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""MCP Server for Codemode - Code-First Tool Composition.

This module provides an MCP server that implements the "Code Mode" pattern
inspired by Cloudflare's approach: instead of calling many tools individually,
agents write code that composes tools programmatically.

Key features:
- Tool Search Tool: Progressive tool discovery for large tool catalogs
- Code Execution: Execute Python code that calls tools
- Skills: Save and reuse code-based tool compositions
- Programmatic Tool Calling: Tools marked for code-based invocation

Based on:
- Cloudflare Code Mode: https://blog.cloudflare.com/introducing-code-mode
- Anthropic Programmatic Tool Calling
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from .composition.executor import CodeModeExecutor
from .discovery.registry import ToolRegistry
from .models import CodeModeConfig, SearchResult, Skill

logger = logging.getLogger(__name__)

# Create the MCP server
mcp = FastMCP("Codemode MCP Server 🚀")

# Global instances (configured at startup)
_registry: Optional[ToolRegistry] = None
_executor: Optional[CodeModeExecutor] = None
_config: Optional[CodeModeConfig] = None


def configure(
    config: Optional[CodeModeConfig] = None,
    registry: Optional[ToolRegistry] = None,
) -> None:
    """Configure the Codemode MCP server.
    
    Args:
        config: Configuration for the server.
        registry: Optional pre-configured tool registry.
    """
    global _registry, _executor, _config
    
    _config = config or CodeModeConfig()
    _registry = registry or ToolRegistry()
    _executor = CodeModeExecutor(_registry, _config)
    
    logger.debug("Codemode MCP server configured")


def get_registry() -> ToolRegistry:
    """Get the tool registry."""
    global _registry
    if _registry is None:
        configure()
    return _registry


def get_executor() -> CodeModeExecutor:
    """Get the code executor."""
    global _executor
    if _executor is None:
        configure()
    return _executor


# =============================================================================
# Tool Search Tool - Progressive Discovery
# =============================================================================

@mcp.tool()
async def search_tools(
    query: str,
    server: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 10,
    include_deferred: bool = True,
) -> dict[str, Any]:
    """Search for available tools matching a query.
    
    This is the Tool Search Tool - use it to discover relevant tools
    before deciding which ones to use. Especially useful when there
    are many tools available (100+).
    
    Instead of loading all tool definitions upfront, this allows
    progressive discovery of relevant tools based on your task.
    
    Args:
        query: Natural language description of what you're looking for.
               Examples: "file operations", "data analysis", "web scraping"
        server: Optional filter by MCP server name.
        category: Optional filter by category (e.g., "filesystem", "network").
        limit: Maximum number of results to return (default: 10).
        include_deferred: Whether to include tools marked defer_loading.
    
    Returns:
        Dictionary with:
        - tools: List of matching tools with name, description, server
        - total: Total number of matches (may be more than returned)
        - has_more: Whether there are more results available
        
    Example:
        # Find tools for working with files
        result = search_tools("read and write files")
        # Returns: {"tools": [{"name": "fs__read_file", ...}], "total": 5}
    """
    registry = get_registry()
    result = await registry.search_tools(
        query, server=server, limit=limit, include_deferred=include_deferred
    )
    
    # Filter by category if specified
    tools = result.tools
    if category:
        tools = [t for t in tools if category.lower() in (t.description or "").lower()]
    
    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "server": t.server_name,
                "input_schema": t.input_schema,
                "output_schema": t.output_schema,
                "input_examples": t.input_examples[:2],
                "defer_loading": t.defer_loading,
            }
            for t in tools[:limit]
        ],
        "total": result.total,
        "has_more": result.total > limit,
    }


@mcp.tool()
async def list_servers() -> dict[str, Any]:
    """List all connected MCP servers.
    
    Returns information about all MCP servers that are currently
    connected and available for tool discovery.
    
    Returns:
        Dictionary with:
        - servers: List of server info (name, description, tool_count)
        - total: Total number of servers
    """
    registry = get_registry()
    servers = await registry.list_servers()
    
    return {
        "servers": [
            {
                "name": s.name,
                "description": s.description,
                "tool_count": s.tool_count,
            }
            for s in servers
        ],
        "total": len(servers),
    }


@mcp.tool()
async def get_tool_details(tool_name: str) -> dict[str, Any]:
    """Get detailed information about a specific tool.
    
    After finding a tool with search_tools, use this to get
    the full schema and usage information.
    
    Args:
        tool_name: The full tool name (format: server__toolname).
    
    Returns:
        Dictionary with full tool definition including:
        - name: Tool name
        - description: Full description
        - input_schema: JSON Schema for parameters
        - examples: Usage examples if available
    """
    registry = get_registry()
    tool = registry.get_tool(tool_name)
    
    if tool is None:
        return {"error": f"Tool not found: {tool_name}"}
    
    return {
        "name": tool.name,
        "description": tool.description,
        "server": tool.server_name,
        "input_schema": tool.input_schema,
        "output_schema": tool.output_schema,
        "input_examples": tool.input_examples,
        "defer_loading": tool.defer_loading,
    }


# =============================================================================
# Code Execution - The Core of Code Mode
# =============================================================================

@mcp.tool()
async def execute_code(
    code: str,
    timeout: Optional[float] = 30.0,
    context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Execute Python code that can compose and call tools.
    
    This is the core of Code Mode - instead of calling tools one by one,
    write Python code that orchestrates multiple tool calls efficiently.
    
    The code runs in an isolated sandbox with:
    - Access to all discovered tools as Python functions
    - Async/await support for parallel tool calls
    - State persistence between calls
    - Error handling and result capture
    
    Benefits of Code Mode:
    - Reduce LLM calls for multi-step operations
    - Better error handling with try/except
    - Parallel execution with asyncio.gather
    - Complex logic with loops and conditionals
    
    Args:
        code: Python code to execute. Can use `await` for async operations.
              Import tools from `generated.servers.<server_name>`.
        timeout: Maximum execution time in seconds (default: 30).
        context: Optional variables to inject into the execution context.
    
    Returns:
        Dictionary with:
        - success: Whether execution completed without errors
        - result: The result of the last expression
        - output: Captured stdout/stderr
        - execution_time: Time taken in seconds
        - error: Error message if execution failed
        
    Example:
        # Read multiple files in parallel
        code = '''
        import asyncio
        from generated.servers.filesystem import read_file
        
        files = ["/path/file1.txt", "/path/file2.txt"]
        results = await asyncio.gather(*[read_file({"path": f}) for f in files])
        '''
        result = execute_code(code)
    """
    executor = get_executor()
    
    # Ensure executor is set up
    if not executor._setup_done:
        await executor.setup()
    
    # Inject context variables if provided
    if context and executor._sandbox:
        for name, value in context.items():
            executor._sandbox.set_variable(name, value)
    
    try:
        execution = await executor.execute(code, timeout=timeout)
        
        return {
            "success": not execution.error,
            "result": execution.text,
            "results": [
                {
                    "data": r.data,
                    "is_main_result": r.is_main_result,
                    "extra": r.extra,
                }
                for r in execution.results
            ],
            "stdout": execution.stdout,
            "stderr": execution.stderr,
            "output": execution.stdout,
            "execution_time": execution.execution_time or 0,
            "error": str(execution.error) if execution.error else None,
        }
    except Exception as e:
        logger.debug("Code execution failed", exc_info=e)
        return {
            "success": False,
            "result": None,
            "output": "",
            "execution_time": 0,
            "error": str(e),
        }


@mcp.tool()
async def call_tool(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Call a single tool directly.
    
    For simple cases where you just need to call one tool,
    this provides direct access without writing code.
    
    For complex multi-tool operations, prefer execute_code().
    
    Args:
        tool_name: The full tool name (format: server__toolname).
    
    Returns:
        Dictionary with:
        - success: Whether the call succeeded
        - result: The tool's return value
        - error: Error message if call failed
    """
    executor = get_executor()
    
    try:
        result = await executor.call_tool(tool_name, arguments)
        return {
            "success": True,
            "result": result,
            "error": None,
        }
    except Exception as e:
        logger.debug("Tool call failed: %s", tool_name, exc_info=e)
        return {
            "success": False,
            "result": None,
            "error": str(e),
        }


# =============================================================================
# Skills - Saved Tool Compositions
# =============================================================================

@mcp.tool()
async def save_skill(
    name: str,
    code: str,
    description: str,
    tags: Optional[list[str]] = None,
    parameters: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Save a reusable skill (code-based tool composition).
    
    Skills are saved code snippets that can be executed later.
    Think of them as macros or recipes for common multi-tool operations.
    
    Args:
        name: Unique name for the skill.
        code: Python code implementing the skill.
        description: Human-readable description of what it does.
        tags: Optional list of tags for categorization.
        parameters: Optional JSON schema for skill parameters.
    
    Returns:
                    logger.debug("Tool call failed: %s: %s", tool_name, e)
        - success: Whether the skill was saved
        - skill_id: The assigned skill ID
        - error: Error message if save failed
    """
    from agent_skills import SimpleSkillsManager, SimpleSkill
    
    config = _config or CodeModeConfig()
    manager = SimpleSkillsManager(config.skills_path)
    
    skill = SimpleSkill(
        name=name,
        description=description,
        code=code,
        tags=tags or [],
        parameters=parameters or {},
    )
    
    try:
        manager.save_skill(skill=skill)
        return {
            "success": True,
            "skill_id": name,
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "skill_id": None,
            "error": str(e),
        }


@mcp.tool()
async def run_skill(
    name: str,
    arguments: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Execute a saved skill.
    
    Args:
        name: Name of the skill to execute.
        arguments: Optional arguments to pass to the skill.
    
    Returns:
        Dictionary with execution results (same as execute_code).
    """
    executor = get_executor()
    
    try:
        execution = await executor.execute_skill(name, arguments)
        
        return {
            "success": not execution.error,
            "result": execution.text,
            "results": [
                {
                    "data": r.data,
                    "is_main_result": r.is_main_result,
                    "extra": r.extra,
                }
                for r in execution.results
            ],
            "stdout": execution.stdout,
            "stderr": execution.stderr,
            "output": execution.stdout,
            "execution_time": execution.execution_time or 0,
            "error": str(execution.error) if execution.error else None,
        }
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "output": "",
            "execution_time": 0,
            "error": str(e),
        }


@mcp.tool()
async def list_skills(
    tags: Optional[list[str]] = None,
) -> dict[str, Any]:
    """List available skills.
    
    Args:
        tags: Optional filter by tags.
    
    Returns:
        Dictionary with list of skills.
    """
    from agent_skills import SimpleSkillsManager
    
    config = _config or CodeModeConfig()
    manager = SimpleSkillsManager(config.skills_path)
    
    skills = manager.list_skills()
    
    # Filter by tags if specified
    if tags:
        skills = [s for s in skills if any(t in s.tags for t in tags)]
    
    return {
        "skills": [
            {
                "name": s.name,
                "description": s.description,
                "tags": s.tags,
            }
            for s in skills
        ],
        "total": len(skills),
    }


@mcp.tool()
async def delete_skill(name: str) -> dict[str, Any]:
    """Delete a saved skill.
    
    Args:
        name: Name of the skill to delete.
    
    Returns:
        Dictionary with success status.
    """
    from agent_skills import SimpleSkillsManager
    
    config = _config or CodeModeConfig()
    manager = SimpleSkillsManager(config.skills_path)
    
    success = manager.delete_skill(name)
    
    return {
        "success": success,
        "error": None if success else f"Skill not found: {name}",
    }


# =============================================================================
# Tool History & Debugging
# =============================================================================

@mcp.tool()
async def get_execution_history(limit: int = 10) -> dict[str, Any]:
    """Get recent tool execution history.
    
    Useful for debugging and understanding what tools have been called.
    
    Args:
        limit: Maximum number of entries to return.
    
    Returns:
        Dictionary with list of recent executions.
    """
    executor = get_executor()
    history = executor.tool_call_history[-limit:]
    
    return {
        "history": [
            {
                "tool_name": h.tool_name,
                "success": h.success,
                "execution_time": h.execution_time,
                "error": h.error,
            }
            for h in history
        ],
        "total": len(executor.tool_call_history),
    }


# =============================================================================
# Server Management
# =============================================================================

@mcp.tool()
async def add_mcp_server(
    name: str,
    url: Optional[str] = None,
    command: Optional[str] = None,
    args: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Add a new MCP server to discover tools from.
    
    Supports both HTTP-based and stdio-based MCP servers.
    
    Args:
        name: Unique name for the server.
        url: HTTP URL for HTTP-based servers.
        command: Command to run for stdio-based servers.
        args: Arguments for the command.
    
    Returns:
        Dictionary with:
        - success: Whether the server was added
        - tools_discovered: Number of tools discovered
        - error: Error message if failed
    """
    from .models import MCPServerConfig
    
    registry = get_registry()
    
    if url:
        config = MCPServerConfig(
            name=name,
            transport="http",
            url=url,
        )
    elif command:
        config = MCPServerConfig(
            name=name,
            transport="stdio",
            command=command,
            args=args or [],
        )
    else:
        return {
            "success": False,
            "tools_discovered": 0,
            "error": "Either url or command must be provided",
        }
    
    try:
        registry.add_server(config)
        await registry.discover_tools(name)
        
        tools = registry.list_tools(server=name)
        
        return {
            "success": True,
            "tools_discovered": len(tools),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "tools_discovered": 0,
            "error": str(e),
        }


def run() -> None:
    """Run the MCP server."""
    configure()
    mcp.run()


if __name__ == "__main__":
    run()
