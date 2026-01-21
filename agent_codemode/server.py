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

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server

from .composition.executor import CodeModeExecutor
from .discovery.registry import ToolRegistry
from .models import CodeModeConfig, SearchResult, Skill

logger = logging.getLogger(__name__)

# Create the MCP server
mcp = Server("codemode")

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
# Tool Definitions
# =============================================================================

TOOLS = [
    types.Tool(
        name="search_tools",
        description="""Search for available tools matching a query.

This is the Tool Search Tool - use it to discover relevant tools
before deciding which ones to use. Especially useful when there
are many tools available (100+).

Instead of loading all tool definitions upfront, this allows
progressive discovery of relevant tools based on your task.

Example:
    # Find tools for working with files
    result = search_tools("read and write files")
    # Returns: {"tools": [{"name": "fs__read_file", ...}], "total": 5}""",
        inputSchema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of what you're looking for. Examples: 'file operations', 'data analysis', 'web scraping'",
                },
                "server": {
                    "type": "string",
                    "description": "Optional filter by MCP server name.",
                },
                "category": {
                    "type": "string",
                    "description": "Optional filter by category (e.g., 'filesystem', 'network').",
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of results to return (default: 10).",
                },
                "include_deferred": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to include tools marked defer_loading.",
                },
            },
        },
    ),
    types.Tool(
        name="list_servers",
        description="""List all connected MCP servers.

Returns information about all MCP servers that are currently
connected and available for tool discovery.""",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    types.Tool(
        name="get_tool_details",
        description="""Get detailed information about a specific tool.

After finding a tool with search_tools, use this to get
the full schema and usage information.""",
        inputSchema={
            "type": "object",
            "required": ["tool_name"],
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "The full tool name (format: server__toolname).",
                },
            },
        },
    ),
    types.Tool(
        name="execute_code",
        description="""Execute Python code that can compose and call tools.

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

Example:
    # Read multiple files in parallel
    code = '''
    import asyncio
    from generated.servers.filesystem import read_file
    
    files = ["/path/file1.txt", "/path/file2.txt"]
    results = await asyncio.gather(*[read_file({"path": f}) for f in files])
    '''
    result = execute_code(code)""",
        inputSchema={
            "type": "object",
            "required": ["code"],
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Can use `await` for async operations. Import tools from `generated.servers.<server_name>`.",
                },
                "timeout": {
                    "type": "number",
                    "default": 30.0,
                    "description": "Maximum execution time in seconds (default: 30).",
                },
                "context": {
                    "type": "object",
                    "description": "Optional variables to inject into the execution context.",
                },
            },
        },
    ),
    types.Tool(
        name="call_tool",
        description="""Call a single tool directly.

For simple cases where you just need to call one tool,
this provides direct access without writing code.

For complex multi-tool operations, prefer execute_code().""",
        inputSchema={
            "type": "object",
            "required": ["tool_name", "arguments"],
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "The full tool name (format: server__toolname).",
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments to pass to the tool.",
                },
            },
        },
    ),
    types.Tool(
        name="save_skill",
        description="""Save a reusable skill (code-based tool composition).

Skills are saved code snippets that can be executed later.
Think of them as macros or recipes for common multi-tool operations.""",
        inputSchema={
            "type": "object",
            "required": ["name", "code", "description"],
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Unique name for the skill.",
                },
                "code": {
                    "type": "string",
                    "description": "Python code implementing the skill.",
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description of what it does.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of tags for categorization.",
                },
                "parameters": {
                    "type": "object",
                    "description": "Optional JSON schema for skill parameters.",
                },
            },
        },
    ),
    types.Tool(
        name="run_skill",
        description="""Execute a saved skill.""",
        inputSchema={
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the skill to execute.",
                },
                "arguments": {
                    "type": "object",
                    "description": "Optional arguments to pass to the skill.",
                },
            },
        },
    ),
    types.Tool(
        name="list_skills",
        description="""List available skills.""",
        inputSchema={
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional filter by tags.",
                },
            },
        },
    ),
    types.Tool(
        name="delete_skill",
        description="""Delete a saved skill.""",
        inputSchema={
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the skill to delete.",
                },
            },
        },
    ),
    types.Tool(
        name="get_execution_history",
        description="""Get recent tool execution history.

Useful for debugging and understanding what tools have been called.""",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of entries to return.",
                },
            },
        },
    ),
    types.Tool(
        name="add_mcp_server",
        description="""Add a new MCP server to discover tools from.

Supports both HTTP-based and stdio-based MCP servers.""",
        inputSchema={
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Unique name for the server.",
                },
                "url": {
                    "type": "string",
                    "description": "HTTP URL for HTTP-based servers.",
                },
                "command": {
                    "type": "string",
                    "description": "Command to run for stdio-based servers.",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Arguments for the command.",
                },
            },
        },
    ),
]


# =============================================================================
# Tool Handlers
# =============================================================================

async def handle_search_tools(arguments: dict[str, Any]) -> dict[str, Any]:
    """Search for available tools matching a query."""
    query = arguments["query"]
    server = arguments.get("server")
    category = arguments.get("category")
    limit = arguments.get("limit", 10)
    include_deferred = arguments.get("include_deferred", True)

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


async def handle_list_servers(arguments: dict[str, Any]) -> dict[str, Any]:
    """List all connected MCP servers."""
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


async def handle_get_tool_details(arguments: dict[str, Any]) -> dict[str, Any]:
    """Get detailed information about a specific tool."""
    tool_name = arguments["tool_name"]
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


async def handle_execute_code(arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute Python code that can compose and call tools."""
    code = arguments["code"]
    timeout = arguments.get("timeout", 30.0)
    context = arguments.get("context")

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


async def handle_call_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    """Call a single tool directly."""
    tool_name = arguments["tool_name"]
    tool_arguments = arguments["arguments"]

    executor = get_executor()
    
    try:
        result = await executor.call_tool(tool_name, tool_arguments)
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


async def handle_save_skill(arguments: dict[str, Any]) -> dict[str, Any]:
    """Save a reusable skill (code-based tool composition)."""
    from agent_skills import SimpleSkillsManager, SimpleSkill
    
    name = arguments["name"]
    code = arguments["code"]
    description = arguments["description"]
    tags = arguments.get("tags", [])
    parameters = arguments.get("parameters", {})

    config = _config or CodeModeConfig()
    manager = SimpleSkillsManager(config.skills_path)
    
    skill = SimpleSkill(
        name=name,
        description=description,
        code=code,
        tags=tags,
        parameters=parameters,
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


async def handle_run_skill(arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a saved skill."""
    name = arguments["name"]
    skill_arguments = arguments.get("arguments")

    executor = get_executor()
    
    try:
        execution = await executor.execute_skill(name, skill_arguments)
        
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


async def handle_list_skills(arguments: dict[str, Any]) -> dict[str, Any]:
    """List available skills."""
    from agent_skills import SimpleSkillsManager
    
    tags = arguments.get("tags")

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


async def handle_delete_skill(arguments: dict[str, Any]) -> dict[str, Any]:
    """Delete a saved skill."""
    from agent_skills import SimpleSkillsManager
    
    name = arguments["name"]

    config = _config or CodeModeConfig()
    manager = SimpleSkillsManager(config.skills_path)
    
    success = manager.delete_skill(name)
    
    return {
        "success": success,
        "error": None if success else f"Skill not found: {name}",
    }


async def handle_get_execution_history(arguments: dict[str, Any]) -> dict[str, Any]:
    """Get recent tool execution history."""
    limit = arguments.get("limit", 10)

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


async def handle_add_mcp_server(arguments: dict[str, Any]) -> dict[str, Any]:
    """Add a new MCP server to discover tools from."""
    from .models import MCPServerConfig
    
    name = arguments["name"]
    url = arguments.get("url")
    command = arguments.get("command")
    args = arguments.get("args", [])

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
            args=args,
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


# =============================================================================
# Tool Dispatcher
# =============================================================================

TOOL_HANDLERS = {
    "search_tools": handle_search_tools,
    "list_servers": handle_list_servers,
    "get_tool_details": handle_get_tool_details,
    "execute_code": handle_execute_code,
    "call_tool": handle_call_tool,
    "save_skill": handle_save_skill,
    "run_skill": handle_run_skill,
    "list_skills": handle_list_skills,
    "delete_skill": handle_delete_skill,
    "get_execution_history": handle_get_execution_history,
    "add_mcp_server": handle_add_mcp_server,
}


# =============================================================================
# MCP Server Handlers
# =============================================================================

@mcp.list_tools()
async def list_tools() -> list[types.Tool]:
    """Return the list of available tools."""
    return TOOLS


@mcp.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Handle tool calls."""
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name}")
    
    result = await handler(arguments)
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


# =============================================================================
# Server Entry Points
# =============================================================================

def run(transport: str = "stdio", port: int = 8000) -> None:
    """Run the MCP server.
    
    Args:
        transport: Transport type - "stdio" or "sse" (default: "stdio").
        port: Port for SSE transport (default: 8000).
    """
    configure()
    
    if transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.responses import Response
        from starlette.routing import Mount, Route
        from starlette.requests import Request
        import uvicorn

        sse = SseServerTransport("/messages/")

        async def handle_sse(request: Request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await mcp.run(
                    streams[0], streams[1], mcp.create_initialization_options()
                )
            return Response()

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=handle_sse, methods=["GET"]),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        uvicorn.run(starlette_app, host="127.0.0.1", port=port)
    else:
        from mcp.server.stdio import stdio_server

        async def arun():
            async with stdio_server() as streams:
                await mcp.run(
                    streams[0], streams[1], mcp.create_initialization_options()
                )

        anyio.run(arun)


if __name__ == "__main__":
    import sys
    
    transport = "stdio"
    port = 8000
    
    # Simple CLI argument parsing
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--transport" and i + 1 < len(args):
            transport = args[i + 1]
        elif arg == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
    
    run(transport=transport, port=port)
