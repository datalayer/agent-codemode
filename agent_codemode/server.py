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
from .models import CodeModeConfig, SearchResult

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
    
    if registry is not None:
        logger.debug(f"Using provided registry with {len(registry._servers)} servers")
        logger.debug(f"Server names: {list(registry._servers.keys())}")
        _registry = registry
    else:
        logger.debug("Creating new empty registry")
        _registry = ToolRegistry()
    
    _executor = CodeModeExecutor(_registry, _config)
    
    logger.info(f"Codemode MCP server configured with {len(_registry._servers)} servers")


def get_registry() -> ToolRegistry:
    """Get the tool registry."""
    global _registry
    if _registry is None:
        logger.debug("Registry is None, calling configure()")
        configure()
    else:
        logger.debug(f"Returning existing registry with {len(_registry._servers)} servers")
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

def _build_tools() -> list[types.Tool]:
    """Build MCP tool definitions from shared schemas."""
    from .tool_definitions import TOOL_SCHEMAS
    
    tools = []
    
    # Core codemode tools
    for name in ["search_tools", "list_tool_names", "list_servers", "get_tool_details", 
                 "execute_code", "call_tool"]:
        schema = TOOL_SCHEMAS[name]
        tools.append(types.Tool(
            name=name,
            description=schema["description"],
            inputSchema=schema["parameters"],
        ))
    
    # Skill management tools
    tools.extend([
        types.Tool(
            name="save_skill",
            description="Save a reusable skill (code-based tool composition).",
            inputSchema={
                "type": "object",
                "required": ["name", "code", "description"],
                "properties": {
                    "name": {"type": "string", "description": "Unique skill name"},
                    "code": {"type": "string", "description": "Python code"},
                    "description": {"type": "string", "description": "What the skill does"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "parameters": {"type": "object"},
                },
            },
        ),
        types.Tool(
            name="run_skill",
            description="Execute a saved skill.",
            inputSchema={
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string", "description": "Skill name"},
                    "arguments": {"type": "object"},
                },
            },
        ),
        types.Tool(
            name="list_skills",
            description="List available skills.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
            },
        ),
        types.Tool(
            name="delete_skill",
            description="Delete a saved skill.",
            inputSchema={
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="get_execution_history",
            description="Get recent tool execution history.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 10},
                },
            },
        ),
        types.Tool(
            name="add_mcp_server",
            description="Add a new MCP server to discover tools from.",
            inputSchema={
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "url": {"type": "string"},
                    "command": {"type": "string"},
                    "args": {"type": "array", "items": {"type": "string"}},
                },
            },
        ),
    ])
    
    return tools

TOOLS = _build_tools()


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


async def handle_list_tool_names(arguments: dict[str, Any]) -> dict[str, Any]:
    """List available tool names."""
    server = arguments.get("server")
    keywords = arguments.get("keywords")
    limit = arguments.get("limit", 100)
    include_deferred = arguments.get("include_deferred", False)

    registry = get_registry()
    names = registry.list_tool_names(
        server=server,
        keywords=keywords,
        limit=limit,
        include_deferred=include_deferred,
    )
    total = len(registry.list_tools(server=server, include_deferred=include_deferred))
    
    return {
        "tool_names": names,
        "returned": len(names),
        "total": total,
        "truncated": len(names) < total,
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
            "stdout": execution.stdout,
            "stderr": execution.stderr,
            "error": str(execution.error) if execution.error else None,
        }
    except Exception as e:
        logger.error("Code execution failed", exc_info=e)
        return {
            "success": False,
            "result": None,
            "output": "",
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
    from agent_skills.simple import SimpleSkill, SimpleSkillsManager
    
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
            "stdout": execution.stdout,
            "stderr": execution.stderr,
            "output": execution.stdout,
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
    from agent_skills.simple import SimpleSkillsManager
    
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
    from agent_skills.simple import SimpleSkillsManager
    
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
            url=url,
        )
    elif command:
        config = MCPServerConfig(
            name=name,
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
        await registry.discover_server(name)
        
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
    "list_tool_names": handle_list_tool_names,
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
    config = _config or CodeModeConfig()
    if config.allow_direct_tool_calls:
        return TOOLS
    return [tool for tool in TOOLS if tool.name != "call_tool"]


@mcp.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Handle tool calls."""
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name}")
    
    result = await handler(arguments)
    json_str = json.dumps(result, indent=2)
    return [types.TextContent(type="text", text=json_str)]


# =============================================================================
# Server Entry Points
# =============================================================================

def run(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the MCP server.
    
    Args:
        transport: Transport type - "stdio" or "streamable-http" (default: "stdio").
        host: Host for HTTP transport (default: "127.0.0.1").
        port: Port for HTTP transport (default: 8000).
    """
    # Note: configure() should be called by launcher before run()
    # Ensure registry is initialized (will use existing if already configured)
    if _registry is None:
        configure()
    
    if transport == "streamable-http":
        from mcp.server.streamable_http import StreamableHTTPServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Route
        import uvicorn

        async def handle_mcp(request):
            async with StreamableHTTPServerTransport(
                "/mcp", request.scope, request.receive, request._send
            ) as transport:
                await mcp.run(
                    transport.read_stream,
                    transport.write_stream,
                    mcp.create_initialization_options(),
                )

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/mcp", endpoint=handle_mcp, methods=["POST"]),
            ],
        )

        uvicorn.run(starlette_app, host=host, port=port)
    else:
        from mcp.server.stdio import stdio_server

        async def arun():
            # Do discovery before accepting requests to avoid blocking during tool calls
            registry = get_registry()
            if registry and not registry.list_tools():
                logger.info("Performing upfront discovery before starting server...")
                try:
                    await registry.discover_all()
                    logger.info(f"Upfront discovery complete: {len(registry.list_tools())} tools")
                except Exception as e:
                    logger.warning(f"Upfront discovery failed: {e}")
            
            async with stdio_server() as streams:
                await mcp.run(
                    streams[0], streams[1], mcp.create_initialization_options()
                )

        anyio.run(arun)


if __name__ == "__main__":
    import sys
    
    transport = "stdio"
    host = "127.0.0.1"
    port = 8000
    
    # Simple CLI argument parsing
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--transport" and i + 1 < len(args):
            transport = args[i + 1]
        elif arg == "--host" and i + 1 < len(args):
            host = args[i + 1]
        elif arg == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
    
    run(transport=transport, host=host, port=port)
