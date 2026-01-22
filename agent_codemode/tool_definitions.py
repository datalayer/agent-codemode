# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Shared tool definitions for Agent Codemode.

This module provides a single source of truth for tool schemas,
used by both the MCP server and pydantic-ai toolset.
"""

from typing import Any

# Concise tool definitions - descriptions kept minimal
TOOL_SCHEMAS = {
    "search_tools": {
        "description": "Search for available tools by natural language query.",
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of what you're looking for",
                },
                "server": {
                    "type": "string",
                    "description": "Optional: filter by MCP server name",
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum results (default: 10)",
                },
                "include_deferred": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include deferred tools",
                },
            },
        },
    },
    "get_tool_details": {
        "description": "Get detailed schema and documentation for a specific tool.",
        "parameters": {
            "type": "object",
            "required": ["tool_name"],
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Full tool name (format: server__toolname)",
                },
            },
        },
    },
    "list_tool_names": {
        "description": "List available tool names without full schemas.",
        "parameters": {
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": "Optional: filter by server name",
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: filter by keywords",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default: 100)",
                    "default": 100,
                },
                "include_deferred": {
                    "type": "boolean",
                    "description": "Include deferred tools",
                    "default": False,
                },
            },
        },
    },
    "list_servers": {
        "description": "List all connected MCP servers.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    "execute_code": {
        "description": "Execute Python code that composes and calls tools. Import tools using: `from generated.servers.<server_name> import <function_name>`. CRITICAL: (1) All generated tools are async - MUST use `await` when calling them. (2) ALWAYS call get_tool_details first to check parameter names and return value structure. (3) Tool return values may be dicts - extract fields before using.",
        "parameters": {
            "type": "object",
            "required": ["code"],
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Use 'await' for all tool calls.",
                },
                "timeout": {
                    "type": "number",
                    "description": "Execution timeout in seconds (default: 30)",
                    "default": 30,
                },
            },
        },
    },
    "call_tool": {
        "description": "Call a single tool directly with arguments.",
        "parameters": {
            "type": "object",
            "required": ["tool_name", "arguments"],
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Full tool name (format: server__toolname)",
                },
                "arguments": {
                    "type": "object",
                    "description": "Tool arguments",
                },
            },
        },
    },
}


def get_tool_schema(tool_name: str) -> dict[str, Any]:
    """Get the schema for a specific tool.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Dictionary with 'description' and 'parameters' keys
        
    Raises:
        KeyError: If tool not found
    """
    return TOOL_SCHEMAS[tool_name]
