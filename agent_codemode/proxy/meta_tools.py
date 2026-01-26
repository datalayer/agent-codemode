# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Meta-tools for agent tool discovery and execution.

These are the tools exposed to AI agents for discovering
and executing MCP tools programmatically.

Based on the TypeScript implementation from agent-codemode-claude-poc:
- Progressive tool discovery (search_tools, list_tool_names, get_tool_definition)
- AI-powered tool selection using a subagent
- Container/sandbox execution routing

Key insight from Code Mode: agents should use meta-tools to discover relevant
tools, then write code that uses the generated bindings to call those tools
directly without LLM inference overhead.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from ..discovery.registry import ToolRegistry
    from ..composition.executor import CodeModeExecutor

from ..models import ToolDefinition

logger = logging.getLogger(__name__)


# Type for AI tool selector function
AIToolSelector = Callable[[str, list[dict[str, Any]]], list[str]]


class MetaToolProvider:
    """Provider for meta-tools that agents can use.

    Meta-tools allow agents to:
    - Search for available tools (with AI-powered selection)
    - Get tool definitions
    - Execute code that composes tools

    Based on the TypeScript MetaToolProxy, this provides:
    1. list_tool_names - Fast listing when simple filtering works
    2. search_tools - AI-powered tool discovery with full definitions
    3. get_tool_definition - Get schema for a specific tool
    4. execute_code - Run Python code in a sandboxed environment

    Example:
        registry = ToolRegistry()
        executor = CodeModeExecutor(registry)
        provider = MetaToolProvider(registry, executor)

        # Fast listing
        names = provider.list_tool_names(keywords=["file", "read"])
        
        # AI-powered search
        tools = await provider.search_tools("read CSV files and analyze data")
        
        # Execute code
        result = await provider.execute_code('''
            from generated.servers.filesystem import read_file
            content = await read_file({"path": "data.csv"})
            print(content[:100])
        ''')
    """

    def __init__(
        self,
        registry: ToolRegistry,
        executor: Optional[CodeModeExecutor] = None,
        ai_selector: Optional[AIToolSelector] = None,
    ):
        """Initialize the meta-tool provider.

        Args:
            registry: Tool registry.
            executor: Optional code executor for container execution.
            ai_selector: Optional AI-powered tool selector function.
                        Takes (query, tool_list) and returns relevant tool names.
        """
        self.registry = registry
        self.executor = executor
        self._ai_selector = ai_selector

    async def search_tools(
        self,
        query: str,
        server: Optional[str] = None,
        limit: int = 10,
        include_deferred: bool = True,
    ) -> dict[str, Any]:
        """Search for tools matching a query.

        This is a meta-tool that agents can use to discover relevant tools.
        Uses AI-powered selection if an ai_selector is configured.

        Args:
            query: Search query describing what you're looking for.
            server: Optional server filter.
            limit: Maximum number of results.

        Returns:
            Dictionary with 'tools' list and 'total' count.
        """
        # Get all tools
        all_tools = self.registry.list_tools(server=server, include_deferred=include_deferred)
        
        # Use AI selector if available
        if self._ai_selector and query:
            tool_list = [
                {"name": t.name, "description": t.description}
                for t in all_tools
            ]
            try:
                selected_names = await self._ai_selector(query, tool_list)
                all_tools = [t for t in all_tools if t.name in selected_names]
            except Exception as e:
                logger.debug(
                    "AI tool selection failed, falling back to keyword search: %s",
                    e,
                )
                all_tools = self._keyword_filter(all_tools, query)
        elif query:
            all_tools = self._keyword_filter(all_tools, query)
        
        # Apply limit
        tools = all_tools[:limit]

        return {
            "total": len(all_tools),
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "server": tool.server_name,
                    "input_schema": tool.input_schema,
                    "output_schema": tool.output_schema,
                    "input_examples": tool.input_examples[:2],
                    "defer_loading": tool.defer_loading,
                }
                for tool in tools
            ],
        }
    
    def _keyword_filter(
        self, tools: list[ToolDefinition], query: str
    ) -> list[ToolDefinition]:
        """Filter tools by keywords in query.

        Args:
            tools: List of tools to filter.
            query: Search query.

        Returns:
            Filtered and scored list of tools.
        """
        query_lower = query.lower()
        keywords = query_lower.split()
        
        scored_tools = []
        for tool in tools:
            tool_text = f"{tool.name} {tool.description or ''}".lower()
            # Count matching keywords
            score = sum(1 for kw in keywords if kw in tool_text)
            if score > 0:
                scored_tools.append((score, tool))
        
        # Sort by score descending
        scored_tools.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored_tools]

    def list_tool_names(
        self,
        server: Optional[str] = None,
        keywords: Optional[list[str]] = None,
        limit: int = 100,
        include_deferred: bool = False,
    ) -> dict[str, Any]:
        """List available tool names.

        This is a fast way to see what tools are available without
        loading full definitions.

        Args:
            server: Optional server filter.
            keywords: Optional keywords to filter by.
            limit: Maximum number of names to return.

        Returns:
            Dictionary with 'tool_names' list and metadata.
        """
        all_names = self.registry.list_tool_names(
            server=server, keywords=keywords, limit=limit, include_deferred=include_deferred
        )
        total = len(self.registry.list_tools(server=server, include_deferred=include_deferred))

        return {
            "tool_names": all_names,
            "returned": len(all_names),
            "total": total,
            "truncated": len(all_names) < total,
            "include_deferred": include_deferred,
        }

    def get_tool_definition(self, tool_name: str) -> dict[str, Any]:
        """Get the full definition of a tool.

        Args:
            tool_name: Full tool name (server__toolname format).

        Returns:
            Tool definition with schema and metadata.
        """
        tool = self.registry.get_tool(tool_name)
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
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                }
                for p in tool.parameters
            ],
        }

    async def execute_code(self, code: str, timeout: float = 60.0) -> dict[str, Any]:
        """Execute code that composes MCP tools.

        The code can import from generated tool bindings and call
        multiple tools without LLM inference overhead.

        Args:
            code: Python code to execute.
            timeout: Execution timeout in seconds.

        Returns:
            Execution result with output and any errors.
        """
        if self.executor is None:
            return {"error": "Code executor not configured"}

        try:
            result = await self.executor.execute(code, timeout=timeout)
            error_message = (
                result.execution_error
                if not result.execution_ok
                else f"Process exited with code: {result.exit_code}"
                if result.exit_code not in (None, 0)
                else str(result.code_error)
                if result.code_error
                else None
            )

            return {
                "success": result.success,
                "execution_ok": result.execution_ok,
                "execution_error": result.execution_error,
                "code_error": str(result.code_error) if result.code_error else None,
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "result": result.text,
                "error": error_message,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def get_meta_tools(self) -> list[dict[str, Any]]:
        """Get the meta-tool definitions for exposing to agents.

        Returns:
            List of tool definitions in MCP format.
        """
        tools = [
            {
                "name": "search_tools",
                "description": (
                    "Search for available MCP tools by description. "
                    "Use this to discover tools relevant to your task."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query describing what you're looking for",
                        },
                        "server": {
                            "type": "string",
                            "description": "Optional server name to filter by",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 10)",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "list_tool_names",
                "description": (
                    "List available tool names. Fast way to see what tools exist."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "server": {
                            "type": "string",
                            "description": "Optional server name to filter by",
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional keywords to filter by",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of names (default: 100)",
                            "default": 100,
                        },
                    },
                },
            },
            {
                "name": "get_tool_definition",
                "description": (
                    "Get the full definition and schema of a specific tool."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Full tool name (server__toolname format)",
                        },
                    },
                    "required": ["tool_name"],
                },
            },
        ]

        # Add execute_code if executor is available
        if self.executor is not None:
            tools.append({
                "name": "execute_code",
                "description": (
                    "Execute Python code that composes MCP tools. "
                    "The code can import from generated tool bindings and "
                    "call multiple tools efficiently. Use this for complex "
                    "operations that require loops, conditionals, or state."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": (
                                "Python code to execute. Can use async/await "
                                "and import from generated.servers.*"
                            ),
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Execution timeout in seconds (default: 60)",
                            "default": 60,
                        },
                    },
                    "required": ["code"],
                },
            })

        return tools

    async def handle_tool_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle a meta-tool call.

        Args:
            tool_name: Name of the meta-tool.
            arguments: Tool arguments.

        Returns:
            Tool result.
        """
        if tool_name == "search_tools":
            return await self.search_tools(
                query=arguments.get("query", ""),
                server=arguments.get("server"),
                limit=arguments.get("limit", 10),
            )
        elif tool_name == "list_tool_names":
            return self.list_tool_names(
                server=arguments.get("server"),
                keywords=arguments.get("keywords"),
                limit=arguments.get("limit", 100),
            )
        elif tool_name == "get_tool_definition":
            return self.get_tool_definition(
                tool_name=arguments.get("tool_name", ""),
            )
        elif tool_name == "execute_code":
            return await self.execute_code(
                code=arguments.get("code", ""),
                timeout=arguments.get("timeout", 60),
            )
        else:
            return {"error": f"Unknown meta-tool: {tool_name}"}
