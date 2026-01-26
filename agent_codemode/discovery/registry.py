# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Tool Registry - Central registry for MCP tools.

The ToolRegistry manages tool discovery, registration, and search
across multiple MCP servers.
"""

import logging
from typing import Optional

from ..types import MCPServerConfig, SearchResult, ServerInfo, ToolDefinition
from ..proxy.mcp_client import MCPClient

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for MCP tools.

    The ToolRegistry connects to multiple MCP servers, discovers their tools,
    and provides search and lookup capabilities.

    Example:
        registry = ToolRegistry()
        registry.add_server(MCPServerConfig(name="bash", url="http://localhost:8001"))
        await registry.discover_all()

        # Search for tools
        results = await registry.search_tools("file operations")
        for tool in results.tools:
            print(tool.name, tool.description)

        # Get a specific tool
        tool = registry.get_tool("bash__ls")
    """

    def __init__(self):
        """Initialize the tool registry."""
        self._servers: dict[str, MCPServerConfig] = {}
        self._clients: dict[str, MCPClient] = {}
        self._tools: dict[str, ToolDefinition] = {}
        self._tools_by_server: dict[str, list[str]] = {}
        self._call_count: int = 0

    def add_server(self, config: MCPServerConfig) -> None:
        """Add an MCP server to the registry.

        Args:
            config: Server configuration.
        """
        self._servers[config.name] = config
        if config.enabled:
            self._clients[config.name] = MCPClient(
                name=config.name,
                url=config.url,
                command=config.command,
                args=config.args,
                env=config.env,
            )
        self._tools_by_server[config.name] = []

    def remove_server(self, name: str) -> None:
        """Remove an MCP server from the registry.

        Args:
            name: Server name.
        """
        if name in self._servers:
            del self._servers[name]
        if name in self._clients:
            del self._clients[name]
        # Remove tools from this server
        for tool_name in self._tools_by_server.get(name, []):
            if tool_name in self._tools:
                del self._tools[tool_name]
        if name in self._tools_by_server:
            del self._tools_by_server[name]

    async def discover_all(self) -> dict[str, list[ToolDefinition]]:
        """Discover tools from all registered servers.

        Returns:
            Dictionary mapping server names to lists of tools.
        """
        results = {}
        for server_name, client in self._clients.items():
            try:
                tools = await client.list_tools()
                server_config = self._servers[server_name]

                for tool in tools:
                    # Create full tool name with server prefix
                    full_name = f"{server_name}__{tool['name']}"
                    input_examples = (
                        tool.get("inputExamples")
                        or tool.get("input_examples")
                        or tool.get("examples")
                        or []
                    )
                    defer_loading = (
                        tool.get("deferLoading")
                        or tool.get("defer_loading")
                        or tool.get("deferred")
                        or False
                    )
                    tool_def = ToolDefinition(
                        name=full_name,
                        description=tool.get("description", ""),
                        input_schema=tool.get("inputSchema", {}),
                        output_schema=tool.get("outputSchema"),
                        input_examples=input_examples,
                        defer_loading=bool(defer_loading),
                        server_name=server_name,
                        server_url=server_config.url,
                    )
                    self._tools[full_name] = tool_def
                    self._tools_by_server[server_name].append(full_name)

                results[server_name] = [
                    self._tools[name] for name in self._tools_by_server[server_name]
                ]
            except Exception as e:
                logger.debug("Error discovering tools from %s: %s", server_name, e)
                results[server_name] = []

        return results

    async def discover_server(self, server_name: str) -> list[ToolDefinition]:
        """Discover tools from a specific server.

        Args:
            server_name: Name of the server to discover.

        Returns:
            List of discovered tools.
        """
        if server_name not in self._clients:
            return []

        client = self._clients[server_name]
        server_config = self._servers[server_name]

        # Clear existing tools from this server
        for tool_name in self._tools_by_server.get(server_name, []):
            if tool_name in self._tools:
                del self._tools[tool_name]
        self._tools_by_server[server_name] = []

        try:
            tools = await client.list_tools()
            for tool in tools:
                full_name = f"{server_name}__{tool['name']}"
                input_examples = (
                    tool.get("inputExamples")
                    or tool.get("input_examples")
                    or tool.get("examples")
                    or []
                )
                defer_loading = (
                    tool.get("deferLoading")
                    or tool.get("defer_loading")
                    or tool.get("deferred")
                    or False
                )
                tool_def = ToolDefinition(
                    name=full_name,
                    description=tool.get("description", ""),
                    input_schema=tool.get("inputSchema", {}),
                    output_schema=tool.get("outputSchema"),
                    input_examples=input_examples,
                    defer_loading=bool(defer_loading),
                    server_name=server_name,
                    server_url=server_config.url,
                )
                self._tools[full_name] = tool_def
                self._tools_by_server[server_name].append(full_name)

            return [self._tools[name] for name in self._tools_by_server[server_name]]
        except Exception as e:
            logger.debug("Error discovering tools from %s: %s", server_name, e)
            return []

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name.

        Args:
            name: Full tool name (server__toolname format).

        Returns:
            Tool definition or None if not found.
        """
        return self._tools.get(name)

    def list_tools(
        self,
        server: Optional[str] = None,
        limit: Optional[int] = None,
        include_deferred: bool = False,
    ) -> list[ToolDefinition]:
        """List all registered tools.

        Args:
            server: Optional server filter.
            limit: Maximum number of tools to return.

        Returns:
            List of tool definitions.
        """
        if server:
            tool_names = self._tools_by_server.get(server, [])
            tools = [self._tools[name] for name in tool_names if name in self._tools]
        else:
            tools = list(self._tools.values())

        if not include_deferred:
            tools = [t for t in tools if not t.defer_loading]

        if limit:
            tools = tools[:limit]

        return tools

    def list_tool_names(
        self,
        server: Optional[str] = None,
        keywords: Optional[list[str]] = None,
        limit: int = 100,
        include_deferred: bool = False,
    ) -> list[str]:
        """List tool names with optional filtering.

        Args:
            server: Optional server filter.
            keywords: Optional keywords to filter by.
            limit: Maximum number of names to return.

        Returns:
            List of tool names.
        """
        if server:
            tools = [
                self._tools[name]
                for name in self._tools_by_server.get(server, [])
                if name in self._tools
            ]
        else:
            tools = list(self._tools.values())

        if not include_deferred:
            tools = [t for t in tools if not t.defer_loading]

        # Filter by keywords if provided
        if keywords:
            filtered = []
            for tool in tools:
                tool_text = f"{tool.name} {tool.description}".lower()
                if any(kw.lower() in tool_text for kw in keywords):
                    filtered.append(tool)
            tools = filtered

        return [tool.name for tool in tools[:limit]]

    async def search_tools(
        self,
        query: str,
        server: Optional[str] = None,
        limit: int = 10,
        include_deferred: bool = True,
    ) -> SearchResult:
        """Search for tools matching a query.

        This performs a simple keyword-based search. For more sophisticated
        search, you can integrate with an LLM or embedding-based search.

        Args:
            query: Search query.
            server: Optional server filter.
            limit: Maximum number of results.

        Returns:
            Search results.
        """
        query_lower = query.lower()
        query_words = query_lower.split()

        # Get candidate tools
        if server:
            candidates = [
                self._tools[name]
                for name in self._tools_by_server.get(server, [])
                if name in self._tools
            ]
        else:
            candidates = list(self._tools.values())

        if not include_deferred:
            candidates = [t for t in candidates if not t.defer_loading]

        # Score each tool by keyword matches
        scored_tools = []
        for tool in candidates:
            tool_text = f"{tool.name} {tool.description}".lower()
            score = sum(1 for word in query_words if word in tool_text)
            if score > 0:
                scored_tools.append((score, tool))

        # Sort by score descending
        scored_tools.sort(key=lambda x: x[0], reverse=True)
        matched_tools = [tool for _, tool in scored_tools[:limit]]

        return SearchResult(
            tools=matched_tools,
            total=len(scored_tools),
            query=query,
        )

    async def call_tool(
        self, tool_name: str, arguments: dict
    ) -> dict:
        """Call a tool with arguments.

        Args:
            tool_name: Full tool name (server__toolname format).
            arguments: Tool arguments.

        Returns:
            Tool result.
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return {"error": f"Tool not found: {tool_name}"}

        client = self._clients.get(tool.server_name)
        if not client:
            return {"error": f"Server not available: {tool.server_name}"}

        # Extract the original tool name (without server prefix)
        original_name = tool_name.split("__", 1)[1] if "__" in tool_name else tool_name

        self._call_count += 1
        return await client.call_tool(original_name, arguments)

    @property
    def mcp_call_count(self) -> int:
        """Get total MCP tool call count."""
        return self._call_count

    @property
    def servers(self) -> list[str]:
        """Get list of registered server names."""
        return list(self._servers.keys())

    @property
    def tool_count(self) -> int:
        """Get total number of registered tools."""
        return len(self._tools)

    async def list_servers(self) -> list[ServerInfo]:
        """List all connected servers with their metadata.

        Returns:
            List of ServerInfo objects.
        """
        result = []
        for name, config in self._servers.items():
            tool_count = len(self._tools_by_server.get(name, []))
            result.append(
                ServerInfo(
                    name=name,
                    description=f"MCP server: {name}",
                    tool_count=tool_count,
                    url=config.url,
                    status="connected" if name in self._clients else "disconnected",
                )
            )
        return result

    def __repr__(self) -> str:
        return f"ToolRegistry(servers={len(self._servers)}, tools={len(self._tools)})"
