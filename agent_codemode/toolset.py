# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Codemode Toolset for Pydantic AI - Method-based tool composition.

This module provides a PydanticAI-compatible toolset that exposes codemode
tools directly as method calls, bypassing MCP for efficiency.

Key tools:
- search_tools: Progressive tool discovery
- get_tool_details: Get full tool schema
- execute_code: Run Python code that composes tools
- call_tool: Direct single-tool invocation

Example:
    from pydantic_ai import Agent
    from agent_codemode import CodemodeToolset, ToolRegistry
    
    # Set up registry
    registry = ToolRegistry()
    registry.add_server(MCPServerConfig(name="bash", url="..."))
    await registry.discover_all()
    
    # Create toolset
    toolset = CodemodeToolset(registry=registry)
    
    # Use with agent
    agent = Agent(
        model='openai:gpt-4o',
        toolsets=[toolset],
    )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

# Lazy imports to avoid circular dependencies
# ToolRegistry and CodeModeExecutor are imported at runtime in methods
if TYPE_CHECKING:
    from .discovery.registry import ToolRegistry
    from .composition.executor import CodeModeExecutor

from .types import CodeModeConfig

logger = logging.getLogger(__name__)


# Check if pydantic-ai is available
try:
    from pydantic_ai.toolsets import AbstractToolset
    from pydantic_ai.toolsets.abstract import ToolsetTool
    from pydantic_ai.tools import ToolDefinition
    from pydantic_ai._run_context import RunContext
    from pydantic_core import SchemaValidator, core_schema
    
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    AbstractToolset = object


if PYDANTIC_AI_AVAILABLE:
    
    # Schema validator for any args
    CODEMODE_ARGS_VALIDATOR = SchemaValidator(schema=core_schema.any_schema())
    
    @dataclass
    class CodemodeToolset(AbstractToolset):
        """Codemode toolset for pydantic-ai with method-based tool execution.
        
        This provides the same tools as the MCP server but via direct method
        calls, which is more efficient for in-process agent usage.
        
        Provides:
        - search_tools: Find relevant tools by query
        - get_tool_details: Get full tool definition
        - list_servers: List connected MCP servers
        - execute_code: Run Python code that composes tools
        - call_tool: Call a single tool directly
        
        Example:
            from agent_codemode import CodemodeToolset, ToolRegistry
            from pydantic_ai import Agent
            
            registry = ToolRegistry()
            # ... configure registry ...
            
            toolset = CodemodeToolset(registry=registry)
            
            agent = Agent(
                model='openai:gpt-4o',
                toolsets=[toolset],
            )
        """
        
        registry: ToolRegistry | None = None
        config: CodeModeConfig = field(default_factory=CodeModeConfig)
        sandbox: Any | None = None  # Optional pre-configured sandbox (e.g., LocalEvalSandbox)
        allow_direct_tool_calls: bool | None = None
        allow_discovery_tools: bool = True
        tool_reranker: Callable[[list, str, Optional[str]], Awaitable[list]] | None = None
        _id: str | None = None
        
        # Internal state
        _executor: CodeModeExecutor | None = field(default=None, repr=False)
        _initialized: bool = field(default=False, repr=False)
        _codemode_call_count: int = field(default=0, repr=False)
        
        def __post_init__(self):
            if self.registry is None:
                # Import at runtime to avoid circular dependency
                from .discovery.registry import ToolRegistry
                self.registry = ToolRegistry()
            # Default the direct-call policy from config if not provided
            if self.allow_direct_tool_calls is None:
                self.allow_direct_tool_calls = self.config.allow_direct_tool_calls
        
        @property
        def id(self) -> str | None:
            return self._id
        
        @property
        def label(self) -> str:
            return "Codemode Toolset"
        
        async def _ensure_initialized(self) -> None:
            """Initialize the executor if not already done."""
            if self._initialized:
                return
            
            if self._executor is None:
                # Import at runtime to avoid circular dependency
                from .composition.executor import CodeModeExecutor
                # Ensure tools are discovered before generating bindings
                if self.registry is not None and not self.registry.list_tools():
                    logger.info("Codemode registry empty; discovering tools...")
                    await self.registry.discover_all()
                tool_count = len(self.registry.list_tools()) if self.registry is not None else 0
                logger.info("Codemode registry tool count: %s", tool_count)
                self._executor = CodeModeExecutor(
                    registry=self.registry,
                    config=self.config,
                    sandbox=self.sandbox,
                )
                await self._executor.setup()
                logger.info(
                    "Codemode executor setup complete (generated_path=%s)",
                    self.config.generated_path,
                )
            
            self._initialized = True

        async def start(self) -> None:
            """Start the toolset and executor."""
            await self._ensure_initialized()
        
        async def cleanup(self) -> None:
            """Clean up resources."""
            if self._executor:
                await self._executor.cleanup()
                self._executor = None
            self._initialized = False

        async def get_tools(self, ctx: RunContext) -> dict[str, ToolsetTool]:
            """Get the tools provided by this toolset."""
            from .tool_definitions import get_tool_schema
            
            tools = {}
            
            if self.allow_discovery_tools:
                # Discovery tools
                for tool_name in ["list_tool_names", "search_tools", "get_tool_details", "list_servers"]:
                    schema = get_tool_schema(tool_name)
                    tools[tool_name] = ToolsetTool(
                        toolset=self,
                        tool_def=ToolDefinition(
                            name=tool_name,
                            description=schema["description"],
                            parameters_json_schema=schema["parameters"],
                        ),
                        max_retries=0,
                        args_validator=CODEMODE_ARGS_VALIDATOR,
                    )
            
            # execute_code - always available
            schema = get_tool_schema("execute_code")
            tools["execute_code"] = ToolsetTool(
                toolset=self,
                tool_def=ToolDefinition(
                    name="execute_code",
                    description=schema["description"],
                    parameters_json_schema=schema["parameters"],
                ),
                max_retries=0,
                args_validator=CODEMODE_ARGS_VALIDATOR,
            )
            
            # call_tool (optional)
            if self.allow_direct_tool_calls:
                schema = get_tool_schema("call_tool")
                tools["call_tool"] = ToolsetTool(
                    toolset=self,
                    tool_def=ToolDefinition(
                        name="call_tool",
                        description=schema["description"],
                        parameters_json_schema=schema["parameters"],
                    ),
                    max_retries=1,
                    args_validator=CODEMODE_ARGS_VALIDATOR,
                )
            
            return tools
        
        async def call_tool(
            self,
            name: str,
            tool_args: dict[str, Any],
            ctx: RunContext,
            tool: ToolsetTool,
        ) -> Any:
            """Call a tool by name."""
            await self._ensure_initialized()
            self._codemode_call_count += 1
            
            if name == "list_tool_names":
                return await self._list_tool_names(
                    server=tool_args.get("server"),
                    keywords=tool_args.get("keywords"),
                    limit=tool_args.get("limit"),
                    include_deferred=tool_args.get("include_deferred", False),
                )
            elif name == "search_tools":
                return await self._search_tools(
                    query=tool_args.get("query", ""),
                    server=tool_args.get("server"),
                    limit=tool_args.get("limit", 10),
                    include_deferred=tool_args.get("include_deferred", True),
                )
            elif name == "get_tool_details":
                return await self._get_tool_details(
                    tool_name=tool_args.get("tool_name", ""),
                )
            elif name == "list_servers":
                return await self._list_servers()
            elif name == "execute_code":
                return await self._execute_code(
                    code=tool_args.get("code", ""),
                    timeout=tool_args.get("timeout", 30),
                )
            elif name == "call_tool" and self.allow_direct_tool_calls:
                return await self._call_tool(
                    tool_name=tool_args.get("tool_name", ""),
                    arguments=tool_args.get("arguments", {}),
                )
            else:
                raise ValueError(f"Unknown tool: {name}")
        
        async def _list_tool_names(
            self,
            server: Optional[str] = None,
            keywords: Optional[list[str]] = None,
            limit: Optional[int] = None,
            include_deferred: bool = False,
        ) -> dict[str, Any]:
            """List all tool names quickly without descriptions."""
            tools = self.registry.list_tools(server=server, include_deferred=include_deferred)
            total_available = len(tools)
            if keywords:
                lowered = [kw.lower() for kw in keywords]
                filtered = []
                for t in tools:
                    text = f"{t.name} {t.description}".lower()
                    if any(kw in text for kw in lowered):
                        filtered.append(t)
                tools = filtered
                total_available = len(filtered)
            if limit:
                tools = tools[:limit]
            
            # Group by server for better organization with import hints
            by_server: dict[str, list[str]] = {}
            import_hints: dict[str, str] = {}
            for t in tools:
                server_name = t.server_name or "unknown"
                if server_name not in by_server:
                    by_server[server_name] = []
                # Convert tool name to function name (replace dashes with underscores)
                func_name = t.name.split("__")[-1].replace("-", "_")
                by_server[server_name].append(func_name)
            
            # Generate import hints for each server
            for server_name, funcs in by_server.items():
                import_hints[server_name] = f"from generated.servers.{server_name} import {', '.join(funcs)}"
            
            return {
                "tools": [t.name for t in tools],
                "by_server": by_server,
                "import_hints": import_hints,
                "total": total_available,
                "returned": len(tools),
                "truncated": bool(limit) and len(tools) < total_available,
                "include_deferred": include_deferred,
                "usage": "Use import_hints to get the correct import statement for execute_code",
            }
        
        async def _search_tools(
            self,
            query: str,
            server: Optional[str] = None,
            limit: int = 10,
            include_deferred: bool = True,
        ) -> dict[str, Any]:
            """Search for tools matching a query."""
            result = await self.registry.search_tools(
                query, server=server, limit=limit, include_deferred=include_deferred
            )
            tools = result.tools

            # Optional reranker hook
            if self.tool_reranker:
                try:
                    from inspect import isawaitable

                    before = [t.name for t in tools]
                    reranked = self.tool_reranker(tools, query, server)
                    if isawaitable(reranked):
                        tools = await reranked
                    else:
                        tools = reranked  # type: ignore[assignment]
                    after = [t.name for t in tools]
                    logger.debug(
                        "Applied tool reranker: before=%s, after=%s",
                        before,
                        after,
                    )
                except Exception as e:
                    logger.debug(
                        "Tool reranker failed; falling back to registry order: %s",
                        e,
                    )
            
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
        
        async def _get_tool_details(self, tool_name: str) -> dict[str, Any]:
            """Get detailed information about a tool."""
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
            }
        
        async def _list_servers(self) -> dict[str, Any]:
            """List all connected MCP servers with import paths."""
            servers = await self.registry.list_servers()
            
            # Get tools for each server to provide function names
            server_tools: dict[str, list[str]] = {}
            for tool in self.registry.list_tools(include_deferred=True):
                sname = tool.server_name or "unknown"
                if sname not in server_tools:
                    server_tools[sname] = []
                # Convert tool name to function name (replace dashes with underscores)
                func_name = tool.name.split("__")[-1].replace("-", "_")
                server_tools[sname].append(func_name)
            
            return {
                "servers": [
                    {
                        "name": s.name,
                        "description": s.description,
                        "tool_count": s.tool_count,
                        "import_path": f"from generated.servers.{s.name} import {', '.join(server_tools.get(s.name, []))}",
                        "functions": server_tools.get(s.name, []),
                    }
                    for s in servers
                ],
                "total": len(servers),
                "usage_hint": "Use the import_path to import tools in execute_code",
            }
        
        async def _execute_code(
            self,
            code: str,
            timeout: float = 30.0,
        ) -> dict[str, Any]:
            """Execute Python code that composes tools."""
            if self._executor is None:
                return {"success": False, "error": "Executor not initialized"}
            
            try:
                start_time = time.monotonic()
                code_preview = (code or "")[:100]
                logger.info("Codemode execute_code: calling executor.execute() code=%r", code_preview)
                execution = await self._executor.execute(code, timeout=timeout)
                log_message = (
                    "Codemode execute_code: executor.execute() returned - "
                    f"stdout: {execution.logs.stdout_text!r}, "
                    f"stderr: {execution.logs.stderr_text!r}, "
                    f"execution_ok: {execution.execution_ok}, "
                    f"code_error: {execution.code_error}, "
                    f"exit_code: {execution.exit_code}"
                )
                if not execution.execution_ok:
                    logger.error(log_message)
                elif execution.exit_code not in (None, 0):
                    logger.warning(log_message)
                elif execution.code_error:
                    logger.warning(log_message)
                else:
                    logger.info(log_message)
                elapsed = time.monotonic() - start_time
                
                error_message = (
                    execution.execution_error
                    if not execution.execution_ok
                    else f"Process exited with code: {execution.exit_code}"
                    if execution.exit_code not in (None, 0)
                    else str(execution.code_error)
                    if execution.code_error
                    else None
                )
                result = {
                    "success": execution.success,
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
                    "execution_time": elapsed,
                    "execution_ok": execution.execution_ok,
                    "execution_error": execution.execution_error,
                    "code_error": str(execution.code_error) if execution.code_error else None,
                    "exit_code": execution.exit_code,
                    "error": error_message,
                }
                return result
            except Exception as e:
                # Log actual error for debugging
                logger.error(f"Code execution failed: {e}", exc_info=True)
                return {
                    "success": False,
                    "result": None,
                    "output": "",
                    "execution_time": 0,
                    "error": str(e),
                }
        
        async def _call_tool(
            self,
            tool_name: str,
            arguments: dict[str, Any],
        ) -> dict[str, Any]:
            """Call a single tool directly."""
            try:
                result = await self.registry.call_tool(tool_name, arguments)
                return {
                    "success": True,
                    "result": result,
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                }

        def get_call_counts(self) -> dict[str, int]:
            """Return counts for codemode and MCP tool calls."""
            mcp_calls = 0
            if self.registry is not None:
                mcp_calls = getattr(self.registry, "mcp_call_count", 0)
            return {
                "codemode_tool_calls": self._codemode_call_count,
                "mcp_tool_calls": mcp_calls,
            }


else:
    # Fallback when pydantic-ai is not available
    class CodemodeToolset:  # type: ignore
        """Placeholder when pydantic-ai is not installed."""
        
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "pydantic-ai is required for CodemodeToolset. "
                "Install with: pip install pydantic-ai"
            )
