# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Agent Codemode - Programmatic MCP tool calling and composition.

This package enables:
- Progressive tool discovery
- Programmatic tool composition (code that chains tools)
- State persistence
- Skill building (reusable tool patterns)

Example:
    from agent_codemode import ToolRegistry, CodeModeExecutor, MCPServerConfig

    # Set up registry with MCP servers
    registry = ToolRegistry()
    registry.add_server(MCPServerConfig(name="bash", url="http://localhost:8001"))
    await registry.discover_all()

    # Execute code that composes tools
    async with CodeModeExecutor(registry) as executor:
        result = await executor.execute('''
            from generated.mcp.bash import ls, cat

            files = await ls({"path": "/tmp"})
            print(f"Found {len(files)} files")
        ''')
"""

# Import skills functionality from agent_skills
from agent_skills import (  # type: ignore[import-untyped]
    RateLimiter,
    Skill,
    SkillDirectory,
    SkillFile,
    SkillsManager,
    parallel,
    retry,
    run_with_timeout,
    setup_skills_directory,
    wait_for,
)

from .composition.executor import CodeModeExecutor
from .discovery.codegen import PythonCodeGenerator
from .discovery.registry import ToolRegistry
from .proxy.mcp_client import MCPClient
from .proxy.meta_tools import MetaToolProvider
from .server import configure as configure_server
from .server import mcp as codemode_server
from .toolset import PYDANTIC_AI_AVAILABLE, CodemodeToolset
from .types import (
    CodeModeConfig,
    MCPServerConfig,
    SearchResult,
    ServerInfo,
    ToolCallResult,
    ToolDefinition,
    ToolParameter,
)

__all__ = [
    "PYDANTIC_AI_AVAILABLE",
    "CodeModeConfig",
    "CodeModeExecutor",
    # Pydantic AI Toolset
    "CodemodeToolset",
    # Proxy
    "MCPClient",
    "MCPServerConfig",
    "MetaToolProvider",
    "PythonCodeGenerator",
    "RateLimiter",
    "SearchResult",
    "ServerInfo",
    # Skills (from agent_skills)
    "Skill",
    "SkillDirectory",
    "SkillFile",
    "SkillsManager",
    "ToolCallResult",
    # Models
    "ToolDefinition",
    "ToolParameter",
    # Core components
    "ToolRegistry",
    # MCP Server
    "codemode_server",
    "configure_server",
    "parallel",
    "retry",
    "run_with_timeout",
    "setup_skills_directory",
    # Helpers (from agent_skills)
    "wait_for",
]
