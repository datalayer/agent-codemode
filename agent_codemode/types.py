# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Types for Agent Codemode.

These models define tool definitions and execution contexts.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ToolParameter(BaseModel):
    """A parameter for an MCP tool.

    Attributes:
        name: Parameter name.
        type: JSON Schema type (string, number, boolean, object, array).
        description: Human-readable description.
        required: Whether the parameter is required.
        default: Default value if not provided.
        enum: List of allowed values (for enum types).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    description: str = ""
    required: bool = False
    default: Any = None
    enum: Optional[list[Any]] = None


class ToolDefinition(BaseModel):
    """Definition of an MCP tool.

    Attributes:
        name: Unique tool name (usually server__toolname format).
        description: Human-readable description.
        input_schema: JSON Schema for input parameters.
        output_schema: JSON Schema for output (optional).
        input_examples: Example inputs (tool use examples).
        defer_loading: Whether this tool is deferred from default listings.
        server_name: Name of the MCP server providing this tool.
        server_url: URL of the MCP server.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: Optional[dict[str, Any]] = None
    input_examples: list[dict[str, Any]] = Field(default_factory=list)
    defer_loading: bool = False
    server_name: str = ""
    server_url: str = ""

    @property
    def parameters(self) -> list[ToolParameter]:
        """Extract parameters from input schema."""
        params: list[ToolParameter] = []
        properties = self.input_schema.get("properties", {})
        required = self.input_schema.get("required", [])

        for name, prop in properties.items():
            params.append(
                ToolParameter(
                    name=name,
                    type=prop.get("type", "any"),
                    description=prop.get("description", ""),
                    required=name in required,
                    default=prop.get("default"),
                    enum=prop.get("enum"),
                )
            )
        return params

    @property
    def examples(self) -> list[dict[str, Any]]:
        """Backward-compatible alias for input examples."""
        return self.input_examples

    def __repr__(self) -> str:
        return f"ToolDefinition(name={self.name!r}, server={self.server_name!r})"


class ToolCallResult(BaseModel):
    """Result of a tool call.

    Attributes:
        tool_name: Name of the tool that was called.
        success: Whether the call succeeded.
        result: The result data (if successful).
        error: Error message (if failed).
        execution_time: Time taken in seconds.
    """

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0

    def __repr__(self) -> str:
        status = "success" if self.success else f"error={self.error}"
        return f"ToolCallResult({self.tool_name}, {status})"


class ServerInfo(BaseModel):
    """Information about an MCP server.

    Attributes:
        name: Server name.
        description: Human-readable description.
        tool_count: Number of tools provided by this server.
        url: Server URL (if HTTP-based).
        status: Connection status (connected, disconnected, etc.).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    tool_count: int = 0
    url: str = ""
    status: str = "connected"

    def __repr__(self) -> str:
        return f"ServerInfo(name={self.name!r}, tools={self.tool_count})"


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server.

    Attributes:
        name: Server name (used as prefix for tools).
        url: Server URL.
        command: Command to start the server (for stdio servers).
        args: Arguments for the command.
        env: Environment variables for the server.
        enabled: Whether the server is enabled.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    url: str = ""
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True

    @property
    def is_stdio(self) -> bool:
        """Check if this is a stdio-based server."""
        return bool(self.command)

    @property
    def is_http(self) -> bool:
        """Check if this is an HTTP-based server."""
        return bool(self.url) and not self.command


class CodeModeConfig(BaseModel):
    """Configuration for the CodeMode executor.

    Attributes:
        workspace_path: Path for workspace files.
        skills_path: Path for saved skills.
        generated_path: Path for generated code bindings.
        sandbox_variant: Which sandbox to use for execution.
        sandbox_image: Optional sandbox image (for Docker-based sandboxes).
        allow_direct_tool_calls: Whether to expose call_tool in the toolset.
        max_tool_calls: Optional safety cap for tool calls per execute() run.
    """

    model_config = ConfigDict(extra="forbid")

    workspace_path: str = "./workspace"
    skills_path: str = "./skills"
    generated_path: str = "./generated"
    sandbox_variant: str = "local-eval"
    sandbox_image: str | None = None
    allow_direct_tool_calls: bool = False
    max_tool_calls: int | None = None


class SearchResult(BaseModel):
    """Result of a tool search.

    Attributes:
        tools: List of matching tools.
        total: Total number of matches.
        query: The search query used.
    """

    model_config = ConfigDict(extra="forbid")

    tools: list[ToolDefinition]
    total: int
    query: str

    def __repr__(self) -> str:
        return f"SearchResult(query={self.query!r}, found={len(self.tools)}/{self.total})"
