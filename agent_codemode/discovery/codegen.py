# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Python Code Generator for MCP tool bindings.

Generates Python functions from MCP tool schemas, allowing programmatic
tool composition without LLM inference overhead.
"""

import logging
from pathlib import Path
logger = logging.getLogger(__name__)

from typing import Any

from ..models import ToolDefinition


class PythonCodeGenerator:
    """Generates Python bindings for MCP tools.

    Creates Python modules that can be imported and used to call MCP tools
    directly from code, enabling efficient tool composition.

    Example:
        generator = PythonCodeGenerator("./generated")
        generator.generate_from_tools({"bash__ls": tool_def, "bash__cat": tool_def})

        # Generated code can be imported:
        # from generated.servers.bash import ls, cat
        # result = await ls({"path": "/tmp"})
    """

    def __init__(self, output_path: str = "./generated"):
        """Initialize the code generator.

        Args:
            output_path: Directory to write generated code.
        """
        self.output_path = Path(output_path)
        self.servers_path = self.output_path / "servers"

    def generate_from_tools(self, tools: dict[str, ToolDefinition]) -> None:
        """Generate Python bindings for all tools.

        Args:
            tools: Dictionary mapping tool names to definitions.
        """
        # Group tools by server
        server_tools: dict[str, list[ToolDefinition]] = {}
        for name, tool in tools.items():
            server = tool.server_name or name.split("__")[0]
            if server not in server_tools:
                server_tools[server] = []
            server_tools[server].append(tool)

        # Create directory structure
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.servers_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Generating MCP bindings to %s (servers dir: %s)",
            self.output_path,
            self.servers_path,
        )

        # Generate client module
        self._generate_client_module()

        # Generate server modules
        for server_name, tools_list in server_tools.items():
            logger.info(
                "Generating bindings for server '%s' into %s",
                server_name,
                self.servers_path / server_name,
            )
            self._generate_server_module(server_name, tools_list)

        # Generate index module
        self._generate_index_module(list(server_tools.keys()))

    def _generate_client_module(self) -> None:
        """Generate the client module for making tool calls."""
        client_code = '''# Auto-generated MCP tool client
# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Client for calling MCP tools."""

from typing import Any, TypeVar

T = TypeVar("T")

# Global tool caller - set by the executor
_tool_caller = None


def set_tool_caller(caller) -> None:
    """Set the global tool caller function.
    
    Args:
        caller: Async function that takes (tool_name, arguments) and returns result.
    """
    global _tool_caller
    _tool_caller = caller


async def call_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
    """Call an MCP tool.
    
    Args:
        tool_name: Full tool name (server__toolname format).
        arguments: Tool arguments.
        
    Returns:
        Tool result.
        
    Raises:
        RuntimeError: If no tool caller is configured.
    """
    if _tool_caller is None:
        raise RuntimeError(
            "No tool caller configured. "
            "Use set_tool_caller() or run through CodeModeExecutor."
        )
    result = await _tool_caller(tool_name, arguments)
    
    # helper to check if something is a list
    if not isinstance(result, (dict, object)) or result is None:
        return result

    # Extract content list
    content_list = None
    if isinstance(result, dict):
        content_list = result.get("content")
    elif hasattr(result, "content"):
        content_list = result.content
    
    if not isinstance(content_list, list):
        return result

    # Concatenate text parts
    text_content = ""
    has_text = False
    
    for part in content_list:
        part_type = None
        part_text = None
        
        if isinstance(part, dict):
            part_type = part.get("type")
            part_text = part.get("text")
        elif hasattr(part, "type") and hasattr(part, "text"):
            part_type = part.type
            part_text = part.text
            
        if part_type == "text" and part_text is not None:
            text_content += part_text
            has_text = True
            
    if has_text:
        # Try to parse as JSON first, as many tools return JSON string
        try:
            import json
            return json.loads(text_content)
        except Exception:
            return text_content
            
    return result
'''
        client_path = self.output_path / "client.py"
        client_path.write_text(client_code)

    def _generate_server_module(
        self, server_name: str, tools: list[ToolDefinition]
    ) -> None:
        """Generate a module for a server's tools.

        Args:
            server_name: Name of the server.
            tools: List of tool definitions.
        """
        server_dir = self.servers_path / server_name
        server_dir.mkdir(parents=True, exist_ok=True)

        # Generate individual tool files
        for tool in tools:
            self._generate_tool_file(server_dir, server_name, tool)

        # Generate server index
        self._generate_server_index(server_dir, server_name, tools)

    def _generate_tool_file(
        self, server_dir: Path, server_name: str, tool: ToolDefinition
    ) -> None:
        """Generate a file for a single tool.

        Args:
            server_dir: Server directory path.
            server_name: Server name.
            tool: Tool definition.
        """
        # Extract tool name without server prefix
        if tool.name.startswith(f"{server_name}__"):
            short_name = tool.name[len(server_name) + 2:]
        else:
            short_name = tool.name

        # Sanitize function name
        func_name = self._sanitize_name(short_name)

        # Generate type hints from schema
        input_type = self._schema_to_type_hint(tool.input_schema)
        output_type = "Any"  # Could be improved with output schema

        # Generate docstring
        docstring = self._generate_docstring(tool)

        # Generate the function
        code = f'''# Auto-generated tool binding for {tool.name}
# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Tool: {tool.name}"""

from typing import Any, Optional
from ...client import call_tool


async def {func_name}(arguments: Optional[{input_type}] = None, **kwargs: Any) -> {output_type}:
    """{docstring}"""
    if arguments is None:
        arguments = kwargs
    else:
        arguments.update(kwargs)
    return await call_tool("{tool.name}", arguments)


# Convenience alias
{func_name}_sync = None  # Sync version can be added if needed
'''

        tool_path = server_dir / f"{func_name}.py"
        tool_path.write_text(code)

    def _generate_server_index(
        self, server_dir: Path, server_name: str, tools: list[ToolDefinition]
    ) -> None:
        """Generate the server index file.

        Args:
            server_dir: Server directory path.
            server_name: Server name.
            tools: List of tool definitions.
        """
        imports = []
        exports = []

        for tool in tools:
            if tool.name.startswith(f"{server_name}__"):
                short_name = tool.name[len(server_name) + 2:]
            else:
                short_name = tool.name
            func_name = self._sanitize_name(short_name)
            imports.append(f"from .{func_name} import {func_name}")
            exports.append(f'    "{func_name}",')

        code = f'''# Auto-generated server module for {server_name}
# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Tools from {server_name} server."""

{chr(10).join(imports)}

__all__ = [
{chr(10).join(exports)}
]
'''

        index_path = server_dir / "__init__.py"
        index_path.write_text(code)

    def _generate_index_module(self, server_names: list[str]) -> None:
        """Generate the main index module.

        Args:
            server_names: List of server names.
        """
        imports = []
        for name in server_names:
            imports.append(f"from .servers import {name}")

        code = f'''# Auto-generated MCP tool bindings index
# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Generated MCP tool bindings.

Import tools from server modules:
    from generated.servers.bash import ls, cat
    from generated.servers.computer import screenshot
"""

from .client import call_tool, set_tool_caller

__all__ = [
    "call_tool",
    "set_tool_caller",
]
'''

        index_path = self.output_path / "__init__.py"
        index_path.write_text(code)

        # Also create servers/__init__.py
        servers_index = f'''# Auto-generated servers index
# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Server modules."""

{chr(10).join(f"from . import {name}" for name in server_names)}

__all__ = {server_names!r}
'''

        servers_index_path = self.servers_path / "__init__.py"
        servers_index_path.write_text(servers_index)

    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name to be a valid Python identifier.

        Args:
            name: Original name.

        Returns:
            Valid Python identifier.
        """
        # Replace invalid characters with underscores
        result = ""
        for i, char in enumerate(name):
            if char.isalnum() or char == "_":
                result += char
            else:
                result += "_"

        # Ensure it doesn't start with a number
        if result and result[0].isdigit():
            result = "_" + result

        # Handle Python keywords
        import keyword
        if keyword.iskeyword(result):
            result = result + "_"

        return result or "_unnamed"

    def _schema_to_type_hint(self, schema: dict[str, Any]) -> str:
        """Convert JSON Schema to Python type hint.

        Args:
            schema: JSON Schema object.

        Returns:
            Python type hint string.
        """
        if not schema:
            return "dict[str, Any]"

        schema_type = schema.get("type", "object")

        if schema_type == "object":
            return "dict[str, Any]"
        elif schema_type == "array":
            items = schema.get("items", {})
            item_type = self._schema_to_type_hint(items)
            return f"list[{item_type}]"
        elif schema_type == "string":
            return "str"
        elif schema_type == "number":
            return "float"
        elif schema_type == "integer":
            return "int"
        elif schema_type == "boolean":
            return "bool"
        elif schema_type == "null":
            return "None"
        else:
            return "Any"

    def _generate_docstring(self, tool: ToolDefinition) -> str:
        """Generate a docstring for a tool function.

        Args:
            tool: Tool definition.

        Returns:
            Docstring content.
        """
        lines = [tool.description or f"Call {tool.name} tool."]
        lines.append("")
        lines.append("Args:")
        lines.append("    arguments: Tool input arguments.")

        params = tool.parameters
        if params:
            lines.append("")
            lines.append("Input schema properties:")
            for param in params:
                req = " (required)" if param.required else ""
                lines.append(f"    - {param.name}: {param.type}{req}")
                if param.description:
                    lines.append(f"      {param.description}")

        lines.append("")
        if tool.input_examples:
            import json

            lines.append("")
            lines.append("Examples:")
            for example in tool.input_examples[:3]:
                lines.append(f"    {json.dumps(example, ensure_ascii=False)}")

        lines.append("")
        lines.append("Returns:")
        lines.append("    Tool execution result.")

        return "\n    ".join(lines)
