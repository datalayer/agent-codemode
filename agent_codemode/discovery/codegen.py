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

from ..types import ToolDefinition


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
        self.mcp_path = self.servers_path / "mcp"

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
        self.mcp_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Generating MCP bindings to %s (mcp dir: %s)",
            self.output_path,
            self.mcp_path,
        )

        # Generate client module
        self._generate_client_module()

        # Generate server modules under servers/mcp/
        for server_name, tools_list in server_tools.items():
            logger.info(
                "Generating bindings for server '%s' into %s",
                server_name,
                self.mcp_path / server_name,
            )
            self._generate_server_module(server_name, tools_list)

        # Generate index modules
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

    # Check for error response
    is_error = False
    if isinstance(result, dict):
        is_error = result.get("isError", False)
    elif hasattr(result, "isError"):
        is_error = result.isError

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
    
    # If this was an error response, raise an exception
    if is_error and has_text:
        raise RuntimeError(text_content)
            
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
        server_dir = self.mcp_path / server_name
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
        output_type = self._schema_to_type_hint(tool.output_schema) if tool.output_schema else "Any"

        # Generate docstring
        docstring = self._generate_docstring(tool)

        # Check if this tool has the pattern of tool_name + arguments parameters
        needs_flexible_params = self._needs_flexible_parameters(tool.input_schema)

        # Generate the function
        if needs_flexible_params:
            # Extract the main parameter names from the schema
            schema_props = tool.input_schema.get("properties", {}) if tool.input_schema else {}
            param_names = list(schema_props.keys())
            
            # Build parameter list for flexible functions
            param_list = []
            for prop_name, prop_def in schema_props.items():
                param_type = self._schema_property_to_type_hint(prop_def)
                param_list.append(f"    {prop_name}: Optional[{param_type}] = None")
            
            param_signature = ",\n".join(param_list) + ",\n    **kwargs: Any" if param_list else "**kwargs: Any"
            
            # Generate flexible call logic
            call_logic = self._generate_flexible_call_logic(param_names)
            
            code = f'''# Auto-generated tool binding for {tool.name}
# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Tool: {tool.name}"""

from typing import Any, Optional
from ....client import call_tool


async def {func_name}(
{param_signature}
) -> {output_type}:
    """{docstring}"""
{call_logic}
    return await call_tool("{tool.name}", call_args)


# Convenience alias
{func_name}_sync = None  # Sync version can be added if needed
'''
        else:
            # Standard function generation
            code = f'''# Auto-generated tool binding for {tool.name}
# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Tool: {tool.name}"""

from typing import Any, Optional
from ....client import call_tool


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
        code = '''# Auto-generated MCP tool bindings index
# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Generated tool bindings.

Import MCP tools from server modules:
    from generated.servers.mcp.bash import ls, cat
    from generated.servers.mcp.computer import screenshot

Import skill tools (when skills are enabled):
    from generated.servers.skills import list_skills, run_skill
"""

from .client import call_tool, set_tool_caller

__all__ = [
    "call_tool",
    "set_tool_caller",
]
'''

        index_path = self.output_path / "__init__.py"
        index_path.write_text(code)

        # Create servers/__init__.py (namespace package for mcp/ and skills/)
        servers_index = '''# Auto-generated servers index
# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Server modules (mcp and skills)."""

__all__: list[str] = []
'''
        servers_index_path = self.servers_path / "__init__.py"
        servers_index_path.write_text(servers_index)

        # Create servers/mcp/__init__.py with MCP server imports
        mcp_index = f'''# Auto-generated MCP servers index
# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""MCP server modules."""

{chr(10).join(f"from . import {name}" for name in server_names)}

__all__ = {server_names!r}
'''
        mcp_index_path = self.mcp_path / "__init__.py"
        mcp_index_path.write_text(mcp_index)

    def generate_skill_bindings(
        self,
        skills: list[dict[str, Any]],
    ) -> None:
        """Generate Python bindings for skills under servers/skills/.

        Each skill becomes a callable async function that routes through
        the same ``call_tool`` / ``__call_tool__`` mechanism used by MCP
        tools, with tool names prefixed by ``skills__``.

        Generated functions:

        * ``list_skills()`` – returns JSON list of available skills
        * ``load_skill(skill_name)`` – returns full SKILL.md content
        * ``read_skill_resource(skill_name, resource_name)`` – reads a resource
        * ``run_skill(skill_name, script_name, args)`` – executes a script

        Args:
            skills: List of skill metadata dicts, each with at least
                ``name``, ``description``, and optionally ``scripts``
                and ``resources``.
        """
        skills_dir = self.servers_path / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Generating skill bindings for %d skills into %s",
            len(skills),
            skills_dir,
        )

        # Embed the skill catalog as a constant so list_skills is self-contained
        import json as _json
        skill_catalog_json = _json.dumps(skills, ensure_ascii=False, indent=2)

        # --- list_skills ---
        list_skills_code = f'''# Auto-generated skill binding
# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""List all available skills."""

from typing import Any
from ...client import call_tool

_SKILL_CATALOG = {skill_catalog_json}


async def list_skills() -> list[dict[str, Any]]:
    """List all available skills with their names and descriptions.

    Returns a list of skill metadata dictionaries. Each entry contains
    at least ``name`` and ``description`` keys, and may include
    ``scripts`` and ``resources``.

    This is a local lookup – no tool call is made.
    """
    return _SKILL_CATALOG
'''
        (skills_dir / "list_skills.py").write_text(list_skills_code)

        # --- load_skill ---
        load_skill_code = '''# Auto-generated skill binding
# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Load full skill content."""

from typing import Any
from ...client import call_tool


async def load_skill(skill_name: str) -> Any:
    """Load the full content and instructions for a skill.

    Args:
        skill_name: Name of the skill to load.

    Returns:
        Full SKILL.md content as a string.
    """
    return await call_tool("skills__load_skill", {"skill_name": skill_name})
'''
        (skills_dir / "load_skill.py").write_text(load_skill_code)

        # --- read_skill_resource ---
        read_resource_code = '''# Auto-generated skill binding
# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Read a skill resource."""

from typing import Any
from ...client import call_tool


async def read_skill_resource(skill_name: str, resource_name: str) -> Any:
    """Read a resource file from a skill.

    Args:
        skill_name: Name of the skill.
        resource_name: Name of the resource to read.

    Returns:
        Resource content as a string.
    """
    return await call_tool(
        "skills__read_skill_resource",
        {"skill_name": skill_name, "resource_name": resource_name},
    )
'''
        (skills_dir / "read_skill_resource.py").write_text(read_resource_code)

        # --- run_skill ---
        run_skill_code = '''# Auto-generated skill binding
# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Run a skill script."""

from typing import Any
from ...client import call_tool


async def run_skill(
    skill_name: str,
    script_name: str,
    args: list[str] | None = None,
) -> Any:
    """Execute a script from a skill with arguments.

    The result is a dict with keys: ``output``, ``exit_code``,
    ``success``, ``error``, ``error_type``, ``error_value``,
    ``error_traceback``, ``execution_time``, ``script_name``.

    Args:
        skill_name: Name of the skill.
        script_name: Name of the script to run.
        args: Arguments to pass to the script (default: []).

    Returns:
        ScriptExecutionResult dict.
    """
    return await call_tool(
        "skills__run_skill_script",
        {
            "skill_name": skill_name,
            "script_name": script_name,
            "args": args or [],
        },
    )
'''
        (skills_dir / "run_skill.py").write_text(run_skill_code)

        # --- __init__.py ---
        init_code = '''# Auto-generated skills module
# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Skill bindings – import and call skills from execute_code.

Example::

    from generated.servers.skills import list_skills, run_skill

    skills = await list_skills()
    result = await run_skill("pdf-extractor", "extract", ["report.pdf"])
"""

from .list_skills import list_skills
from .load_skill import load_skill
from .read_skill_resource import read_skill_resource
from .run_skill import run_skill

__all__ = [
    "list_skills",
    "load_skill",
    "read_skill_resource",
    "run_skill",
]
'''
        (skills_dir / "__init__.py").write_text(init_code)

        logger.info("Generated skill bindings: list_skills, load_skill, read_skill_resource, run_skill")
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
        if tool.output_schema:
            lines.append("    Tool execution result with the following structure:")
            self._add_schema_description(lines, tool.output_schema, "    ")
        else:
            lines.append("    Tool execution result.")

        return "\n    ".join(lines)
    
    def _add_schema_description(self, lines: list[str], schema: dict[str, Any], indent: str) -> None:
        """Add schema description to docstring lines.
        
        Args:
            lines: List of docstring lines to append to
            schema: JSON Schema to describe
            indent: Indentation prefix for schema lines
        """
        if not schema:
            return
            
        schema_type = schema.get("type")
        if schema_type == "object" and "properties" in schema:
            lines.append(f"{indent}Object with properties:")
            for prop_name, prop_def in schema["properties"].items():
                prop_type = prop_def.get("type", "any")
                prop_desc = prop_def.get("description", "")
                req_marker = " (required)" if prop_name in schema.get("required", []) else ""
                if prop_desc:
                    lines.append(f"{indent}  - {prop_name}: {prop_type}{req_marker} - {prop_desc}")
                else:
                    lines.append(f"{indent}  - {prop_name}: {prop_type}{req_marker}")
        elif schema_type == "array" and "items" in schema:
            lines.append(f"{indent}Array of items:")
            self._add_schema_description(lines, schema["items"], indent + "  ")
        elif schema_type:
            desc = schema.get("description", "")
            if desc:
                lines.append(f"{indent}{schema_type} - {desc}")
            else:
                lines.append(f"{indent}{schema_type}")

    def _needs_flexible_parameters(self, schema: dict[str, Any]) -> bool:
        """Check if a tool schema needs flexible parameter handling.
        
        Returns True if the schema has properties that suggest it needs
        individual parameter handling (like tool_name + arguments pattern).
        
        Args:
            schema: Tool input schema
            
        Returns:
            True if flexible parameters are needed
        """
        if not schema or "properties" not in schema:
            return False
            
        props = schema["properties"]
        
        # Check for common patterns that need flexible handling:
        # 1. Has both "tool_name" and "arguments" 
        # 2. Has "function_name" or "method_name" with "arguments"/"parameters"
        # 3. Any combination that suggests nested tool calling
        
        has_tool_identifier = any(key in props for key in [
            "tool_name", "function_name", "method_name", "action", "command"
        ])
        has_arguments = any(key in props for key in [
            "arguments", "parameters", "args", "params", "input", "data"
        ])
        
        return has_tool_identifier and has_arguments
    
    def _schema_property_to_type_hint(self, prop_def: dict[str, Any]) -> str:
        """Convert a single JSON Schema property to Python type hint.
        
        Args:
            prop_def: Property definition from JSON Schema
            
        Returns:
            Python type hint string
        """
        prop_type = prop_def.get("type", "any")
        
        if prop_type == "string":
            return "str"
        elif prop_type == "number":
            return "float"
        elif prop_type == "integer":
            return "int"
        elif prop_type == "boolean":
            return "bool"
        elif prop_type == "array":
            return "list[Any]"
        elif prop_type == "object":
            return "dict[str, Any]"
        else:
            return "Any"
    
    def _generate_flexible_call_logic(self, param_names: list[str]) -> str:
        """Generate the call logic for flexible parameter functions.
        
        Args:
            param_names: List of parameter names from schema
            
        Returns:
            Indented Python code for parameter handling
        """
        lines = [
            "    # Support both calling styles:",
            "    # 1. Natural keyword arguments: func(param1=val1, param2=val2)",  
            "    # 2. Single arguments dict: func(arguments={'param1': val1, 'param2': val2})",
            "    ",
            "    # Check if any schema parameters were provided directly",
            f"    direct_params = {{k: v for k, v in locals().items() if k in {param_names!r} and v is not None}}",
            "    ",
            "    if direct_params:",
            "        # Style 1: Direct parameters provided",
            "        call_args = direct_params",
            "        call_args.update(kwargs)",
            "    elif 'arguments' in kwargs and isinstance(kwargs['arguments'], dict):",
            "        # Style 2: Single arguments dict provided", 
            "        call_args = kwargs['arguments'].copy()",
            "        # Add any other kwargs that aren't 'arguments'",
            "        call_args.update({k: v for k, v in kwargs.items() if k != 'arguments'})",
            "    else:",
            "        # Fallback: use kwargs as arguments",
            "        call_args = kwargs",
        ]
        
        return "\n".join(lines)
