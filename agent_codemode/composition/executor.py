# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""CodeMode Executor - Execute code that composes MCP tools.

This is the core component that enables programmatic tool composition,
running code that imports and calls generated tool bindings without
LLM inference overhead.

Identity Context Support:
    The executor supports automatic injection of OAuth identity tokens
    into the execution environment. When identities are set in the request
    context (via agent_runtimes.context.identities), they are automatically
    made available as environment variables in the sandbox:
    
    - GITHUB_TOKEN for GitHub OAuth
    - GITLAB_TOKEN for GitLab OAuth
    - GOOGLE_ACCESS_TOKEN for Google OAuth
    - AZURE_ACCESS_TOKEN for Microsoft OAuth
    
    This allows code executed via execute_code to access authenticated
    APIs without explicitly passing credentials.
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional

from code_sandboxes import Sandbox, ExecutionResult, SandboxConfig

from ..discovery.registry import ToolRegistry
from ..discovery.codegen import PythonCodeGenerator
from ..types import CodeModeConfig, ToolCallResult

logger = logging.getLogger(__name__)


def _get_identity_env() -> dict[str, str]:
    """Get identity environment variables from request context.
    
    This function attempts to import the identity context from agent_runtimes.
    If not available (standalone codemode usage), returns empty dict.
    
    Returns:
        Dictionary of environment variable names to token values.
    """
    try:
        from agent_runtimes.context.identities import get_identity_env
        return get_identity_env()
    except ImportError:
        return {}


class CodeModeExecutor:
    """Execute code that composes MCP tools programmatically.

    The CodeModeExecutor provides an environment where code can import
    and call MCP tools directly, without going through LLM inference
    for each tool call.

    Key benefits:
    - Tool composition: Chain multiple tools in code
    - State persistence: Store variables and reuse results
    - Control flow: Loops, conditionals, error handling
    - Efficiency: Many tool calls in one execution

    Example:
        registry = ToolRegistry()
        registry.add_server(MCPServerConfig(name="bash", url="..."))
        await registry.discover_all()

        executor = CodeModeExecutor(registry=registry)
        await executor.setup()

        result = await executor.execute('''
            from generated.mcp.bash import ls, cat

            files = await ls({"path": "/tmp"})
            for file in files["entries"]:
                content = await cat({"path": file})
                print(f"{file}: {len(content)} bytes")
        ''')
    """

    def __init__(
        self,
        registry: ToolRegistry,
        config: Optional[CodeModeConfig] = None,
        sandbox: Optional[Sandbox] = None,
    ):
        """Initialize the executor.

        Args:
            registry: Tool registry with discovered tools.
            config: Executor configuration.
            sandbox: Optional pre-configured sandbox. If not provided,
                creates one based on config.
        """
        self.registry = registry
        self.config = config or CodeModeConfig()
        self._sandbox = sandbox
        self._codegen = PythonCodeGenerator(self.config.generated_path)
        self._setup_done = False
        self._tool_call_history: list[ToolCallResult] = []
        self._in_execute = False
        self._tool_calls_in_run = 0
        self._skill_tool_caller: Any = None
        self._skills_metadata: list[dict[str, Any]] = []

    def set_skill_tool_caller(self, caller: Any) -> None:
        """Set a callback for routing skill tool calls.

        When generated skill bindings call ``call_tool("skills__<name>", args)``,
        the executor routes the call through this callback instead of the
        MCP tool registry.

        Args:
            caller: An async callable ``(tool_name, arguments) -> result``.
        """
        self._skill_tool_caller = caller

    def set_skills_metadata(self, metadata: list[dict[str, Any]]) -> None:
        """Store skill metadata for remote sandbox code generation.

        When a remote/Jupyter sandbox is used, the executor regenerates
        tool bindings inside the sandbox.  This metadata is used to also
        generate ``skills/`` bindings there.

        Called by ``wire_skills_into_codemode`` in agent_factory.
        """
        self._skills_metadata = metadata

    def _is_local_eval_sandbox(self) -> bool:
        """Check if the sandbox is a local-eval type (has in-memory namespaces).
        
        This checks the actual sandbox instance, not the config, to handle
        cases where an external sandbox is passed that differs from config.
        """
        return self._sandbox is not None and hasattr(self._sandbox, '_namespaces')

    @property
    def sandbox(self) -> Optional[Sandbox]:
        """Get the sandbox instance."""
        return self._sandbox

    async def setup(self) -> None:
        """Set up the executor.

        This generates code bindings for all registered tools and
        prepares the sandbox environment.
        """
        import sys as _sys
        print(f"[EXECUTOR.setup] Starting setup, sandbox_variant={self.config.sandbox_variant}", file=_sys.stderr)
        
        # Generate code bindings
        tools_dict = {tool.name: tool for tool in self.registry.list_tools()}
        self._codegen.generate_from_tools(tools_dict)

        # Create sandbox if not provided
        if self._sandbox is None:
            import os
            # Pass the complete environment to the sandbox
            env_vars = dict(os.environ)

            sandbox_config = SandboxConfig(
                timeout=self.config.sandbox_variant == "datalayer-runtime" and 300 or 30,
                working_dir=self.config.workspace_path,
                env_vars=env_vars,
            )
            sandbox_kwargs: dict[str, Any] = {}
            if self.config.sandbox_image:
                sandbox_kwargs["image"] = self.config.sandbox_image
            self._sandbox = Sandbox.create(
                variant=self.config.sandbox_variant,  # type: ignore
                config=sandbox_config,
                **sandbox_kwargs,
            )

        # Start the sandbox
        self._sandbox.start()

        # Set up the generated module path in the sandbox
        await self._setup_sandbox_environment()

        self._setup_done = True

    async def _setup_sandbox_environment(self) -> None:
        """Set up the sandbox environment for tool execution."""
        if self._sandbox is None:
            return

        generated_path = Path(self.config.generated_path).resolve()
        skills_path = Path(self.config.skills_path).resolve()

        # For Jupyter/remote sandboxes, generate tools directly in the sandbox
        # Use actual sandbox type detection, not config
        is_local_eval = self._is_local_eval_sandbox()
        if not is_local_eval:
            await self._generate_tools_in_sandbox()
            # Use /tmp so 'from generated.mcp...' works (files are at /tmp/generated/)
            sandbox_generated_path = "/tmp"
        else:
            # For local-eval, use the parent directory so 'from generated.mcp...' works
            # The generated_path might be './generated', we need its parent on sys.path
            sandbox_generated_path = str(generated_path.parent)

        # Add generated path and skills path to sys.path and clear any stale module cache
        setup_code = f'''
import sys
generated_path = {sandbox_generated_path!r}
skills_path = {str(skills_path)!r}

# Add generated path to sys.path
if generated_path not in sys.path:
    sys.path.insert(0, str(generated_path))
    # Clear any stale generated module cache
    for mod_name in list(sys.modules.keys()):
        if mod_name == "generated" or mod_name.startswith('generated.'):
            del sys.modules[mod_name]

    # Force-load the generated package from the configured path
    try:
        import importlib.util
        import os

        __generated_init__ = os.path.join(generated_path, "__init__.py")
        __generated_spec__ = importlib.util.spec_from_file_location(
            "generated",
            __generated_init__,
            submodule_search_locations=[generated_path],
        )
        if __generated_spec__ and __generated_spec__.loader:
            __generated_module__ = importlib.util.module_from_spec(__generated_spec__)
            sys.modules["generated"] = __generated_module__
            __generated_spec__.loader.exec_module(__generated_module__)
    except Exception:
        pass

# Add skills path to sys.path (for skills imports)
if skills_path not in sys.path:
    sys.path.insert(0, str(skills_path))
'''
        self._sandbox.run_code(setup_code)

        # Register tool caller with the sandbox
        import sys as _sys
        print(f"[SETUP ENV DEBUG] About to call register_tool_caller, sandbox={self._sandbox} id={id(self._sandbox)}", file=_sys.stderr)
        self._sandbox.register_tool_caller(self.call_tool)
        print(f"[SETUP ENV DEBUG] register_tool_caller called", file=_sys.stderr)
        
        # Verify __call_tool__ was set
        verify_code = '''
import sys
try:
    print(f"[VERIFY] __call_tool__ = {__call_tool__}", file=sys.stderr)
except NameError:
    print("[VERIFY] __call_tool__ NOT SET after register_tool_caller!", file=sys.stderr)
'''
        self._sandbox.run_code(verify_code)

        # For Jupyter/remote sandboxes, set up in-sandbox registry for tool calling
        # Use actual sandbox type detection, not config
        print(f"[SETUP ENV] is_local_eval={is_local_eval}, config.mcp_proxy_url={self.config.mcp_proxy_url}", file=_sys.stderr)
        if not is_local_eval:
            # =======================================================================
            # Two-Container Codemode Architecture
            # =======================================================================
            #
            # When using a remote sandbox (Jupyter kernel in another container or
            # process), the generated code needs to call MCP tools. There are two modes:
            #
            # 1. HTTP Proxy Mode (recommended for two-container setups):
            #    - mcp_proxy_url is set (e.g., "http://agent-runtimes:8765/api/v1/mcp/proxy")
            #    - Tool calls from the Jupyter kernel go via HTTP to the proxy endpoint
            #    - The proxy routes to the stdio MCP servers in agent-runtimes
            #
            #    ┌─────────────────────┐         ┌─────────────────────┐
            #    │  Jupyter Container  │  HTTP   │  Agent-Runtimes     │
            #    │  ┌───────────────┐  │ ──────▶ │  ┌───────────────┐  │
            #    │  │ Kernel        │  │         │  │ /mcp/proxy    │  │
            #    │  │ await tool()  │  │         │  │   ↓           │  │
            #    │  └───────────────┘  │         │  │ MCP Server    │  │
            #    └─────────────────────┘         │  │ (stdio)       │  │
            #                                    │  └───────────────┘  │
            #                                    └─────────────────────┘
            #
            # 2. Direct MCP Client Mode (legacy, requires agent-codemode in sandbox):
            #    - mcp_proxy_url is not set
            #    - Each MCP server config is serialized and MCPClient is created in sandbox
            #    - Requires stdio MCP processes accessible from the sandbox (not always possible)
            #
            # =======================================================================
            
            if self.config.mcp_proxy_url:
                # HTTP Proxy Mode: Use HTTP calls to agent-runtimes proxy endpoint
                proxy_url = self.config.mcp_proxy_url.rstrip("/")
                
                in_sandbox_http_caller_setup = f'''
# =======================================================================
# HTTP Proxy Tool Caller for Two-Container Codemode
# =======================================================================
#
# This code is injected into the Jupyter kernel to enable tool calls
# via HTTP to the agent-runtimes container's MCP proxy endpoint.
#
# Architecture:
#   Jupyter Kernel                    Agent-Runtimes
#   ─────────────                    ──────────────
#   await github__star_repo()   ───HTTP POST───▶  /api/v1/mcp/proxy/github/tools/star_repo
#                                                      │
#                                                      ▼
#                                              MCP Server (stdio)
#                                                      │
#                                                      ▼
#                               ◀───────────────  Tool Result
# =======================================================================

import httpx
import json
import sys
import asyncio

__MCP_PROXY_URL__ = "{proxy_url}"

async def __call_tool__(tool_name: str, arguments: dict) -> dict:
    """Call a tool via HTTP proxy to agent-runtimes.
    
    This function routes tool calls through the MCP proxy endpoint,
    which forwards them to the appropriate stdio MCP server.
    
    Args:
        tool_name: Full tool name in format "server__toolname" (e.g., "github__star_repo")
        arguments: Tool arguments dictionary
        
    Returns:
        Tool result dictionary
    """
    # Parse server name from tool_name (format: server__toolname)
    if "__" not in tool_name:
        return {{"isError": True, "content": [{{"type": "text", "text": f"Invalid tool name format: {{tool_name}}. Expected server__toolname"}}]}}
    
    server_name, original_tool_name = tool_name.split("__", 1)
    
    # Build the proxy URL
    # Format: /api/v1/mcp/proxy/{{server_name}}/tools/{{tool_name}}
    url = f"{{__MCP_PROXY_URL__}}/{{server_name}}/tools/{{original_tool_name}}"
    
    print(f"[HTTP Proxy] Calling tool: {{tool_name}} -> {{url}}", file=sys.stderr)
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            response = await client.post(
                url,
                json={{"arguments": arguments}},
                headers={{"Content-Type": "application/json"}},
            )
            
            if response.status_code == 404:
                return {{
                    "isError": True,
                    "content": [{{"type": "text", "text": f"MCP server '{{server_name}}' not found or tool '{{original_tool_name}}' not available"}}]
                }}
            
            if response.status_code != 200:
                return {{
                    "isError": True,
                    "content": [{{"type": "text", "text": f"HTTP {{response.status_code}}: {{response.text}}"}}]
                }}
            
            result = response.json()
            
            # Convert proxy response to MCP format
            if result.get("success", False):
                return {{
                    "isError": False,
                    "content": [{{"type": "text", "text": json.dumps(result.get("result", {{}})) if isinstance(result.get("result"), (dict, list)) else str(result.get("result", ""))}}]
                }}
            else:
                return {{
                    "isError": True,
                    "content": [{{"type": "text", "text": result.get("error", "Unknown error")}}]
                }}
                
    except httpx.ConnectError as e:
        return {{
            "isError": True,
            "content": [{{"type": "text", "text": f"Connection error to MCP proxy at {{__MCP_PROXY_URL__}}: {{e}}"}}]
        }}
    except Exception as e:
        return {{
            "isError": True,
            "content": [{{"type": "text", "text": f"HTTP proxy error: {{e}}"}}]
        }}

print(f"[SETUP] HTTP proxy tool caller configured for {{__MCP_PROXY_URL__}}", file=sys.stderr)
'''
                self._sandbox.run_code(in_sandbox_http_caller_setup)
            else:
                # Direct MCP Client Mode (legacy) - requires agent-codemode in sandbox
                # This mode is used when MCP servers can be accessed directly from the sandbox
                server_configs = [
                    {
                        "name": config.name,
                        "url": config.url,
                        "command": config.command,
                        "args": config.args,
                        "env": config.env,
                    }
                    for config in self.registry._servers.values()
                ]
                
                in_sandbox_registry_setup = f'''
try:
    from agent_codemode.proxy.mcp_client import MCPClient
except Exception as exc:
    raise RuntimeError(
        "agent_codemode.proxy.mcp_client is required in the sandbox environment. "
        "Consider using mcp_proxy_url for two-container setups."
    ) from exc

class _SandboxRegistry:
    """In-sandbox tool registry for Jupyter/remote execution."""
    def __init__(self, server_configs):
        self._clients = {{}}
        self._tools = {{}}
        for config in server_configs:
            client = MCPClient(
                name=config["name"],
                url=config["url"],
                command=config["command"],
                args=config["args"],
                env=config["env"],
            )
            self._clients[config["name"]] = client
    
    async def call_tool(self, tool_name, arguments):
        """Call a tool with arguments."""
        # Parse server name from tool_name (format: server__toolname)
        if "__" in tool_name:
            server_name, original_name = tool_name.split("__", 1)
        else:
            return {{"error": f"Invalid tool name format: {{tool_name}}"}}
        
        client = self._clients.get(server_name)
        if not client:
            return {{"error": f"Server not available: {{server_name}}"}}
        
        return await client.call_tool(original_name, arguments)

class _SandboxExecutor:
    """In-sandbox executor for tool calls."""
    def __init__(self, registry):
        self._registry = registry
    
    async def call_tool(self, tool_name, arguments):
        return await self._registry.call_tool(tool_name, arguments)

__sandbox_registry__ = _SandboxRegistry({server_configs!r})
__executor__ = _SandboxExecutor(__sandbox_registry__)

async def __call_tool__(tool_name, arguments):
    return await __sandbox_registry__.call_tool(tool_name, arguments)
'''
                self._sandbox.run_code(in_sandbox_registry_setup)

        # Set up the generated client to use __call_tool__
        caller_setup_code = '''
try:
    from generated.client import set_tool_caller
    set_tool_caller(__call_tool__)
except (ImportError, NameError) as e:
    import sys
    print(f"[SETUP] caller_setup_code error: {type(e).__name__}: {e}", file=sys.stderr)
'''
        self._sandbox.run_code(caller_setup_code)

    async def _generate_tools_in_sandbox(self) -> None:
        """Generate tool bindings directly in the remote sandbox.
        
        For Jupyter/remote sandboxes, we can't easily upload files, so instead
        we send the code generation logic to be executed in the sandbox.
        This way the generated modules exist in the sandbox's filesystem.
        """
        if self._sandbox is None:
            return
            
        # Get tool definitions for code generation
        tools_dict = {tool.name: tool for tool in self.registry.list_tools()}
        
        # Build serializable tool data
        tools_data = []
        for name, tool in tools_dict.items():
            tools_data.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
                "server_name": tool.server_name,
            })
        
        # Code to generate bindings in the sandbox
        generation_code = f'''
import os
import keyword
from pathlib import Path
from typing import Any

# Tool data from the registry
__tools_data__ = {tools_data!r}

# Output path in sandbox - use /tmp/generated so 'from generated.mcp...' works
__generated_path__ = Path("/tmp/generated")
__mcp_path__ = __generated_path__ / "mcp"

def _sanitize_name(name: str) -> str:
    """Sanitize a name to be a valid Python identifier."""
    result = ""
    for char in name:
        if char.isalnum() or char == "_":
            result += char
        else:
            result += "_"
    if result and result[0].isdigit():
        result = "_" + result
    if keyword.iskeyword(result):
        result = result + "_"
    return result or "_unnamed"

def _schema_to_type_hint(schema: dict) -> str:
    """Convert JSON Schema to Python type hint."""
    if not schema:
        return "dict[str, Any]"
    schema_type = schema.get("type", "object")
    if schema_type == "object":
        return "dict[str, Any]"
    elif schema_type == "array":
        return "list[Any]"
    elif schema_type == "string":
        return "str"
    elif schema_type == "number":
        return "float"
    elif schema_type == "integer":
        return "int"
    elif schema_type == "boolean":
        return "bool"
    return "Any"

# Create directory structure
__generated_path__.mkdir(parents=True, exist_ok=True)
__mcp_path__.mkdir(parents=True, exist_ok=True)

# Generate client module
__client_code__ = """# Auto-generated MCP tool client
from typing import Any, TypeVar

T = TypeVar("T")
_tool_caller = None

def set_tool_caller(caller) -> None:
    global _tool_caller
    _tool_caller = caller

async def call_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
    if _tool_caller is None:
        raise RuntimeError("No tool caller configured.")
    result = await _tool_caller(tool_name, arguments)
    
    if not isinstance(result, (dict, object)) or result is None:
        return result

    is_error = False
    if isinstance(result, dict):
        is_error = result.get("isError", False)
    elif hasattr(result, "isError"):
        is_error = result.isError

    content_list = None
    if isinstance(result, dict):
        content_list = result.get("content")
    elif hasattr(result, "content"):
        content_list = result.content
    
    if not isinstance(content_list, list):
        return result

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
    
    if is_error and has_text:
        raise RuntimeError(text_content)
            
    if has_text:
        try:
            import json
            return json.loads(text_content)
        except Exception:
            return text_content
            
    return result
"""
(__generated_path__ / "client.py").write_text(__client_code__)

# Group tools by server
__server_tools__: dict[str, list] = {{}}
for tool in __tools_data__:
    server = tool.get("server_name") or tool["name"].split("__")[0]
    if server not in __server_tools__:
        __server_tools__[server] = []
    __server_tools__[server].append(tool)

# Generate server modules
for server_name, tools_list in __server_tools__.items():
    server_dir = __mcp_path__ / server_name
    server_dir.mkdir(parents=True, exist_ok=True)
    
    imports = []
    exports = []
    
    for tool in tools_list:
        tool_name = tool["name"]
        if tool_name.startswith(f"{{server_name}}__"):
            short_name = tool_name[len(server_name) + 2:]
        else:
            short_name = tool_name
        
        func_name = _sanitize_name(short_name)
        input_type = _schema_to_type_hint(tool.get("input_schema", {{}}))
        output_type = _schema_to_type_hint(tool.get("output_schema")) if tool.get("output_schema") else "Any"
        description = tool.get("description", f"Call {{tool_name}} tool.")
        
        # Generate tool file
        tool_code = f"""# Auto-generated tool binding for {{tool_name}}
from typing import Any, Optional
from ...client import call_tool

async def {{func_name}}(arguments: Optional[{{input_type}}] = None, **kwargs: Any) -> {{output_type}}:
    \\"\\"\\"{{description}}\\"\\"\\"
    if arguments is None:
        arguments = kwargs
    else:
        arguments.update(kwargs)
    return await call_tool("{{tool_name}}", arguments)
"""
        (server_dir / f"{{func_name}}.py").write_text(tool_code)
        
        imports.append(f"from .{{func_name}} import {{func_name}}")
        exports.append(f'    "{{func_name}}",')
    
    # Generate server index
    server_index = f"""# Auto-generated server module for {{server_name}}
{{chr(10).join(imports)}}

__all__ = [
{{chr(10).join(exports)}}
]
"""
    (server_dir / "__init__.py").write_text(server_index)

# Generate main index
__main_index__ = """# Auto-generated MCP tool bindings index
from .client import call_tool, set_tool_caller

__all__ = ["call_tool", "set_tool_caller"]
"""
(__generated_path__ / "__init__.py").write_text(__main_index__)

# Generate MCP servers index
__server_names__ = list(__server_tools__.keys())
__mcp_index__ = f"""# Auto-generated MCP servers index
{{chr(10).join(f"from . import {{name}}" for name in __server_names__)}}

__all__ = {{__server_names__!r}}
"""
(__mcp_path__ / "__init__.py").write_text(__mcp_index__)

print(f"Generated tool bindings for {{len(__tools_data__)}} tools in {{__generated_path__}}")
'''
        self._sandbox.run_code(generation_code)

        # Generate skill bindings in the sandbox if skills metadata is available
        if self._skills_metadata:
            self.generate_skills_in_sandbox()

    def generate_skills_in_sandbox(self) -> None:
        """Generate skill bindings in the remote sandbox filesystem.

        This creates ``/tmp/generated/skills/`` with bindings that route
        ``skills__*`` tool calls back through ``call_tool`` (and ultimately
        through the HTTP proxy or local executor).

        Can be called independently after :meth:`_generate_tools_in_sandbox`
        has already run — for example when ``wire_skills_into_codemode`` sets
        the skills metadata *after* initial sandbox setup.
        """
        if self._sandbox is None or not self._skills_metadata:
            return

        # Skip for local-eval sandboxes (they use the on-disk generated files)
        if self._is_local_eval_sandbox():
            return

        skills_generation_code = f'''
import os
import sys
from pathlib import Path

__skills_metadata__ = {self._skills_metadata!r}
__generated_path__ = Path("/tmp/generated")
__skills_path__ = __generated_path__ / "skills"
__skills_path__.mkdir(parents=True, exist_ok=True)

# --- list_skills binding ---
__list_skills_code__ = """# Auto-generated skill binding for list_skills
from typing import Any, Optional
from ..client import call_tool

async def list_skills(arguments: Optional[dict[str, Any]] = None, **kwargs: Any) -> Any:
    \\"\\"\\"List all available skills and their descriptions.\\"\\"\\"
    if arguments is None:
        arguments = kwargs
    else:
        arguments.update(kwargs)
    return await call_tool("skills__list_skills", arguments)
"""
(__skills_path__ / "list_skills.py").write_text(__list_skills_code__)

# --- load_skill binding ---
__load_skill_code__ = """# Auto-generated skill binding for load_skill
from typing import Any, Optional
from ..client import call_tool

async def load_skill(arguments: Optional[dict[str, Any]] = None, **kwargs: Any) -> Any:
    \\"\\"\\"Load a skill by name and return its full description, scripts, and resources.\\"\\"\\"
    if arguments is None:
        arguments = kwargs
    else:
        arguments.update(kwargs)
    return await call_tool("skills__load_skill", arguments)
"""
(__skills_path__ / "load_skill.py").write_text(__load_skill_code__)

# --- read_skill_resource binding ---
__read_resource_code__ = """# Auto-generated skill binding for read_skill_resource
from typing import Any, Optional
from ..client import call_tool

async def read_skill_resource(arguments: Optional[dict[str, Any]] = None, **kwargs: Any) -> Any:
    \\"\\"\\"Read a resource file from a skill.\\"\\"\\"
    if arguments is None:
        arguments = kwargs
    else:
        arguments.update(kwargs)
    return await call_tool("skills__read_skill_resource", arguments)
"""
(__skills_path__ / "read_skill_resource.py").write_text(__read_resource_code__)

# --- run_skill binding ---
__run_skill_code__ = """# Auto-generated skill binding for run_skill
from typing import Any, Optional
from ..client import call_tool

async def run_skill(arguments: Optional[dict[str, Any]] = None, **kwargs: Any) -> Any:
    \\"\\"\\"Run a script from a skill.\\"\\"\\"
    if arguments is None:
        arguments = kwargs
    else:
        arguments.update(kwargs)
    return await call_tool("skills__run_skill_script", arguments)
"""
(__skills_path__ / "run_skill.py").write_text(__run_skill_code__)

# --- __init__.py for skills ---
__skills_init__ = """# Auto-generated skill bindings
from .list_skills import list_skills
from .load_skill import load_skill
from .read_skill_resource import read_skill_resource
from .run_skill import run_skill

__all__ = ["list_skills", "load_skill", "read_skill_resource", "run_skill"]
"""
(__skills_path__ / "__init__.py").write_text(__skills_init__)

# Clear stale generated.skills module cache so re-import picks up new files
for mod_name in list(sys.modules.keys()):
    if mod_name == "generated.skills" or mod_name.startswith("generated.skills."):
        del sys.modules[mod_name]

print(f"Generated skill bindings for {{len(__skills_metadata__)}} skills in {{__skills_path__}}")
'''
        self._sandbox.run_code(skills_generation_code)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool through the registry.

        This method is called by the generated tool bindings.

        Args:
            tool_name: Full tool name.
            arguments: Tool arguments.

        Returns:
            Tool result.
        """
        start_time = time.time()

        if self._in_execute and self.config.max_tool_calls is not None:
            if self._tool_calls_in_run >= self.config.max_tool_calls:
                raise RuntimeError(
                    f"Tool call limit exceeded ({self.config.max_tool_calls})."
                )
            self._tool_calls_in_run += 1

        try:
            # Route skill-prefixed calls to the skill tool caller
            if tool_name.startswith("skills__") and self._skill_tool_caller is not None:
                result = await self._skill_tool_caller(tool_name, arguments)
            else:
                result = await self.registry.call_tool(tool_name, arguments)
            execution_time = time.time() - start_time

            # Record in history
            self._tool_call_history.append(
                ToolCallResult(
                    tool_name=tool_name,
                    success=True,
                    result=result,
                    execution_time=execution_time,
                )
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self._tool_call_history.append(
                ToolCallResult(
                    tool_name=tool_name,
                    success=False,
                    error=str(e),
                    execution_time=execution_time,
                )
            )
            raise

    async def execute(
        self,
        code: str,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """Execute code that may use generated tool bindings.

        The code can import from the generated modules and call tools
        using async/await syntax.
        
        Identity tokens from the request context are automatically injected
        as environment variables (e.g., GITHUB_TOKEN, GITLAB_TOKEN).

        Args:
            code: Python code to execute.
            timeout: Execution timeout in seconds.

        Returns:
            Execution result.

        Raises:
            RuntimeError: If setup() hasn't been called.
        """
        if not self._setup_done or self._sandbox is None:
            raise RuntimeError("Executor not set up. Call setup() first.")

        self._in_execute = True
        self._tool_calls_in_run = 0

        try:
            # Get identity environment variables from request context
            identity_env = _get_identity_env()
            
            # Get the generated path for sys.path setup
            # For remote sandboxes, use /tmp so 'from generated.mcp...' works (files at /tmp/generated/)
            # For local-eval, use parent of generated_path so 'from generated.mcp...' works
            # Use actual sandbox type detection, not config
            is_local_eval = self._is_local_eval_sandbox()
            if not is_local_eval:
                generated_path = "/tmp"
            else:
                generated_path = str(Path(self.config.generated_path).resolve().parent)
            
            # Build identity injection code if we have tokens
            identity_injection = ""
            if identity_env:
                # Inject identity tokens as environment variables in the sandbox
                identity_injection = f"""
# Inject identity tokens from OAuth context
import os
os.environ.update({identity_env!r})
"""
            
            # Set up the environment before running user code
            setup_code = f'''{identity_injection}
import sys

# Ensure generated path is first on sys.path and purge stale generated modules
# __generated_path__ is the PARENT dir that contains 'generated/' folder
__generated_path__ = {generated_path!r}
if __generated_path__ in sys.path:
    sys.path.remove(__generated_path__)
sys.path.insert(0, __generated_path__)
for mod_name in list(sys.modules.keys()):
    if mod_name == "generated" or mod_name.startswith("generated."):
        del sys.modules[mod_name]

# Force-load the generated package from the configured path
try:
    import importlib.util
    import os

    # The 'generated' folder is a subdir of __generated_path__
    __generated_folder__ = os.path.join(__generated_path__, "generated")
    __generated_init__ = os.path.join(__generated_folder__, "__init__.py")
    __generated_spec__ = importlib.util.spec_from_file_location(
        "generated",
        __generated_init__,
        submodule_search_locations=[__generated_folder__],
    )
    if __generated_spec__ and __generated_spec__.loader:
        __generated_module__ = importlib.util.module_from_spec(__generated_spec__)
        sys.modules["generated"] = __generated_module__
        __generated_spec__.loader.exec_module(__generated_module__)
except Exception:
    pass
'''
            # Branch based on actual sandbox type (already computed above)
            if is_local_eval:
                # For local-eval, we can access _namespaces directly
                return await self._execute_local_eval(code, setup_code, timeout)
            else:
                # For Jupyter/remote sandboxes, use run_code()
                return await self._execute_jupyter(code, setup_code, timeout)
        finally:
            self._in_execute = False
    async def _execute_local_eval(
        self,
        code: str,
        setup_code: str,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """Execute code in local-eval sandbox with direct namespace access."""
        import sys
        import io
        import time
        from contextlib import redirect_stdout, redirect_stderr
        from code_sandboxes.models import ExecutionResult, Logs, OutputMessage
        
        # Get the namespace directly
        namespace = self._sandbox._namespaces[self._sandbox._default_context.id]
        
        # Execute setup_code directly in namespace (avoids async wrapper issues)
        exec(setup_code, namespace, namespace)
        
        # Configure the generated.client tool caller if available
        if '__call_tool__' in namespace:
            try:
                from generated.client import set_tool_caller
                set_tool_caller(namespace['__call_tool__'])
            except ImportError:
                pass
        
        # For async code, we need to handle it specially to avoid event loop conflicts
        if "await " in code or "async " in code:
            # Wrap user code in async function
            def _indent_code(value: str, spaces: int) -> str:
                indent = " " * spaces
                return "\n".join(indent + line for line in value.split("\n"))
            
            async_wrapper = f"""
async def __user_code__():
{_indent_code(code, 4)}
    return locals()
"""
            # Execute the wrapper in namespace
            exec(async_wrapper, namespace, namespace)
            
            # Capture stdout/stderr
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            
            exit_code = None

            # Call the async function directly (we're already in async context)
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                coro = namespace["__user_code__"]()
                try:
                    locals_value = await coro
                except SystemExit as exc:
                    if isinstance(exc.code, int):
                        exit_code = exc.code
                    elif exc.code:
                        exit_code = 1
                    else:
                        exit_code = 0
                    locals_value = {}
            
            # Update namespace with returned locals
            if isinstance(locals_value, dict):
                for key, value in locals_value.items():
                    if key in ("__builtins__", "__name__", "__doc__", "__package__", 
                             "__loader__", "__spec__", "__annotations__", "__cached__",
                             "__file__"):
                        continue
                    namespace[key] = value
            
            stdout_lines = stdout_buffer.getvalue().splitlines()
            stderr_lines = stderr_buffer.getvalue().splitlines()
            timestamp = time.time()
            
            result = ExecutionResult(
                execution_ok=True,
                code_error=None,
                exit_code=exit_code,
                results=[],
                logs=Logs(
                    stdout=[OutputMessage(line=line, timestamp=timestamp, error=False) for line in stdout_lines],
                    stderr=[OutputMessage(line=line, timestamp=timestamp, error=True) for line in stderr_lines],
                ),
                execution_count=self._sandbox._execution_count[self._sandbox._default_context.id],
                context_id=self._sandbox._default_context.id,
            )
        else:
            # For sync code, use sandbox's run_code
            result = self._sandbox.run_code(code, timeout=timeout)
        
        return result

    async def _execute_jupyter(
        self,
        code: str,
        setup_code: str,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """Execute code in Jupyter/remote sandbox using run_code().
        
        IMPORTANT: The sandbox.run_code() is synchronous and blocks waiting
        for the kernel to complete. When the kernel code calls back to the
        agent-runtimes server (e.g., via MCP proxy for tool calls), we need
        the event loop to be free to handle those requests. Therefore, we run
        the blocking code in a thread pool using asyncio.to_thread().
        
        This prevents the deadlock:
        1. FastAPI endpoint -> Jupyter sandbox (waiting)
        2. Jupyter sandbox -> MCP proxy HTTP request
        3. Without to_thread: DEADLOCK (event loop blocked)
        4. With to_thread: MCP proxy handles request, kernel continues
        """
        import asyncio
        
        # Run setup code in thread pool to avoid blocking event loop
        await asyncio.to_thread(self._sandbox.run_code, setup_code, timeout=timeout)
        
        # Re-register the tool caller since the module cache was cleared
        # The __call_tool__ function was defined during initial setup and persists
        # in the kernel's global namespace, but we need to re-wire it to the
        # freshly-loaded generated.client module
        tool_caller_rewire = '''
try:
    from generated.client import set_tool_caller
    set_tool_caller(__call_tool__)
except (ImportError, NameError) as e:
    import sys
    print(f"[EXECUTE] Failed to rewire tool caller: {type(e).__name__}: {e}", file=sys.stderr)
'''
        await asyncio.to_thread(self._sandbox.run_code, tool_caller_rewire, timeout=timeout)
        
        # Run user code in thread pool - this is where tool calls happen
        # and the kernel may call back to the MCP proxy
        return await asyncio.to_thread(self._sandbox.run_code, code, timeout=timeout)

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by a number of spaces.

        Args:
            code: Code to indent.
            spaces: Number of spaces.

        Returns:
            Indented code.
        """
        indent = " " * spaces
        lines = code.split("\n")
        return "\n".join(indent + line for line in lines)

    async def execute_skill(
        self,
        skill_name: str,
        arguments: Optional[dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute a saved skill.

        Args:
            skill_name: Name of the skill to execute.
            arguments: Optional arguments to pass to the skill.

        Returns:
            Execution result.
        """
        from agent_skills.simple import SimpleSkillsManager

        manager = SimpleSkillsManager(self.config.skills_path)
        skill = manager.load_skill(skill_name)

        if skill is None:
            raise ValueError(f"Skill not found: {skill_name}")

        # Set arguments as variables if provided
        if arguments and self._sandbox:
            for name, value in arguments.items():
                self._sandbox.set_variable(name, value)

        return await self.execute(skill.code)

    @property
    def tool_call_history(self) -> list[ToolCallResult]:
        """Get the history of tool calls."""
        return self._tool_call_history.copy()

    def clear_history(self) -> None:
        """Clear the tool call history."""
        self._tool_call_history.clear()

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._sandbox:
            self._sandbox.stop()
            self._sandbox = None
        self._setup_done = False

    async def __aenter__(self) -> "CodeModeExecutor":
        """Async context manager entry."""
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.cleanup()

    def __repr__(self) -> str:
        return f"CodeModeExecutor(registry={self.registry}, sandbox={self.config.sandbox_variant})"
