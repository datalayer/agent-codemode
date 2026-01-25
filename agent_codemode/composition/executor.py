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

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Optional

from code_sandboxes import Sandbox, Execution, SandboxConfig

from ..discovery.registry import ToolRegistry
from ..discovery.codegen import PythonCodeGenerator
from ..models import CodeModeConfig, ToolCallResult


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
            from generated.servers.bash import ls, cat

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

    @property
    def sandbox(self) -> Optional[Sandbox]:
        """Get the sandbox instance."""
        return self._sandbox

    async def setup(self) -> None:
        """Set up the executor.

        This generates code bindings for all registered tools and
        prepares the sandbox environment.
        """
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

        # Add generated path to sys.path and clear any stale module cache
        setup_code = f'''
import sys
generated_path = {str(generated_path)!r}
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
'''
        self._sandbox.run_code(setup_code)

        # Set up the tool caller
        self._sandbox.set_variable("__tool_registry__", self.registry)
        self._sandbox.set_variable("__executor__", self)

        # Inject the tool caller function
        caller_code = '''
async def __call_tool__(tool_name, arguments):
    """Call an MCP tool through the registry."""
    return await __executor__.call_tool(tool_name, arguments)

# Set up the generated client to use our caller
try:
    from generated.client import set_tool_caller
    set_tool_caller(__call_tool__)
except ImportError:
    pass
'''
        self._sandbox.run_code(caller_code)

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
    ) -> Execution:
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

        import sys
        # print(f"\n[EXECUTOR] execute() called with code length={len(code)}", file=sys.stderr, flush=True)

        self._in_execute = True
        self._tool_calls_in_run = 0

        try:
            # Get identity environment variables from request context
            identity_env = _get_identity_env()
            
            # Get the generated path for sys.path setup
            generated_path = str(Path(self.config.generated_path).resolve())

            # Ensure executor is available in sandbox
            self._sandbox.set_variable("__executor__", self)
            
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

    __generated_init__ = os.path.join(__generated_path__, "__init__.py")
    __generated_spec__ = importlib.util.spec_from_file_location(
        "generated",
        __generated_init__,
        submodule_search_locations=[__generated_path__],
    )
    if __generated_spec__ and __generated_spec__.loader:
        __generated_module__ = importlib.util.module_from_spec(__generated_spec__)
        sys.modules["generated"] = __generated_module__
        __generated_spec__.loader.exec_module(__generated_module__)
except Exception:
    pass

# Define tool caller wrapper that uses the executor
async def __call_tool__(tool_name, arguments):
    """Call an MCP tool through the registry."""
    return await __executor__.call_tool(tool_name, arguments)

# Ensure tool caller is configured
try:
    from generated.client import set_tool_caller
    set_tool_caller(__call_tool__)
except ImportError:
    pass
'''
            # Run setup first
            import sys
            # print("[EXECUTOR DEBUG] Starting setup_code execution...", file=sys.stderr, flush=True)
            self._sandbox.run_code(setup_code)
            
            # For async code, we need to handle it specially to avoid event loop conflicts
            if "await " in code or "async " in code:
                # Get the namespace and execute async code directly in current loop
                namespace = self._sandbox._namespaces[self._sandbox._default_context.id]
                
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
                import io
                from contextlib import redirect_stdout, redirect_stderr
                stdout_buffer = io.StringIO()
                stderr_buffer = io.StringIO()
                
                # Call the async function directly (we're already in async context)
                with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                    coro = namespace["__user_code__"]()
                    locals_value = await coro
                
                # Update namespace with returned locals
                if isinstance(locals_value, dict):
                    for key, value in locals_value.items():
                        if key in ("__builtins__", "__name__", "__doc__", "__package__", 
                                 "__loader__", "__spec__", "__annotations__", "__cached__",
                                 "__file__"):
                            continue
                        namespace[key] = value
                
                # Create execution result with captured output
                from code_sandboxes.models import Execution, Logs, OutputMessage
                import time
                
                stdout_lines = stdout_buffer.getvalue().splitlines()
                stderr_lines = stderr_buffer.getvalue().splitlines()
                timestamp = time.time()
                
                result = Execution(
                    error=None,
                    results=[],
                    logs=Logs(
                        stdout=[OutputMessage(line=line, timestamp=timestamp, error=False) for line in stdout_lines],
                        stderr=[OutputMessage(line=line, timestamp=timestamp, error=True) for line in stderr_lines],
                    ),
                    execution_count=self._sandbox._execution_count[self._sandbox._default_context.id],
                    context_id=self._sandbox._default_context.id,
                )
                
                # print("[EXECUTOR DEBUG] Async code completed", file=sys.stderr, flush=True)
            else:
                # Then run user code (sandbox will handle sync code properly)
                # print(f"[EXECUTOR DEBUG] Starting sync code execution (length={len(code)})...", file=sys.stderr, flush=True)
                result = self._sandbox.run_code(code, timeout=timeout)
                # print("[EXECUTOR DEBUG] Sync code completed", file=sys.stderr, flush=True)
            
            return result
        finally:
            self._in_execute = False

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
    ) -> Execution:
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
