<!--
  ~ Copyright (c) 2025-2026 Datalayer, Inc.
  ~
  ~ BSD 3-Clause License
-->

[![Datalayer](https://assets.datalayer.tech/datalayer-25.svg)](https://datalayer.io)

[![Become a Sponsor](https://img.shields.io/static/v1?label=Become%20a%20Sponsor&message=%E2%9D%A4&logo=GitHub&style=flat&color=1ABC9C)](https://github.com/sponsors/datalayer)

# 🤖 👨‍💻 Agent Codemode

[![PyPI - Version](https://img.shields.io/pypi/v/agent-codemode)](https://pypi.org/project/agent-codemode)

**Generate programmatic tools from MCP Servers and Skills.**

## What is Agent Codemode?

Agent Codemode generates **programmatic tools** from two sources:

1. **MCP Servers** - Connect to any MCP server and generate typed Python bindings for its tools
2. **Skills** - Reusable code patterns that compose multiple tools into higher-level operations

These programmatic tools can be:

- **Used directly by an agent** - Import generated bindings and call tools from your agent's code
- **Exposed as an MCP Server** - Serve the generated tools via MCP protocol for any MCP-compatible client

<p align="center">
  <img src="https://images.datalayer.io/products/mcp-codemode/demo-claude-list-tools.png" width="55%" alt="" />
  <img src="https://images.datalayer.io/products/mcp-codemode/demo-claude-execute-code.png" width="44%" alt="" />
</p>

## Why Agent Codemode?

Traditional AI agents call tools one at a time through LLM inference. Agent Codemode enables a "Code Mode" pattern where agents write Python code that orchestrates multiple tool calls:

- **More efficient**: Single code generation instead of many LLM tool-call round-trips
- **More reliable**: Use try/except for robust error handling
- **More powerful**: Parallel execution with asyncio, loops, conditionals
- **More composable**: Save and reuse patterns as Skills

Same task, same MCP server — Code Mode uses significantly fewer tokens by composing tools in code instead of multiple LLM round-trips.

<table align="center">
  <tr>
    <th>Without Code Mode</th>
    <th>With Code Mode</th>
  </tr>
  <tr>
    <td><img src="https://images.datalayer.io/products/mcp-codemode/tokens-non-codemode.png" width="100%" alt="Without Code Mode" /></td>
    <td><img src="https://images.datalayer.io/products/mcp-codemode/tokens-codemode.png" width="100%" alt="With Code Mode" /></td>
  </tr>
</table>

<p align="center">
  <em>Prompt: "Generate 2000 words of random text and write to a file"</em>
</p>

## Configuration Highlights

| Option | Description |
|--------|-------------|
| `allow_direct_tool_calls` | When `False` (default), `call_tool` is hidden; all execution flows through `execute_code` |
| `max_tool_calls` | Safety cap limiting tool invocations per `execute_code` run |
| `sandbox_variant` | Sandbox type for code execution (default: `"local-eval"`) |
| `workspace_path` | Working directory for sandbox execution |
| `generated_path` | Path where tool bindings are generated |
| `skills_path` | Path for saved skills |

### Tool Discovery Options

- **`list_tool_names`**: Fast listing with `server`, `keywords`, `limit` filters. `include_deferred=False` by default
- **`search_tools`**: Natural language search with `query`, `server`, `limit`. `include_deferred=True` by default
- **`get_tool_details`**: Get full schema and documentation for a specific tool

### Tool Metadata

Tools include `output_schema` and `input_examples` to improve parameter accuracy. Tools marked with `defer_loading=True` are excluded from default listings but included in search results.

## Installation

```bash
pip install agent-codemode
```

## Quick Start

```python
from agent_codemode import ToolRegistry, CodeModeExecutor, MCPServerConfig

# Set up registry with MCP servers
registry = ToolRegistry()

# Add an MCP server (stdio transport - uses command/args)
registry.add_server(MCPServerConfig(
    name="filesystem",
    command="npx",
    args=["-y", "@anthropic/mcp-server-filesystem", "/tmp"]
))

# Or add an HTTP-based server
# registry.add_server(MCPServerConfig(name="web", url="http://localhost:8001"))

await registry.discover_all()

# Execute code that composes tools
async with CodeModeExecutor(registry) as executor:
    result = await executor.execute("""
        from generated.servers.filesystem import read_file, write_file
        
        # Read multiple files
        content1 = await read_file({"path": "/tmp/file1.txt"})
        content2 = await read_file({"path": "/tmp/file2.txt"})
        
        # Process and combine
        combined = content1 + "\\n---\\n" + content2
        
        # Write result
        await write_file({"path": "/tmp/combined.txt", "content": combined})
    """)
```

## Features

### Progressive Tool Discovery

Use the Tool Search Tool to discover relevant tools without loading all definitions upfront:

```python
# Search for tools matching a description (includes deferred tools by default)
result = await registry.search_tools("file operations", limit=10)

for tool in result.tools:
    print(f"{tool.name}: {tool.description}")

# Fast listing (deferred tools excluded by default)
names = registry.list_tool_names(limit=50)

# Include deferred tools explicitly
names_all = registry.list_tool_names(limit=50, include_deferred=True)
```

### Code-Based Tool Composition

Execute Python code in an isolated sandbox with auto-generated tool bindings:

```python
async with CodeModeExecutor(registry) as executor:
    execution = await executor.execute("""
        import asyncio
        from generated.servers.filesystem import ls, read_file
        
        # List all files
        files = await ls({"path": "/data"})
        
        # Read all files in parallel
        contents = await asyncio.gather(*[
            read_file({"path": f}) for f in files
        ])
    """, timeout=30.0)

# Outputs are available on the execution object
print(execution.stdout)
print(execution.stderr)
print(execution.text)
print(execution.success)
```

### 🧰 Skills (Reusable Compositions)

Skills are Python files that compose tools into reusable operations. This allows agents to evolve their own toolbox by saving useful code patterns. Skills functionality is provided by the [agent-skills](https://github.com/datalayer/agent-skills) package.

Note: Skills APIs are owned by agent-skills. Import skill utilities from `agent_skills`.

#### Creating Skills as Code Files

The primary pattern is skills as Python files in a `skills/` directory:

```python
# skills/batch_process.py
"""Process all files in a directory."""

async def batch_process(input_dir: str, output_dir: str) -> dict:
    """Process all files in a directory.
    
    Args:
        input_dir: Input directory path.
        output_dir: Output directory path.
    
    Returns:
        Processing statistics.
    """
    from generated.servers.filesystem import list_directory, read_file, write_file
    
    entries = await list_directory({"path": input_dir})
    processed = 0
    
    for entry in entries.get("entries", []):
        content = await read_file({"path": f"{input_dir}/{entry}"})
        # Process content...
        await write_file({"path": f"{output_dir}/{entry}", "content": content.upper()})
        processed += 1
    
    return {"processed": processed}
```

#### Using Skills in Executed Code

Skills are imported and called like any Python module:

```python
# In executed code
from skills.batch_process import batch_process

result = await batch_process("/data/input", "/data/output")
print(f"Processed {result['processed']} files")
```

#### Managing Skills with SimpleSkillsManager

For programmatic skill management, use the `SimpleSkillsManager`:

```python
from agent_skills import SimpleSkillsManager, SimpleSkill

# Create a skills manager
manager = SimpleSkillsManager("./skills")

# Save a skill
skill = SimpleSkill(
    name="batch_process",
    description="Process files in a directory",
    code='''
async def batch_process(input_dir, output_dir):
    entries = await list_directory({"path": input_dir})
    for entry in entries.get("entries", []):
        content = await read_file({"path": f"{input_dir}/{entry}"})
        await write_file({"path": f"{output_dir}/{entry}", "content": content.upper()})
''',
    tags=["file", "batch"],
)
manager.save_skill(skill)

# Load and use a skill
loaded = manager.load_skill("batch_process")
print(loaded.code)
```

## Examples

See the runnable examples in [examples/](examples/).

### Simple Examples

```bash
python examples/simple/codemode_example.py
python examples/simple/codemode_patterns_example.py
```

### Agent CLI

Interactive CLI agent with Agent Codemode support:

```bash
# Standard mode
python examples/agent/agent_cli.py

# Codemode variant (code-first tool composition)
python examples/agent/agent_cli.py --codemode
```

## Pydantic AI Integration

Use the `CodemodeToolset` for direct integration with Pydantic AI agents:

```python
from pydantic_ai import Agent
from agent_codemode import CodemodeToolset, ToolRegistry, MCPServerConfig

# Set up registry
registry = ToolRegistry()
registry.add_server(MCPServerConfig(
    name="filesystem",
    command="npx",
    args=["-y", "@anthropic/mcp-server-filesystem", "/tmp"]
))
await registry.discover_all()

# Create toolset
toolset = CodemodeToolset(registry=registry)

# Use with Pydantic AI agent
agent = Agent(
    model='anthropic:claude-sonnet-4-5',
    toolsets=[toolset],
)
```

### MCP Server Mode

Expose the programmatic tools as an MCP server for any MCP-compatible client:

```python
from agent_codemode import codemode_server, configure_server
from agent_codemode import ToolRegistry, MCPServerConfig, CodeModeConfig

# Create and configure registry with MCP servers to compose
registry = ToolRegistry()
registry.add_server(MCPServerConfig(
    name="filesystem",
    command="npx",
    args=["-y", "@anthropic/mcp-server-filesystem", "/tmp"]
))

# Configure with custom settings
config = CodeModeConfig(
    workspace_path="./workspace",
    skills_path="./skills",
    generated_path="./generated",
)

configure_server(config=config, registry=registry)
codemode_server.run()
```

Or start with command line:

```bash
python -m agent_codemode.server --workspace ./workspace
```

### Connect to an MCP Client

Create a launcher script to configure which MCP servers to compose:

```python
#!/usr/bin/env python3
import sys
from pathlib import Path

from agent_codemode import ToolRegistry, MCPServerConfig, CodeModeConfig
from agent_codemode.server import configure, run

# Create registry and add MCP servers to compose
registry = ToolRegistry()
registry.add_server(MCPServerConfig(
    name="my_server",
    command=sys.executable,
    args=["/path/to/my_mcp_server.py"]
))

# Configure paths
config = CodeModeConfig(
    workspace_path="./workspace",
    generated_path="./generated",
    skills_path="./skills",
)

configure(config=config, registry=registry)
run()
```

Then configure your MCP client to run the launcher script.

**Tools exposed by the MCP server:**

| Tool | Description |
|------|-------------|
| `search_tools` | Progressive tool discovery |
| `list_servers` | List connected MCP servers |
| `list_tool_names` | Fast listing of tool names |
| `get_tool_details` | Get full tool schema |
| `execute_code` | Run code that composes tools |
| `call_tool` | Direct tool invocation |
| `save_skill` / `run_skill` | Skill management |
| `list_skills` / `delete_skill` | Skill management |
| `add_mcp_server` | Dynamically add servers |

## Recommended System Prompt

When building an agent that uses Codemode, use this system prompt pattern:

```text
You are an AI assistant with access to MCP tools via Code Mode.

## Available Meta-Tools
- **list_tool_names** - Fast listing of tool names
- **search_tools** - AI-powered tool discovery (returns full definitions)
- **get_tool_details** - Get schema for a specific tool
- **execute_code** - Execute Python code in a sandboxed environment

## Execution Model
ALL tool execution must go through execute_code. Write Python code that imports 
and uses the generated tool bindings:

```python
from generated.servers.filesystem import read_file
content = await read_file({"path": "/data/config.json"})
print(content)
```
```

## Workflow

1. Discover tools using search_tools or list_tool_names
2. Write Python code that imports tools from generated.servers.<name>
3. Execute using execute_code


See the [Getting Started guide](https://agent-codemode.datalayer.tech/getting-started) for a complete system prompt example.

## Key Patterns

### Tool Discovery

Instead of loading all tool definitions upfront (which can overwhelm context), use the Tool Search Tool pattern for progressive discovery based on the task at hand.

### Tool Composition

Compose tools through code instead of reading all data into LLM context. This is faster, more reliable (no text reproduction errors), and more efficient.

### Control Flow

Code allows models to implement complex control flow: loops, conditionals, waiting patterns, and parallel execution without burning through context with repeated tool calls.

### State Persistence

When running in a sandbox, state can persist between `execute_code` calls within the same session. Variables, functions, and imported modules remain available for subsequent code executions. Skills can also be saved to disk and loaded later for reuse across sessions.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Agent Codemode                              │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌────────────────┐  ┌───────────────────────┐ │
│  │  Tool Registry  │  │ Code Executor  │  │   CodemodeToolset     │ │
│  │  - Discovery    │  │  - Sandbox     │  │   (Pydantic AI)       │ │
│  │  - Search       │  │  - Bindings    │  │   - search_tools      │ │
│  │  - Cache        │  │  - Execute     │  │   - execute_code      │ │
│  └─────────────────┘  └────────────────┘  └───────────────────────┘ │
│                              │                                      │
│              ┌───────────────┴───────────────┐                      │
│              │        Generated Bindings     │                      │
│              │  generated/servers/<name>.py  │                      │
│              └───────────────────────────────┘                      │
├────────────────────────────────┬────────────────────────────────────┤
│          MCP Servers           │      Agent Skills (agent_skills)   │
│  (filesystem, bash, web, etc.  │    (SimpleSkillsManager,           │
│   connected via MCP protocol)  │     SkillDirectory, skills/*.py)   │
└────────────────────────────────┴────────────────────────────────────┘
```

## References

- [Introducing Code Mode](https://blog.cloudflare.com/introducing-code-mode) - Cloudflare
- [Code Execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp) - Anthropic
- [Programmatic Tool Calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling) - Anthropic
- [Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use) - Anthropic
- [Programmatic MCP Prototype](https://github.com/domdomegg/programmatic-mcp-prototype)

## License

BSD 3-Clause License
