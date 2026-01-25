#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skills Agent CLI with Agent Codemode (STDIO).

This agent connects to a local MCP stdio server and provides an
interactive CLI with skills integration. Codemode is enabled by
default for code-first tool composition with skill discovery.

Skills are loaded from the local `skills/` folder which contains
skill definitions like the PDF processing skill.
"""

from __future__ import annotations

import asyncio
import io
import sys
import logging
from pathlib import Path
from typing import Optional
import inspect

try:
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerStdio
    HAS_PYDANTIC_AI = True
except ImportError:
    HAS_PYDANTIC_AI = False

try:
    from rich.console import Console
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

logger = logging.getLogger(__name__)


def _resolve_mcp_server_path() -> str:
    return str(Path(__file__).with_name("example_mcp_server.py").resolve())


def _build_prompt_examples(codemode: bool) -> str:
    base = [
        "Create random content of 1000 words, write it to file.txt, and read it ten times.",
        "Generate 500 words of random text, write to ./data/sample.txt, then read it once.",
        "Write random content to ./data/log.txt and read it back with max_chars=200.",
    ]
    if codemode:
        base.append(
            "(Codemode) Use execute_code to generate text, write it once, and read it ten times "
            "without returning the full content each time."
        )
    else:
        base.append(
            "(Standard) Use the MCP tools directly for each step."
        )
    return "\n".join(f"  - {item}" for item in base)


def _build_system_prompt(
    codemode: bool,
    tool_hints: Optional[list[tuple[str, str]]] = None,
) -> str:
    if not codemode:
        lines = [
            "You are a helpful AI assistant with access to MCP tools.",
            "Use MCP tools when they are the best way to complete the task.",
            "Avoid tool discovery unless the user asks about tools or a tool is unknown.",
            "When unsure, ask for clarification.",
        ]
        if tool_hints:
            lines.append("Known tools:")
            for name, description in tool_hints:
                if description:
                    lines.append(f"- {name}: {description}")
                else:
                    lines.append(f"- {name}")
        return " ".join(lines)

    # For codemode: emphasize using the 4 core codemode tools + skills
    lines = [
        "You are a helpful AI assistant with Agent Codemode and Skills support.",
        "",
        "## IMPORTANT: Be Honest About Your Capabilities",
        "NEVER claim to have tools or capabilities you haven't verified.",
        "When greeting users or describing yourself, say you can DISCOVER what tools and skills are available.",
        "Use list_skills or search_tools FIRST to see what's actually available before claiming any capabilities.",
        "",
        "## Skills",
        "You have access to skills - pre-built knowledge and scripts for complex tasks.",
        "Use these skill-specific tools:",
        "",
        "- **list_skills** - List all available skills with descriptions",
        "- **load_skill** - Load full content and instructions for a skill",
        "- **read_skill_resource** - Read a resource file from a skill",
        "- **run_skill_script** - Execute a script from a skill with arguments",
        "",
        "Example workflow for skills:",
        "1. Use list_skills to see what's available",
        "2. Use load_skill(skill_name) to get full instructions",
        "3. Use run_skill_script to execute skill scripts",
        "",
        "## Core Codemode Tools",
        "Use these 4 tools to accomplish any task:",
        "",
        "1. **search_tools** - Progressive tool discovery by natural language query",
        "   Use this to find relevant MCP tools before executing tasks.",
        "",
        "2. **get_tool_details** - Get full tool schema and documentation",
        "   Use this to understand tool parameters before calling them.",
        "",
        "3. **execute_code** - Run Python code that composes multiple tools",
        "   Use this for complex multi-step operations. Code runs in a PERSISTENT sandbox.",
        "   Variables, functions, and state PERSIST between execute_code calls.",
        "   Import tools using: `from generated.servers.<server_name> import <function_name>`",
        "   NEVER use `import *` - always use explicit named imports.",
        "",
        "4. **call_tool** - Direct single-tool invocation",
        "   Use this for simple, single-tool operations.",
        "",
        "## Recommended Workflow",
        "1. **Discover**: Use list_skills to see skills, search_tools to find MCP tools",
        "2. **Understand**: Use load_skill or get_tool_details to check instructions/parameters",
        "3. **Execute**: Use run_skill_script for skills, call_tool for simple MCP ops, or execute_code for complex workflows",
        "",
    ]
    return "\n".join(lines)


def _usage_to_dict(usage: object) -> dict[str, object]:
    if usage is None:
        return {}

    if isinstance(usage, dict):
        return usage

    data = None
    if hasattr(usage, "model_dump"):
        try:
            data = usage.model_dump()
        except Exception:
            data = None
    if data is None and hasattr(usage, "dict"):
        try:
            data = usage.dict()
        except Exception:
            data = None
    if data is None and hasattr(usage, "__dict__"):
        data = usage.__dict__

    if isinstance(data, dict):
        return data

    extracted: dict[str, object] = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "requests",
        "cached_tokens",
        "billable_tokens",
    ):
        if hasattr(usage, key):
            try:
                extracted[key] = getattr(usage, key)
            except Exception:
                pass

    return extracted


def _format_usage(usage: object, keys: Optional[list[str]] = None) -> str:
    data = _usage_to_dict(usage)
    if not data:
        return "N/A"

    parts = []
    preferred = [k for k in (keys or [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "requests",
        "cached_tokens",
        "billable_tokens",
        "cache_write_tokens",
        "cache_read_tokens",
        "input_audio_tokens",
        "cache_audio_read_tokens",
        "output_audio_tokens",
        "tool_calls",
        "details",
    ]) if k != "details"]
    for key in preferred:
        if key in data:
            parts.append(f"{key}={data[key]}")
    if parts:
        return ", ".join(parts)
    return str(data)

    return str(usage)


def _usage_to_table(
    prompt_usage: object,
    session_usage: object,
    keys: Optional[list[str]] = None,
) -> "Table":
    prompt_data = _usage_to_dict(prompt_usage)
    session_data = _usage_to_dict(session_usage)
    table = Table(title="Token usage")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Prompt", style="cyan")
    table.add_column("Session", style="cyan")

    preferred = keys or [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "requests",
        "cached_tokens",
        "billable_tokens",
    ]

    if not prompt_data and not session_data:
        table.add_row("usage", "N/A", "N/A")
        return table

    def _format_value(value: object) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            if value.is_integer():
                return str(int(value))
            return str(value)
        if isinstance(value, (int, bool)):
            return str(value)
        return str(value)

    for key in preferred:
        if key in prompt_data or key in session_data:
            table.add_row(
                key,
                _format_value(prompt_data.get(key, "-")),
                _format_value(session_data.get(key, "-")),
            )


    if table.row_count == 0:
        table.add_row("usage", str(prompt_data or "-"), str(session_data or "-"))

    return table


async def _list_available_tools(
    codemode: bool,
    codemode_toolset: object | None,
    mcp_server_path: str,
) -> list[tuple[str, str]]:
    tools: list[tuple[str, str]] = []

    if codemode and codemode_toolset is not None:
        registry = getattr(codemode_toolset, "registry", None)
        if registry is not None:
            if not registry.list_tools():
                await registry.discover_all()
            for tool in registry.list_tools(include_deferred=True):
                tools.append((tool.name, tool.description or ""))
            return tools

    # Use raw subprocess to avoid anyio cancel-scope issues with ClientSession
    try:
        import json as _json
        import os as _os

        env = {**_os.environ}
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            mcp_server_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "agent_cli", "version": "1.0.0"},
            },
        }
        if proc.stdin:
            proc.stdin.write((_json.dumps(request) + "\n").encode())
            await proc.stdin.drain()
        if proc.stdout:
            await proc.stdout.readline()  # discard initialize response
        request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        if proc.stdin:
            proc.stdin.write((_json.dumps(request) + "\n").encode())
            await proc.stdin.drain()
        if proc.stdout:
            line = await proc.stdout.readline()
            result = _json.loads(line.decode())
            for tool in result.get("result", {}).get("tools", []):
                tools.append((tool.get("name", ""), tool.get("description", "")))
        proc.terminate()
        await proc.wait()
    except Exception:
        pass

    return tools


def create_agent(model: str, codemode: bool) -> tuple[Agent, object | None, object | None]:
    if not HAS_PYDANTIC_AI:
        print("❌ Error: pydantic-ai not installed")
        print("   Install with: pip install 'pydantic-ai[mcp]'\n")
        sys.exit(1)

    mcp_server_path = _resolve_mcp_server_path()
    example_dir = Path(__file__).resolve().parent
    skills_dir = example_dir / "skills"
    skills_toolset = None

    if codemode:
        from agent_codemode import CodemodeToolset, ToolRegistry, MCPServerConfig, CodeModeConfig
        from agent_skills import DatalayerSkillsToolset

        registry = ToolRegistry()
        # Server name becomes the tool prefix (e.g., example_mcp__read_text_file)
        # and matches the generated bindings path: generated.servers.example_mcp
        registry.add_server(
            MCPServerConfig(
                name="example_mcp",
                command=sys.executable,
                args=[mcp_server_path],
            )
        )

        repo_root = Path(__file__).resolve().parents[2]
        config = CodeModeConfig(
            workspace_path=str((repo_root / "workspace").resolve()),
            generated_path=str((repo_root / "generated").resolve()),
            skills_path=str(skills_dir.resolve()),  # Local skills folder
            allow_direct_tool_calls=False,
        )

        codemode_toolset = CodemodeToolset(
            registry=registry,
            config=config,
            allow_discovery_tools=True,  # Enable discovery tools (search_tools, get_tool_details, list_tool_names, list_servers)
        )
        
        # Add skills toolset for skill discovery and execution
        skills_toolset = DatalayerSkillsToolset(
            directories=[str(skills_dir.resolve())],
        )
        
        toolsets = [codemode_toolset, skills_toolset]
        toolset = codemode_toolset
    else:
        mcp_server = MCPServerStdio(
            sys.executable,
            args=[mcp_server_path],
            timeout=300.0,
        )
        toolsets = [mcp_server]
        toolset = None

    # Skip upfront tool discovery to avoid anyio cancel-scope issues.
    # Tool hints are omitted; toolset will discover tools lazily on first use.
    tool_hints: list[tuple[str, str]] = []

    agent_kwargs = dict(
        model=model,
        toolsets=toolsets,
        system_prompt=_build_system_prompt(codemode, tool_hints),
    )
    try:
        signature = inspect.signature(Agent)
        if "retries" in signature.parameters:
            agent_kwargs["retries"] = 3
        elif "max_retries" in signature.parameters:
            agent_kwargs["max_retries"] = 3
        elif "model_settings" in signature.parameters:
            agent_kwargs["model_settings"] = {"max_retries": 3}
    except Exception:
        agent_kwargs["model_settings"] = {"max_retries": 3}

    agent = Agent(**agent_kwargs)

    return agent, toolset, skills_toolset


def main() -> None:
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    # Suppress verbose MCP server logs
    logging.getLogger("mcp.server").setLevel(logging.WARNING)
    
    import argparse
    parser = argparse.ArgumentParser(description="Skills Agent CLI with Agent Codemode")
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic:claude-sonnet-4-0",
        help="Model to use (default: anthropic:claude-sonnet-4-0)"
    )
    parser.add_argument(
        "--standard",
        action="store_true",
        help="Disable codemode (use standard MCP mode)"
    )
    args = parser.parse_args()
    
    model = args.model
    codemode = not args.standard  # Codemode is the default

    print("\n" + "=" * 72)
    if codemode:
        print("🤖 𝄃𝄂𝄂𝄀𝄁𝄃𝄂𝄂𝄃 Agent Codemode Agent CLI")
    else:
        print("🤖 MCP Agent CLI")
    print("=" * 72)
    print(f"\nMode: {'Codemode' if codemode else 'Standard MCP'}")
    print(f"Model: {model}")

    print("\n📋 Example prompts:")
    print(_build_prompt_examples(codemode))

    # print("\nCommands:")
    # print("  /exit      - Exit the CLI")
    # print("  /markdown  - Toggle markdown rendering")
    # print("  /multiline - Enter multiline mode")
    # print("  /cp        - Copy last response to clipboard")
    # print("\n" + "=" * 72 + "\n")

    agent, codemode_toolset, skills_toolset = create_agent(model=model, codemode=codemode)

    async def _run_cli() -> None:
        prompt = "𝄃𝄂𝄂𝄀𝄁𝄃𝄂𝄂𝄃 agent-codemode-agent ➤ " if codemode else "mcp-agent ➤ "
        multiline = False
        message_history = []  # Store conversation history
        session_usage: dict[str, float] = {
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "total_tokens": 0.0,
            "requests": 0.0,
            "cached_tokens": 0.0,
            "billable_tokens": 0.0,
            "codemode_tool_calls": 0.0,
            "mcp_tool_calls": 0.0,
            "skills_tool_calls": 0.0,
        }
        previous_counts: dict[str, int] = {
            "codemode_tool_calls": 0,
            "mcp_tool_calls": 0,
            "skills_tool_calls": 0,
        }

        if codemode_toolset:
            if codemode:
                logger.debug("Initializing codemode environment...")
            try:
                if hasattr(codemode_toolset, "start"):
                    await codemode_toolset.start()
            except Exception as e:
                logger.debug("Failed to start codemode", exc_info=e)
                return

        async with agent:
            mcp_server_path = _resolve_mcp_server_path()
            while True:
                if multiline:
                    print("Enter multiline input. End with /end on its own line.")
                    lines: list[str] = []
                    while True:
                        line = await asyncio.to_thread(input, "… ")
                        if line.strip() == "/end":
                            break
                        lines.append(line)
                    user_input = "\n".join(lines).strip()
                    multiline = False
                else:
                    user_input = await asyncio.to_thread(input, prompt)
                    user_input = user_input.strip()

                if not user_input:
                    continue

                if user_input in {"/exit", "/quit"}:
                    break
                if user_input == "/markdown":
                    print("Markdown rendering is not enabled in this minimal CLI.")
                    continue
                if user_input == "/multiline":
                    multiline = True
                    continue
                if user_input == "/cp":
                    print("Clipboard copy is not enabled in this minimal CLI.")
                    continue

                run_result = None
                run_usage = None
                iteration_count = 0
                async with agent.iter(user_input, message_history=message_history) as run:
                    async for node in run:
                        iteration_count += 1
                        node_type = type(node).__name__
                        # Print all node types for debugging
                        logger.debug("  [iter %s] %s", iteration_count, node_type)
                        
                        if node_type == 'CallToolsNode':
                            mr = getattr(node, 'model_response', None)
                            if mr and hasattr(mr, 'parts'):
                                for p in mr.parts:
                                    if hasattr(p, 'tool_name'):
                                        args = getattr(p, 'args', {})
                                        if isinstance(args, dict) and 'code' in args:
                                            code_preview = args['code'][:100].replace('\n', '\\n')
                                            logger.debug(
                                                "    -> %s(code=%s...)",
                                                p.tool_name,
                                                code_preview,
                                            )
                                        else:
                                            logger.debug("    -> %s(%s)", p.tool_name, args)
                        elif node_type == 'HandleResponseNode':
                            # Tool results might be here
                            data = getattr(node, 'data', None)
                            if data:
                                logger.debug("    -> data: %s", str(data)[:200])
                    run_result = run.result
                    run_usage = run.usage()
                    # Update message history with the conversation
                    message_history = run.all_messages()

                if run_result is None:
                    print("No result returned.")
                    continue

                reply = getattr(run_result, "output", None)
                if reply is None:
                    reply = getattr(run_result, "data", None)
                if reply is None:
                    reply = str(run_result)

                usage_data = _usage_to_dict(run_usage)
                if usage_data:
                    for key in session_usage:
                        if key in usage_data and isinstance(usage_data[key], (int, float)):
                            session_usage[key] += float(usage_data[key])
                        elif key in usage_data:
                            try:
                                session_usage[key] += float(usage_data[key])
                            except Exception:
                                pass

                prompt_usage_payload = dict(usage_data)
                if codemode and codemode_toolset is not None:
                    counts = {}
                    if hasattr(codemode_toolset, "get_call_counts"):
                        counts = codemode_toolset.get_call_counts()  # type: ignore[assignment]
                    if counts:
                        prompt_usage_payload["codemode_tool_calls"] = (
                            counts.get("codemode_tool_calls", 0) - previous_counts["codemode_tool_calls"]
                        )
                        prompt_usage_payload["mcp_tool_calls"] = (
                            counts.get("mcp_tool_calls", 0) - previous_counts["mcp_tool_calls"]
                        )
                        previous_counts["codemode_tool_calls"] = counts.get("codemode_tool_calls", 0)
                        previous_counts["mcp_tool_calls"] = counts.get("mcp_tool_calls", 0)
                        session_usage["codemode_tool_calls"] += float(prompt_usage_payload["codemode_tool_calls"])
                        session_usage["mcp_tool_calls"] += float(prompt_usage_payload["mcp_tool_calls"])

                # Track skills tool calls
                if skills_toolset is not None:
                    skills_counts = {}
                    if hasattr(skills_toolset, "get_call_counts"):
                        skills_counts = skills_toolset.get_call_counts()  # type: ignore[assignment]
                    if skills_counts:
                        prompt_usage_payload["skills_tool_calls"] = (
                            skills_counts.get("skills_tool_calls", 0) - previous_counts["skills_tool_calls"]
                        )
                        previous_counts["skills_tool_calls"] = skills_counts.get("skills_tool_calls", 0)
                        session_usage["skills_tool_calls"] += float(prompt_usage_payload["skills_tool_calls"])
                else:
                    prompt_usage_payload["skills_tool_calls"] = "N/A"

                if not codemode:
                    if "tool_calls" in prompt_usage_payload and "mcp_tool_calls" not in prompt_usage_payload:
                        prompt_usage_payload["mcp_tool_calls"] = prompt_usage_payload["tool_calls"]
                    if "tool_calls" in prompt_usage_payload:
                        prompt_usage_payload.pop("tool_calls", None)
                    prompt_usage_payload.setdefault("mcp_tool_calls", 0)
                    prompt_usage_payload["codemode_tool_calls"] = "N/A"
                    if "tool_calls" in usage_data and isinstance(usage_data["tool_calls"], (int, float)):
                        session_usage["mcp_tool_calls"] += float(usage_data["tool_calls"])
                    elif "tool_calls" in usage_data:
                        try:
                            session_usage["mcp_tool_calls"] += float(usage_data["tool_calls"])
                        except Exception:
                            pass

                if "tool_calls" in prompt_usage_payload:
                    prompt_usage_payload.pop("tool_calls", None)

                prompt_usage_keys = [
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "requests",
                    "cached_tokens",
                    "billable_tokens",
                    "cache_write_tokens",
                    "cache_read_tokens",
                    "input_audio_tokens",
                    "cache_audio_read_tokens",
                    "output_audio_tokens",
                    "mcp_tool_calls",
                    "codemode_tool_calls",
                    "skills_tool_calls",
                ]

                print(reply)

                if HAS_RICH:
                    console = Console()
                    console.print()
                    session_usage_payload = dict(session_usage)
                    if not codemode:
                        session_usage_payload["codemode_tool_calls"] = "N/A"
                    if skills_toolset is None:
                        session_usage_payload["skills_tool_calls"] = "N/A"
                    console.print(
                        _usage_to_table(prompt_usage_payload, session_usage_payload, prompt_usage_keys)
                    )
                    console.print()
                else:
                    print(f"\nToken usage (prompt): {_format_usage(run_usage)}")
                    print(
                        f"Token usage (session): {_format_usage(session_usage, prompt_usage_keys)}\n"
                    )

    try:
        asyncio.run(_run_cli())
    except KeyboardInterrupt:
        pass
    finally:
        if codemode_toolset and hasattr(codemode_toolset, "cleanup"):
            try:
                asyncio.run(codemode_toolset.cleanup())
            except Exception as e:
                logger.debug("Error during cleanup", exc_info=e)


if __name__ == "__main__":
    main()
