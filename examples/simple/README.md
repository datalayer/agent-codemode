# Simple Agent CLI

Interactive CLI agent that talks to a local MCP stdio server with file read/write tools.

## Run

Standard MCP mode:

```bash
python agent_cli.py
```

Codemode variant (code-first tool composition):

```bash
python agent_cli.py --codemode
```

Make targets:

```bash
make agent           # Standard MCP mode
make agent-codemode  # Codemode
```

## MCP Server

The agent CLI spawns this stdio MCP server automatically. It provides:

- `generate_random_text(word_count, seed)`
- `write_text_file(path, content)`
- `read_text_file(path, include_content, max_chars)`
- `read_text_file_many(path, times, include_content, max_chars)`

## Generated content

Generated code is written to the repo root (`generated/`) after tool discovery runs.

If you don't see it, run a prompt that triggers tool discovery (e.g., `/list_tool_names` or `/search_tools`).
