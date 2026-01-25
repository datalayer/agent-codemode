# Skills Agent CLI

Interactive CLI agent with Agent Codemode and skills integration. This example demonstrates how to use codemode with the agent-skills framework for document processing capabilities.

## Prerequisites

Install PDF processing dependencies:

```bash
pip install pypdf pdf2image reportlab
```

## Run

Codemode is enabled by default:

```bash
python agent_cli.py
```

To run in standard MCP mode (without codemode):

```bash
python agent_cli.py --standard
```

Make targets:

```bash
make agent           # Codemode (default)
make agent-standard  # Standard MCP mode
```

## Skills

The `skills/` folder contains skill definitions that the agent can discover and use:

| Skill | Description |
|-------|-------------|
| `pdf` | PDF manipulation toolkit - extract text/tables, merge/split documents, fill forms |

### Example prompts

```
> What skills do you have available?
> Extract text from document.pdf
> Fill out the form in application.pdf with the following: name="John Doe", email="john@example.com"
> Merge report1.pdf and report2.pdf into combined.pdf
```

## MCP Server

The agent CLI spawns a local MCP stdio server automatically. It provides:

- `generate_random_text(word_count, seed)`
- `write_text_file(path, content)`
- `read_text_file(path, include_content, max_chars)`
- `read_text_file_many(path, times, include_content, max_chars)`

## Generated content

Generated code is written to the repo root (`generated/`) after tool discovery runs.

If you don't see it, run a prompt that triggers tool discovery (e.g., `/list_tool_names` or `/search_tools`).

## How Skills Work

1. **Discovery**: The agent discovers skills from the `skills/` folder
2. **Understanding**: Skills contain SKILL.md with instructions and scripts
3. **Execution**: The agent uses codemode to execute skill scripts in a sandbox

For more on skills, see the [agent-skills](https://github.com/datalayer/agent-skills) repository.
