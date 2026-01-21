#!/usr/bin/env python3
"""MCP Server - File Tokens Demo (STDIO).

Provides tools that read/write files and generate random text so the
agent can perform token-heavy workflows. Designed for use with the
Codemode agent CLI example.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("example-mcp-server")

_WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    "xray", "yankee", "zulu", "apricot", "banana", "cherry", "date", "elderberry",
    "fig", "grape", "honeydew", "kiwi", "lemon", "mango", "nectarine", "orange",
    "papaya", "quince", "raspberry", "strawberry", "tangerine", "ugli", "vanilla",
    "watermelon", "xigua", "yam", "zucchini", "azure", "binary", "cache",
    "docker", "elastic", "feature", "gateway", "hash", "index", "json", "kafka",
    "lambda", "micro", "node", "object", "python", "query", "router", "schema",
    "token", "vector", "worker", "yaml",
]


def _normalize_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


@mcp.tool()
def generate_random_text(word_count: int = 1000, seed: Optional[int] = None) -> dict:
    """Generate pseudo-random text.

    Args:
        word_count: Number of words to generate.
        seed: Optional random seed for reproducibility.

    Returns:
        Dictionary with generated text and word count.
    """
    rng = random.Random(seed)
    words = [rng.choice(_WORDS) for _ in range(max(word_count, 0))]
    text = " ".join(words)
    return {"text": text, "word_count": len(words)}


@mcp.tool()
def write_text_file(path: str, content: str) -> dict:
    """Write text content to a file.

    Args:
        path: File path to write.
        content: Text content.

    Returns:
        Metadata about the write.
    """
    target = _normalize_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {
        "path": str(target),
        "bytes": len(content.encode("utf-8")),
        "words": len(content.split()),
    }


@mcp.tool()
def read_text_file(
    path: str,
    include_content: bool = True,
    max_chars: Optional[int] = None,
) -> dict:
    """Read text content from a file.

    Args:
        path: File path to read.
        include_content: Whether to include content in the response.
        max_chars: Optional limit for content length.

    Returns:
        File content and metadata.
    """
    target = _normalize_path(path)
    content = target.read_text(encoding="utf-8")
    if max_chars is not None:
        content = content[: max_chars]
    response = {
        "path": str(target),
        "bytes": len(content.encode("utf-8")),
        "words": len(content.split()),
    }
    if include_content:
        response["content"] = content
    return response


@mcp.tool()
def read_text_file_many(
    path: str,
    times: int = 10,
    include_content: bool = False,
    max_chars: Optional[int] = None,
) -> dict:
    """Read a file multiple times.

    Args:
        path: File path to read.
        times: Number of reads.
        include_content: Whether to include content in the response.
        max_chars: Optional limit for content length per read.

    Returns:
        Aggregate statistics and optional content from the final read.
    """
    times = max(times, 0)
    last = {}
    for _ in range(times):
        last = read_text_file(path, include_content=include_content, max_chars=max_chars)
    return {
        "path": last.get("path", path),
        "reads": times,
        "last": last,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
