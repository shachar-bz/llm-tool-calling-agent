"""Writes the task's output files (paths are relative to the task root)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .base import Tool

if TYPE_CHECKING:
    from agent.task import Workspace


def write_file(path: Path, content: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return len(content.encode("utf-8"))


def write_file_tool(workspace: Workspace) -> Tool:
    def handler(file_content: str, file_name: str) -> str:
        size = write_file(workspace.output_path(file_name), file_content)
        return f"Wrote {size} bytes to {file_name}"

    return Tool(
        name="write_file",
        description=(
            "Write a string to a file, creating parent directories if needed. "
            "Use this to produce the output file or files requested by the query."
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_content": {
                    "type": "string",
                    "description": "The exact bytes/text to write.",
                },
                "file_name": {
                    "type": "string",
                    "description": "Filesystem path of the file to write.",
                },
            },
            "required": ["file_content", "file_name"],
        },
        handler=handler,
    )
