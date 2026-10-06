"""The Tool abstraction: an OpenAI function schema bundled with the code that runs it."""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]):
        self._tools = {tool.name: tool for tool in tools}

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema for tool in self._tools.values()]


_FENCED_BLOCK = re.compile(r"```[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)


def strip_code_fences(text: str) -> str:
    """Return the body of the first ``` fenced block in `text`, or `text` itself if there is none."""
    match = _FENCED_BLOCK.search(text)
    return (match.group(1) if match else text).strip()
