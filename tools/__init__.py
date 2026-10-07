"""The agent's tools: each module builds a `Tool` (JSON schema + handler).

This package knows nothing about how a run is driven; `agent` imports it,
never the reverse (the `agent` imports here are for type hints only).
To add a tool, write a module here and list it in `agent.toolbox.build_toolbox`.
"""

from .base import Tool, ToolRegistry
from .build_sql_query import build_sql_query_tool
from .calculator import CALCULATOR
from .execute_sql_query import execute_sql_query_tool
from .extract_from_image import extract_from_image_tool
from .web_search import WEB_SEARCH
from .write_file import write_file_tool

__all__ = [
    "CALCULATOR",
    "Tool",
    "ToolRegistry",
    "WEB_SEARCH",
    "build_sql_query_tool",
    "execute_sql_query_tool",
    "extract_from_image_tool",
    "write_file_tool",
]
