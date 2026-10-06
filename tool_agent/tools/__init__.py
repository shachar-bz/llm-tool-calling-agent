"""The agent's toolbox.

To add a tool: write a module that builds a `Tool` (schema + handler), then
list it in `build_toolbox`. Nothing else in the agent needs to change.
"""

from ..llm import LLM
from ..task import Workspace
from .base import Tool, ToolRegistry
from .build_sql_query import build_sql_query_tool
from .calculator import CALCULATOR
from .execute_sql_query import execute_sql_query_tool
from .extract_from_image import extract_from_image_tool
from .web_search import WEB_SEARCH
from .write_file import write_file_tool

__all__ = ["Tool", "ToolRegistry", "build_toolbox"]


def build_toolbox(llm: LLM, workspace: Workspace) -> ToolRegistry:
    return ToolRegistry([
        CALCULATOR,
        extract_from_image_tool(llm, workspace),
        build_sql_query_tool(llm),
        execute_sql_query_tool(workspace),
        WEB_SEARCH,
        write_file_tool(workspace),
    ])
