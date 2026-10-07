"""Assembles the agent's toolbox: binds the LLM and the task workspace to the tools."""

from tools import (
    CALCULATOR,
    WEB_SEARCH,
    ToolRegistry,
    build_sql_query_tool,
    execute_sql_query_tool,
    extract_from_image_tool,
    write_file_tool,
)

from .llm import LLM
from .task import Workspace


def build_toolbox(llm: LLM, workspace: Workspace) -> ToolRegistry:
    return ToolRegistry([
        CALCULATOR,
        extract_from_image_tool(llm, workspace),
        build_sql_query_tool(llm),
        execute_sql_query_tool(workspace),
        WEB_SEARCH,
        write_file_tool(workspace),
    ])
