"""Command-line entry point: wires settings, the LLM client, the tools and the agent together."""

import argparse
import os
import sys

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from .agent import Agent
from .budget import Budget
from .config import ConfigError, Settings
from .llm import LLM
from .run_log import RunLog
from .task import Workspace, load_task
from .tools import build_toolbox


def run_task(manifest_path: str | os.PathLike, settings: Settings) -> str | None:
    """Solve the task described by an input.json manifest. Returns the final answer, or None if capped."""
    task = load_task(manifest_path)
    workspace = Workspace(task.root, task.resources)
    budget = Budget(settings.max_llm_calls, settings.max_tool_calls)
    llm = LLM(OpenAI(api_key=settings.api_key, base_url=settings.endpoint), settings.model, budget)

    with RunLog.open(workspace.output_path(task.log_name)) as log:
        agent = Agent(llm, build_toolbox(llm, workspace), budget, log)
        return agent.run(task)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tool-agent",
        description="Answer a query with an LLM that chains tools (vision, SQL, calculator, web search, files).",
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        default="input.json",
        help="path to the task's input.json (default: ./input.json)",
    )
    args = parser.parse_args(argv)

    load_dotenv(find_dotenv(usecwd=True))
    try:
        settings = Settings.from_env()
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    answer = run_task(args.manifest, settings)
    return 0 if answer is not None else 1
