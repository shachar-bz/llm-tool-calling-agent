"""Entry points: `run_task` assembles and runs one task; `main` is the command line around it."""

import argparse
import os
import sys

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from .agent import Agent
from .config import ConfigError, Limits, ProviderSettings
from .llm import LLM, OpenAITransport, Transport
from .run_log import RunLog
from .task import TaskError, Workspace, load_task
from .tools import build_toolbox


def run_task(
    manifest_path: str | os.PathLike,
    transport: Transport,
    limits: Limits = Limits(),
) -> str | None:
    """Solve the task described by an input.json manifest.

    Returns the model's final answer, or None if a call cap stopped the run.
    Output files and the <query>.log run log are written next to the manifest.
    Raises TaskError if the manifest or query file can't be loaded.
    """
    task = load_task(manifest_path)
    workspace = Workspace(task)
    llm = LLM(transport, limits.max_llm_calls)

    with RunLog.open(task.log_path) as log:
        agent = Agent(llm, build_toolbox(llm, workspace), log, limits.max_tool_calls)
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
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=Limits.max_llm_calls,
        metavar="N",
        help="cap on model requests, including those made inside tools (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=Limits.max_tool_calls,
        metavar="N",
        help="cap on tool invocations (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    load_dotenv(find_dotenv(usecwd=True))
    try:
        settings = ProviderSettings.from_env()
        transport = OpenAITransport(
            OpenAI(api_key=settings.api_key, base_url=settings.endpoint),
            settings.model,
        )
        answer = run_task(args.manifest, transport, Limits(args.max_llm_calls, args.max_tool_calls))
    except (ConfigError, TaskError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0 if answer is not None else 1
