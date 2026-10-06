"""The agent loop: ask the model for its next tool call, run it, feed the result back, repeat."""

import json
from typing import Any

from openai import BadRequestError
from openai.types.chat import ChatCompletionMessageToolCall

from .budget import Budget
from .llm import LLM
from .prompts import LLM_ERROR_RECOVERY_MESSAGE, system_prompt, user_message
from .run_log import RunLog
from .task import Task
from .tools import ToolRegistry


class Agent:
    def __init__(self, llm: LLM, tools: ToolRegistry, budget: Budget, log: RunLog):
        self._llm = llm
        self._tools = tools
        self._budget = budget
        self._log = log

    def run(self, task: Task) -> str | None:
        """Work on `task` until the model answers without calling a tool.

        Returns the model's final answer, or None if a call cap stopped the run.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt(self._tools.names)},
            {"role": "user", "content": user_message(task)},
        ]

        while True:
            if self._budget.llm_exhausted:
                return self._terminate("LLM call cap reached")

            self._log.write("Calling LLM for next tool to invoke")
            try:
                reply = self._llm.complete(
                    messages,
                    tools=self._tools.schemas,
                    tool_choice="auto",
                )
            except BadRequestError as error:
                # Usually a provider content filter: tell the model and let it try another way.
                detail = f"{type(error).__name__}: {error}"
                self._log.write(f"LLM BadRequestError = {detail}")
                messages.append({
                    "role": "user",
                    "content": LLM_ERROR_RECOVERY_MESSAGE.format(error=detail),
                })
                continue

            messages.append(reply.model_dump(exclude_none=True))
            if not reply.tool_calls:
                self._log.write(f"final response is = {reply.content}")
                return reply.content

            for call in reply.tool_calls:
                tool = self._tools.get(call.function.name)
                if self._budget.tools_exhausted:
                    return self._terminate("tool call cap reached")
                if tool is not None and tool.uses_llm and self._budget.llm_exhausted:
                    return self._terminate("LLM call cap reached")

                result = self._run_tool_call(call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, default=str),
                })

    def _run_tool_call(self, call: ChatCompletionMessageToolCall) -> Any:
        """Run one tool call. Failures are returned to the model as {"error": ...} so it can recover."""
        name = call.function.name
        tool = self._tools.get(name)
        if tool is None:
            self._log.write(f"** Unknown tool {name} **")
            return {"error": f"unknown tool {name!r}; available tools: {', '.join(self._tools.names)}"}

        try:
            args = json.loads(call.function.arguments)
            if not isinstance(args, dict):
                raise ValueError("expected a JSON object")
        except ValueError as error:  # json.JSONDecodeError is a ValueError
            self._log.write(f"** Invalid arguments for tool {name} **")
            return {"error": f"arguments for {name} are not valid JSON: {error}"}

        self._log.tool_entry(name, args)
        try:
            result = tool.handler(**args)
        except Exception as error:
            result = {"error": f"{type(error).__name__}: {error}"}
        self._log.tool_exit(name)
        self._budget.tool_calls += 1
        return result

    def _terminate(self, reason: str) -> None:
        self._log.write(f"** TERMINATED: {reason} **")
        return None
