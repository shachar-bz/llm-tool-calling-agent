"""The agent loop: ask the model for its next tool call, run it, feed the result back, repeat."""

import json
from typing import Any

from openai.types.chat import ChatCompletionMessageToolCall

from .llm import LLM, CallCapReached, LLMRejected
from .prompts import REJECTION_RECOVERY_MESSAGE, REMOVED_TOOL_RESULT, system_prompt, user_message
from .run_log import RunLog
from .task import Task
from tools import ToolRegistry

# Keeps one oversized result (e.g. SELECT * on a big table) from overflowing the context.
MAX_TOOL_RESULT_CHARS = 20_000
MAX_REJECTION_DETAIL_CHARS = 1_000


class Agent:
    """Runs one task. The LLM enforces the LLM-call cap; the agent enforces the tool-call cap."""

    def __init__(self, llm: LLM, tools: ToolRegistry, log: RunLog, max_tool_calls: int):
        self._llm = llm
        self._tools = tools
        self._log = log
        self._max_tool_calls = max_tool_calls
        self._tool_calls = 0

    def run(self, task: Task) -> str | None:
        """Work on `task` until the model answers without calling a tool.

        Returns the model's final answer, or None if a call cap stopped the run.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt(self._tools.names)},
            {"role": "user", "content": user_message(task)},
        ]

        try:
            return self._loop(messages)
        except CallCapReached:
            # Raised by the agent's own request or by an LLM-backed tool mid-call.
            return self._terminate("LLM call cap reached")

    def _loop(self, messages: list[dict[str, Any]]) -> str | None:
        just_recovered = False
        while True:
            self._log.write("Calling LLM for next tool to invoke")
            try:
                reply = self._llm.complete(
                    messages,
                    tools=self._tools.schemas,
                    tool_choice="auto",
                )
            except LLMRejected as rejection:
                self._log.write(f"LLM request rejected ({rejection.reason}) = {rejection}")
                if just_recovered:
                    return self._terminate("LLM request rejected again after recovery")
                just_recovered = True
                self._recover_from_rejection(messages, rejection)
                continue
            just_recovered = False

            messages.append(reply.model_dump(exclude_none=True))
            if not reply.tool_calls:
                self._log.write(f"final response is = {reply.content}")
                return reply.content

            for call in reply.tool_calls:
                if self._tool_calls >= self._max_tool_calls:
                    return self._terminate("tool call cap reached")
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": _clip(json.dumps(self._run_tool_call(call), default=str), MAX_TOOL_RESULT_CHARS),
                })

    def _recover_from_rejection(self, messages: list[dict[str, Any]], rejection: LLMRejected) -> None:
        """Remove what most likely triggered the rejection (the newest tool results) and explain why.

        Resending the same conversation would just be rejected again.
        """
        for message in reversed(messages):
            if message["role"] != "tool":
                break
            message["content"] = REMOVED_TOOL_RESULT
        messages.append({
            "role": "user",
            "content": REJECTION_RECOVERY_MESSAGE.format(
                reason=rejection.reason,
                detail=_clip(str(rejection), MAX_REJECTION_DETAIL_CHARS),
            ),
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
        self._tool_calls += 1
        try:
            result = tool.handler(**args)
        except CallCapReached:
            raise
        except Exception as error:
            result = {"error": f"{type(error).__name__}: {error}"}
        self._log.tool_exit(name)
        return result

    def _terminate(self, reason: str) -> None:
        self._log.write(f"** TERMINATED: {reason} **")
        return None


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated {len(text) - limit} characters]"
