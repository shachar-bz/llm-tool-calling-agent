"""Test doubles for driving the agent without a real model."""

import json

from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function

from tool_agent.budget import Budget


class ScriptedLLM:
    """Stands in for `LLM`: returns pre-written replies and records every request."""

    def __init__(self, budget: Budget, replies: list[ChatCompletionMessage]):
        self._budget = budget
        self._replies = list(replies)
        self.requests: list[list[dict]] = []

    def complete(self, messages, **kwargs):
        self._budget.llm_calls += 1
        self.requests.append(list(messages))
        return self._replies.pop(0)


def tool_call_reply(*calls: tuple[str, dict | str]) -> ChatCompletionMessage:
    """An assistant reply that calls tools; args may be a dict or a raw (possibly invalid) JSON string."""
    return ChatCompletionMessage(
        role="assistant",
        tool_calls=[
            ChatCompletionMessageToolCall(
                id=f"call_{index}",
                type="function",
                function=Function(
                    name=name,
                    arguments=args if isinstance(args, str) else json.dumps(args),
                ),
            )
            for index, (name, args) in enumerate(calls)
        ],
    )


def final_reply(text: str) -> ChatCompletionMessage:
    return ChatCompletionMessage(role="assistant", content=text)
