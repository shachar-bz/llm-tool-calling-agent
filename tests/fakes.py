"""Test doubles for driving the agent without a real model."""

import json

from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function


class ScriptedTransport:
    """Stands in for the provider: plays back replies (or raises exceptions) in order."""

    def __init__(self, *steps: ChatCompletionMessage | Exception):
        self._steps = list(steps)
        self.requests: list[dict] = []

    def __call__(self, *, messages, **options):
        self.requests.append({"messages": list(messages), **options})
        step = self._steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


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


def final_reply(text: str | None) -> ChatCompletionMessage:
    return ChatCompletionMessage(role="assistant", content=text)
