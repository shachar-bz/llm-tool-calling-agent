"""The single gateway for model requests, shared by the agent loop and the LLM-backed tools."""

from typing import Any, Protocol

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage


class CallCapReached(Exception):
    """The run has used its LLM-call cap; no further requests will be sent."""


class Transport(Protocol):
    """Sends one chat-completion request and returns the model's reply message."""

    def __call__(self, *, messages: list[dict], **options: Any) -> ChatCompletionMessage: ...


class OpenAITransport:
    """Production transport: an OpenAI-compatible chat-completions endpoint (here, Azure OpenAI)."""

    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    def __call__(self, *, messages: list[dict], **options: Any) -> ChatCompletionMessage:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **options,
        )
        return response.choices[0].message


class LLM:
    """Counts every model request in a run and refuses once the cap is reached.

    Because the agent loop and LLM-backed tools all go through the same LLM,
    the cap holds no matter which of them makes the request.
    """

    def __init__(self, transport: Transport, max_calls: int):
        self._transport = transport
        self._max_calls = max_calls
        self._calls = 0

    def complete(self, messages: list[dict], **options: Any) -> ChatCompletionMessage:
        """Send `messages` to the model. Raises CallCapReached once the cap is used up."""
        if self._calls >= self._max_calls:
            raise CallCapReached(f"LLM call cap of {self._max_calls} reached")
        self._calls += 1
        return self._transport(messages=messages, **options)
