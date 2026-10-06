"""The chat-completions client shared by the agent loop and the LLM-backed tools."""

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

from .budget import Budget


class LLM:
    """Sends requests to one model and charges each request to the run's budget."""

    def __init__(self, client: OpenAI, model: str, budget: Budget):
        self._client = client
        self._model = model
        self._budget = budget

    def complete(self, messages: list[dict], **kwargs) -> ChatCompletionMessage:
        self._budget.llm_calls += 1
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message
