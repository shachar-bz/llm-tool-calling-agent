"""An LLM agent built on a raw OpenAI tool-calling loop (no agent framework)."""

from .cli import run_task
from .config import Limits
from .llm import OpenAITransport

__all__ = ["Limits", "OpenAITransport", "run_task"]
__version__ = "0.1.0"
