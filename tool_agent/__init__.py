"""An LLM agent built on a raw OpenAI tool-calling loop (no agent framework)."""

from .cli import run_task

__all__ = ["run_task"]
__version__ = "0.1.0"
