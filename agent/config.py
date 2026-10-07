"""Run configuration: provider credentials come from the environment, call caps from the CLI."""

import os
from dataclasses import dataclass

DEFAULT_MODEL = "gpt-4.1-mini"
REQUIRED_ENV_VARS = ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT")


class ConfigError(Exception):
    """A required setting is missing."""


@dataclass(frozen=True)
class ProviderSettings:
    api_key: str
    endpoint: str
    model: str = DEFAULT_MODEL

    @classmethod
    def from_env(cls) -> "ProviderSettings":
        missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
        if missing:
            raise ConfigError(
                f"missing environment variable(s): {', '.join(missing)}. "
                "Copy .env.example to .env and fill them in."
            )
        return cls(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            model=os.environ.get("AZURE_OPENAI_MODEL", DEFAULT_MODEL),
        )


@dataclass(frozen=True)
class Limits:
    """Hard caps for one run. Every LLM request counts, including those made inside tools."""

    max_llm_calls: int = 20
    max_tool_calls: int = 20
