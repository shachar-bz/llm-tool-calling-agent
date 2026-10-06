"""Hard caps on how much work a single agent run may do."""

from dataclasses import dataclass


@dataclass
class Budget:
    """Counts the LLM and tool calls made during one run.

    Every LLM request counts, including the ones LLM-backed tools make
    internally (image extraction, SQL generation).
    """

    max_llm_calls: int
    max_tool_calls: int
    llm_calls: int = 0
    tool_calls: int = 0

    @property
    def llm_exhausted(self) -> bool:
        return self.llm_calls >= self.max_llm_calls

    @property
    def tools_exhausted(self) -> bool:
        return self.tool_calls >= self.max_tool_calls
