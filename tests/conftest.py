from pathlib import Path

import pytest

from tool_agent.task import Resource, Task


@pytest.fixture
def task(tmp_path: Path) -> Task:
    return Task(
        query_name="question.txt",
        query_text="What is 2 + 2?",
        resources=(Resource("data.db", "A SQLite database."),),
        root=tmp_path,
    )
