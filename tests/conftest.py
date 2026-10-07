from pathlib import Path

import pytest
from support import make_task

from agent.task import Task


@pytest.fixture
def task(tmp_path: Path) -> Task:
    return make_task(tmp_path, "data.db")
