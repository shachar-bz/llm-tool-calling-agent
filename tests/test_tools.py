import sqlite3
from pathlib import Path

import pytest

from support import make_task

from tool_agent.task import Workspace
from tool_agent.tools.base import strip_code_fences
from tool_agent.tools.execute_sql_query import execute_sql_query, execute_sql_query_tool
from tool_agent.tools.write_file import write_file_tool


@pytest.fixture
def orders_db(tmp_path: Path) -> Path:
    # A directory name with spaces, like a typical macOS Documents path.
    db_path = tmp_path / "my data" / "orders.db"
    db_path.parent.mkdir()
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE orders (merchant TEXT, amount REAL)")
        connection.executemany(
            "INSERT INTO orders VALUES (?, ?)",
            [("Cafe", 10.0), ("Cafe", 5.5), ("Shop", 99.0)],
        )
    connection.close()
    return db_path


def test_execute_sql_query_returns_rows_as_dicts(orders_db: Path):
    rows = execute_sql_query(
        "SELECT merchant, SUM(amount) AS total FROM orders GROUP BY merchant ORDER BY merchant",
        orders_db,
    )
    assert rows == [{"merchant": "Cafe", "total": 15.5}, {"merchant": "Shop", "total": 99.0}]


def test_execute_sql_query_is_read_only(orders_db: Path):
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        execute_sql_query("DELETE FROM orders", orders_db)


def test_execute_sql_query_reports_missing_database(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        execute_sql_query("SELECT 1", tmp_path / "missing.db")


def test_execute_sql_tool_resolves_resource_names(orders_db: Path, tmp_path: Path):
    tool = execute_sql_query_tool(Workspace(make_task(tmp_path, "my data/orders.db")))

    assert tool.handler(sql="SELECT COUNT(*) AS n FROM orders", db_path="orders.db") == [{"n": 3}]


def test_write_file_tool_writes_under_the_task_root(tmp_path: Path):
    tool = write_file_tool(Workspace(make_task(tmp_path)))

    message = tool.handler(file_content='{"ok": true}', file_name="out/result.json")

    assert (tmp_path / "out" / "result.json").read_text(encoding="utf-8") == '{"ok": true}'
    assert message == "Wrote 12 bytes to out/result.json"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SELECT 1", "SELECT 1"),
        ("```sql\nSELECT 1\n```", "SELECT 1"),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ("Here you go:\n```\nSELECT 1\n```\nEnjoy.", "SELECT 1"),
        ("```sql\nSELECT 1", "SELECT 1"),
    ],
)
def test_strip_code_fences(text, expected):
    assert strip_code_fences(text) == expected
