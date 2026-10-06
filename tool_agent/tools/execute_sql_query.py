"""Runs SQL against a SQLite file, opened read-only so the model cannot modify data."""

import sqlite3
from contextlib import closing
from pathlib import Path

from ..task import Workspace
from .base import Tool


def execute_sql_query(sql: str, db_path: Path) -> list[dict]:
    if not db_path.is_file():
        raise FileNotFoundError(f"no such database: {db_path.name}")

    read_only_uri = f"{db_path.resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(read_only_uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(sql).fetchall()]


def execute_sql_query_tool(workspace: Workspace) -> Tool:
    return Tool(
        name="execute_sql_query",
        description=(
            "Execute a SQL query against a SQLite database file in read-only mode. "
            "Returns result rows as a list of dicts keyed by column name. If the SQL "
            "is invalid, returns an error so the query can be rebuilt and retried."
        ),
        parameters={
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL string to execute.",
                },
                "db_path": {
                    "type": "string",
                    "description": "Filesystem path to the SQLite .db file.",
                },
            },
            "required": ["sql", "db_path"],
        },
        handler=lambda sql, db_path: execute_sql_query(sql, workspace.resource_path(db_path)),
    )
