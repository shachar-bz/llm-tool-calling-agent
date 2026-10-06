import sqlite3

EXECUTE_SQL_QUERY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "execute_sql_query",
        "description": (
            "Execute a SQL query against a SQLite database file in read-only mode. "
            "Returns result rows as a list of dicts keyed by column name. If the SQL "
            "is invalid, returns an error row so the query can be rebuilt and retried."
        ),
        "parameters": {
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
    },
}


def execute_sql_query(sql: str, db_path: str) -> list:
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql)
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as exception:
        return [{"error": str(exception)}]
    finally:
        if conn is not None:
            conn.close()
