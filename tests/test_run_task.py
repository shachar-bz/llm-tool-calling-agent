"""A whole run through the public entry points, offline: real task folder, real tools, scripted model."""

import json
import sqlite3
from pathlib import Path

import pytest
from support import ScriptedTransport, final_reply, tool_call_reply

from tool_agent import Limits, run_task
from tool_agent.cli import main

SCHEMA = "orders(merchant TEXT, amount REAL)"
SQL = "SELECT SUM(amount) AS total FROM orders WHERE merchant = 'Cafe'"


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    with sqlite3.connect(tmp_path / "orders.db") as connection:
        connection.execute("CREATE TABLE orders (merchant TEXT, amount REAL)")
        connection.executemany("INSERT INTO orders VALUES (?, ?)", [("Cafe", 10.0), ("Cafe", 5.5), ("Shop", 99.0)])
    connection.close()
    (tmp_path / "question.txt").write_text("Total for Cafe? Write it to answer.json.", encoding="utf-8")
    path = tmp_path / "input.json"
    path.write_text(json.dumps({
        "query_name": "question.txt",
        "resources": [{"file_name": "orders.db", "description": f"SQLite table {SCHEMA}"}],
    }), encoding="utf-8")
    return path


def test_run_task_solves_a_task_folder_end_to_end(manifest: Path):
    transport = ScriptedTransport(
        tool_call_reply(("build_sql_query", {"natural_language": "total for Cafe", "schema_description": SCHEMA})),
        final_reply(f"```sql\n{SQL}\n```"),  # the reply to build_sql_query's own request
        tool_call_reply(("execute_sql_query", {"sql": SQL, "db_path": "orders.db"})),
        tool_call_reply(("write_file", {"file_name": "answer.json", "file_content": '{"total": 15.5}'})),
        final_reply("The total is 15.5."),
    )

    answer = run_task(manifest, transport)

    root = manifest.parent
    assert answer == "The total is 15.5."
    assert json.loads((root / "answer.json").read_text()) == {"total": 15.5}
    tool_results = [m["content"] for m in transport.requests[-1]["messages"] if m["role"] == "tool"]
    assert [json.loads(r) for r in tool_results] == [SQL, [{"total": 15.5}], "Wrote 15 bytes to answer.json"]

    log = (root / "question.log").read_text().splitlines()
    assert log.count("Calling LLM for next tool to invoke") == 4
    assert [line for line in log if line.startswith("** Entering")] == [
        "** Entering tool build_sql_query **",
        "** Entering tool execute_sql_query **",
        "** Entering tool write_file **",
    ]
    assert log[-1] == "final response is = The total is 15.5."


def test_run_task_cap_covers_requests_made_inside_tools(manifest: Path):
    transport = ScriptedTransport(
        tool_call_reply(("build_sql_query", {"natural_language": "total", "schema_description": SCHEMA})),
    )

    assert run_task(manifest, transport, Limits(max_llm_calls=1)) is None
    assert len(transport.requests) == 1
    log = (manifest.parent / "question.log").read_text().splitlines()
    assert log[-1] == "** TERMINATED: LLM call cap reached **"


def test_main_reports_a_bad_manifest_without_a_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # keep the repo's .env out of the test
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.invalid/openai/v1/")
    (tmp_path / "input.json").write_text("{}", encoding="utf-8")

    assert main(["input.json"]) == 2
    assert "missing the field 'query_name'" in capsys.readouterr().err


def test_main_requires_provider_credentials(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)

    assert main([]) == 2
    assert "AZURE_OPENAI_API_KEY" in capsys.readouterr().err
