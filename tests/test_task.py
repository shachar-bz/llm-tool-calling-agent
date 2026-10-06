import json
from pathlib import Path

from tool_agent.task import Resource, Workspace, load_task


def test_load_task_reads_manifest_and_query(tmp_path: Path):
    (tmp_path / "q.txt").write_text("  Find the total.\n", encoding="utf-8")
    (tmp_path / "input.json").write_text(
        json.dumps({
            "query_name": "q.txt",
            "resources": [{"file_name": "orders.db", "description": "Orders."}],
        }),
        encoding="utf-8",
    )

    task = load_task(tmp_path / "input.json")

    assert task.root == tmp_path.resolve()
    assert task.query_text == "  Find the total.\n"
    assert task.resources == (Resource("orders.db", "Orders."),)
    assert task.log_name == Path("q.log")


def test_resource_path_accepts_exact_name_and_unique_basename(tmp_path: Path):
    workspace = Workspace(tmp_path, (Resource("data/receipt.png", ""),))

    assert workspace.resource_path("data/receipt.png") == tmp_path / "data/receipt.png"
    assert workspace.resource_path("receipt.png") == tmp_path / "data/receipt.png"
    assert workspace.resource_path("other/receipt.png") == tmp_path / "data/receipt.png"


def test_resource_path_does_not_guess_between_ambiguous_basenames(tmp_path: Path):
    workspace = Workspace(tmp_path, (Resource("a/x.db", ""), Resource("b/x.db", "")))

    assert workspace.resource_path("x.db") == tmp_path / "x.db"
    assert workspace.resource_path("b/x.db") == tmp_path / "b/x.db"


def test_output_path_is_relative_to_root_unless_absolute(tmp_path: Path):
    workspace = Workspace(tmp_path)

    assert workspace.output_path("out/result.json") == tmp_path / "out/result.json"
    assert workspace.output_path("/abs/result.json") == Path("/abs/result.json")
