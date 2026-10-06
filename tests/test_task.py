import json
from pathlib import Path

import pytest
from support import make_task

from tool_agent.task import Resource, TaskError, Workspace, load_task


def write_manifest(root: Path, manifest) -> Path:
    path = root / "input.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_load_task_reads_manifest_and_query(tmp_path: Path):
    (tmp_path / "q.txt").write_text("  Find the total.\n", encoding="utf-8")
    manifest = write_manifest(tmp_path, {
        "query_name": "q.txt",
        "resources": [{"file_name": "orders.db", "description": "Orders."}],
    })

    task = load_task(manifest)

    assert task.root == tmp_path.resolve()
    assert task.query_text == "  Find the total.\n"
    assert task.resources == (Resource("orders.db", "Orders."),)
    assert task.log_path == tmp_path.resolve() / "q.log"


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ({"resources": []}, "missing the field 'query_name'"),
        ({"query_name": "missing.txt", "resources": []}, "No such file"),
        ([1, 2], "cannot load task"),
    ],
)
def test_load_task_reports_bad_manifests(tmp_path: Path, manifest, message):
    with pytest.raises(TaskError, match=message):
        load_task(write_manifest(tmp_path, manifest))


def test_resource_path_accepts_exact_name_and_unique_basename(tmp_path: Path):
    workspace = Workspace(make_task(tmp_path, "data/receipt.png"))

    assert workspace.resource_path("data/receipt.png") == tmp_path / "data/receipt.png"
    assert workspace.resource_path("receipt.png") == tmp_path / "data/receipt.png"
    assert workspace.resource_path("other/receipt.png") == tmp_path / "data/receipt.png"


def test_resource_path_does_not_guess_between_ambiguous_basenames(tmp_path: Path):
    workspace = Workspace(make_task(tmp_path, "a/x.db", "b/x.db"))

    assert workspace.resource_path("x.db") == tmp_path / "x.db"
    assert workspace.resource_path("b/x.db") == tmp_path / "b/x.db"


def test_output_path_is_relative_to_root(tmp_path: Path):
    workspace = Workspace(make_task(tmp_path))

    assert workspace.output_path("out/result.json") == tmp_path / "out/result.json"
