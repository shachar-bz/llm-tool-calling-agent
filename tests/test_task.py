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
    root = tmp_path.resolve()
    workspace = Workspace(make_task(tmp_path, "data/receipt.png"))

    assert workspace.resource_path("data/receipt.png") == root / "data/receipt.png"
    assert workspace.resource_path("receipt.png") == root / "data/receipt.png"
    assert workspace.resource_path("other/receipt.png") == root / "data/receipt.png"


def test_resource_path_does_not_guess_between_ambiguous_basenames(tmp_path: Path):
    root = tmp_path.resolve()
    workspace = Workspace(make_task(tmp_path, "a/x.db", "b/x.db"))

    assert workspace.resource_path("x.db") == root / "x.db"
    assert workspace.resource_path("b/x.db") == root / "b/x.db"


def test_declared_resources_are_trusted_even_outside_the_root(tmp_path: Path):
    task_root = tmp_path / "task"
    workspace = Workspace(make_task(task_root, "../shared/orders.db"))

    assert workspace.resource_path("orders.db") == tmp_path.resolve() / "shared/orders.db"


def test_output_path_resolves_inside_the_root(tmp_path: Path):
    root = tmp_path.resolve()
    workspace = Workspace(make_task(tmp_path))

    assert workspace.output_path("out/result.json") == root / "out/result.json"
    assert workspace.output_path(str(root / "result.json")) == root / "result.json"


@pytest.mark.parametrize("name", ["../escaped.txt", "out/../../escaped.txt", "/etc/hosts", "/tmp/x.json"])
def test_paths_outside_the_root_are_refused(tmp_path: Path, name: str):
    workspace = Workspace(make_task(tmp_path / "task"))

    with pytest.raises(PermissionError, match="outside the task folder"):
        workspace.output_path(name)
    with pytest.raises(PermissionError, match="outside the task folder"):
        workspace.resource_path(name)


def test_symlinks_cannot_escape_the_root(tmp_path: Path):
    (tmp_path / "task").mkdir()
    (tmp_path / "secret").mkdir()
    (tmp_path / "task" / "link").symlink_to(tmp_path / "secret")
    workspace = Workspace(make_task(tmp_path / "task"))

    with pytest.raises(PermissionError, match="outside the task folder"):
        workspace.output_path("link/stolen.txt")


@pytest.mark.parametrize("name", ["orders.db", "./orders.db", "input.json", "question.txt", "question.log"])
def test_task_inputs_and_log_cannot_be_overwritten(tmp_path: Path, name: str):
    workspace = Workspace(make_task(tmp_path, "orders.db"))

    with pytest.raises(PermissionError, match="must not be overwritten"):
        workspace.output_path(name)
