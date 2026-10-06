"""A task (a query plus the resource files it may use) and the workspace it runs in."""

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Resource:
    file_name: str
    description: str


class TaskError(Exception):
    """The task manifest or its query file is missing or malformed."""


@dataclass(frozen=True)
class Task:
    """A query loaded from an input.json manifest.

    Every relative path in the task (query file, resources, outputs, log)
    is relative to `root`, the directory holding the manifest.
    """

    manifest_path: Path
    query_name: str
    query_text: str
    resources: tuple[Resource, ...]

    @property
    def root(self) -> Path:
        return self.manifest_path.parent

    @property
    def query_path(self) -> Path:
        return self.root / self.query_name

    @property
    def log_path(self) -> Path:
        """The run log is named after the query: receipt_analysis.txt -> receipt_analysis.log."""
        return self.root / Path(self.query_name).with_suffix(".log")


def load_task(manifest_path: str | os.PathLike) -> Task:
    manifest_path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        query_name = manifest["query_name"]
        resources = tuple(
            Resource(entry["file_name"], entry.get("description", ""))
            for entry in manifest["resources"]
        )
        query_text = (manifest_path.parent / query_name).read_text(encoding="utf-8")
    except KeyError as error:
        raise TaskError(f"{manifest_path.name} is missing the field {error}") from error
    except (OSError, ValueError, TypeError, AttributeError) as error:
        raise TaskError(f"cannot load task from {manifest_path}: {error}") from error
    return Task(manifest_path, query_name, query_text, resources)


class Workspace:
    """The task folder as tools see it: every path the model picks resolves inside it, or is refused.

    Refusals are PermissionErrors worded for the model, which gets them back
    as a tool error and can retry with a valid path.
    """

    def __init__(self, task: Task):
        self.root = task.root.resolve()
        self._resource_aliases = _resource_aliases(task.resources)
        inputs = [task.manifest_path, task.query_path, task.log_path]
        inputs += [task.root / resource.file_name for resource in task.resources]
        self._read_only = {path.resolve() for path in inputs}

    def resource_path(self, name: str) -> Path:
        """Path to the input resource the model means by `name`.

        Models often drop directory prefixes ("receipt.png" for "data/receipt.png"),
        so a bare file name is accepted when exactly one resource has it. Declared
        resources are trusted wherever they live; any other path must be inside the root.
        """
        aliases = self._resource_aliases
        declared = aliases.get(name) or aliases.get(os.path.basename(name))
        if declared is not None:
            return (self.root / declared).resolve()
        return self._inside_root(name)

    def output_path(self, name: str) -> Path:
        """Path for a file the model wants to write. The task's inputs and its log are off-limits."""
        path = self._inside_root(name)
        if path in self._read_only:
            raise PermissionError(f"{name} is an input or log of this task and must not be overwritten; write a new file")
        return path

    def _inside_root(self, name: str) -> Path:
        path = (self.root / name).resolve()  # also follows symlinks
        if not path.is_relative_to(self.root):
            raise PermissionError(f"{name} is outside the task folder; use a path inside it")
        return path


def _resource_aliases(resources: tuple[Resource, ...]) -> dict[str, str]:
    aliases = {resource.file_name: resource.file_name for resource in resources}
    basenames = [os.path.basename(resource.file_name) for resource in resources]
    for resource, basename in zip(resources, basenames):
        if basenames.count(basename) == 1:
            aliases.setdefault(basename, resource.file_name)
    return aliases
