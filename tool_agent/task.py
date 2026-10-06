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
        """The transcript is named after the query: receipt_analysis.txt -> receipt_analysis.log."""
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
    """Turns file names chosen by the model into real paths under the task root."""

    def __init__(self, task: Task):
        self.root = task.root
        self._resource_aliases = _resource_aliases(task.resources)

    def resource_path(self, name: str) -> Path:
        """Path to the input resource the model means by `name`.

        Models often drop directory prefixes ("receipt.png" for "data/receipt.png"),
        so a bare file name is accepted when exactly one resource has it.
        """
        aliases = self._resource_aliases
        resolved = aliases.get(name) or aliases.get(os.path.basename(name), name)
        return self.root / resolved

    def output_path(self, name: str | os.PathLike) -> Path:
        return self.root / name


def _resource_aliases(resources: tuple[Resource, ...]) -> dict[str, str]:
    aliases = {resource.file_name: resource.file_name for resource in resources}
    basenames = [os.path.basename(resource.file_name) for resource in resources]
    for resource, basename in zip(resources, basenames):
        if basenames.count(basename) == 1:
            aliases.setdefault(basename, resource.file_name)
    return aliases
