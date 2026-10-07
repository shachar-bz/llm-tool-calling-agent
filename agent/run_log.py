"""The run log: every agent step, echoed to stdout and to <query>.log."""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

PREVIEW_LENGTH = 50
PREVIEW_SUFFIX = "..."


class RunLog:
    def __init__(self, streams: list[TextIO]):
        self._streams = streams

    @classmethod
    @contextmanager
    def open(cls, path: Path) -> Iterator["RunLog"]:
        """A log that writes to stdout and to `path` (overwritten)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as log_file:
            yield cls([sys.stdout, log_file])

    def write(self, message: str) -> None:
        for stream in self._streams:
            stream.write(f"{message}\n")
            stream.flush()

    def tool_entry(self, name: str, args: dict[str, Any]) -> None:
        self.write(f"** Entering tool {name} **")
        for key, value in args.items():
            self.write(f"Parameter {key} = {_preview(value)}")

    def tool_exit(self, name: str) -> None:
        self.write(f"** Exiting tool {name} **")


def _preview(value: Any) -> str:
    text = str(value)
    if len(text) > PREVIEW_LENGTH:
        return text[:PREVIEW_LENGTH] + PREVIEW_SUFFIX
    return text
