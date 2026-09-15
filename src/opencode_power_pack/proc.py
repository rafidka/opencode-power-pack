"""Launching, watching, and stopping child processes.

Child output is captured to a temporary file rather than a pipe so that a process
can never block on a full pipe buffer while callers wait for it to become ready.
"""

import os
import subprocess
import tempfile
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import IO

type Process = subprocess.Popen[bytes]

_SHUTDOWN_TIMEOUT_SECONDS = 5.0
_SUMMARY_LIMIT = 500


@contextmanager
def captured_output() -> Generator[IO[str]]:
    """Provide a rewindable file for a child process's combined output."""
    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as output:
        yield output


def launch(
    command: Sequence[str],
    *,
    output: IO[str],
    environment: Mapping[str, str] | None = None,
    directory: Path | None = None,
) -> Process:
    """Start ``command`` with its output and errors captured into ``output``."""
    return subprocess.Popen(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=output,
        stderr=subprocess.STDOUT,
        env=None if environment is None else dict(environment),
        cwd=None if directory is None else os.fspath(directory),
    )


def terminate(process: Process) -> None:
    """Stop ``process``, escalating to a kill if it does not exit promptly."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def output_lines(output: IO[str]) -> tuple[str, ...]:
    """Return everything a child has written so far, as lines."""
    try:
        output.seek(0)
        return tuple(output.read().splitlines())
    except OSError:
        return ()


def output_summary(output: IO[str], *, limit: int = _SUMMARY_LIMIT) -> str:
    """Return a single-line, length-capped summary of a child's output."""
    return " ".join(" ".join(output_lines(output)).split())[:limit]


def discard_output(output: IO[str]) -> None:
    """Forget what a child has written, before reusing the file for a retry."""
    try:
        output.seek(0)
        output.truncate()
    except OSError:
        return
