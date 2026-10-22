"""Access to OpenCode's own HTTP API for the mutating session commands.

Session deletion is delegated to OpenCode so that cascading rules (child
sessions, messages, and the event log) stay owned by OpenCode instead of being
reimplemented against SQLite.

Requests go through ``opencode api``, which finds or starts OpenCode's shared
background service and authenticates with it the same way the TUI does. That
service always uses its own database, so a command is refused when the service
is not the one writing to the database it was pointed at.
"""

import json
import os
import re
import shutil
import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from . import system
from .database import resolve_database_path
from .errors import OpencodeServerError

_BINARY = "opencode"
_REQUEST_TIMEOUT_SECONDS = 60.0
_NOT_FOUND = 404
_STATUS_PATTERN = re.compile(r"HTTP (\d{3})")
_DETAIL_LIMIT = 200


class SessionNotFound(OpencodeServerError):
    """OpenCode reported that a session ID does not exist."""


class OpencodeServer:
    """A client for the session endpoints of OpenCode's background service."""

    def __init__(self, binary: str) -> None:
        self.binary = binary

    def delete_session(self, session_id: str) -> None:
        """Delete a session and everything OpenCode cascades from it."""
        self._request("DELETE", f"/api/session/{session_id}", session_id=session_id)

    def info(self) -> dict[str, object]:
        """Return the service's ``/api/info``, starting the service if needed."""
        payload = self._request("GET", "/api/info")
        if not isinstance(payload, dict):
            msg = "opencode returned an unexpected response from /api/info"
            raise OpencodeServerError(msg)
        return cast(dict[str, object], payload)

    def _request(
        self, method: str, path: str, *, session_id: str | None = None
    ) -> object:
        try:
            completed = subprocess.run(
                [self.binary, "api", method.lower(), path],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=_REQUEST_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            msg = f"could not reach the opencode service: {error}"
            raise OpencodeServerError(msg) from error
        payload = _json(completed.stdout)
        if completed.returncode != 0:
            raise _failure(completed.stderr, payload, completed.stdout, session_id)
        return payload


@contextmanager
def opencode_server(database: Path) -> Generator[OpencodeServer]:
    """Provide OpenCode's background service, once it is known to use ``database``."""
    target = Path(os.path.realpath(database.expanduser()))
    if not target.exists():
        msg = f"OpenCode database {database} does not exist"
        raise OpencodeServerError(msg)
    binary = shutil.which(_BINARY)
    if binary is None:
        msg = f"{_BINARY} was not found on PATH; deleting sessions requires it"
        raise OpencodeServerError(msg)
    server = OpencodeServer(binary)
    pid = server.info().get("pid")
    if not isinstance(pid, int) or not _serves(pid, target):
        msg = (
            f"the running OpenCode service does not use {database}; sessions can "
            "only be changed in the database of OpenCode's background service"
        )
        raise OpencodeServerError(msg)
    yield server


def _serves(pid: int, database: Path) -> bool:
    """Return whether the service process writes to ``database``.

    Where the operating system does not expose open files, the service is
    assumed to use OpenCode's default database.
    """
    holds = system.process_holds_file(pid, database)
    if holds is not None:
        return holds
    return database == Path(os.path.realpath(resolve_database_path()))


def _json(text: str) -> object:
    try:
        return json.loads(text)
    except ValueError:
        return None


def _failure(
    stderr: str, payload: object, stdout: str, session_id: str | None
) -> OpencodeServerError:
    match = _STATUS_PATTERN.search(stderr)
    status = int(match.group(1)) if match else None
    if status == _NOT_FOUND and session_id is not None:
        return SessionNotFound(f"session {session_id} does not exist")
    detail = _detail(payload) or " ".join((stderr or stdout).split())[:_DETAIL_LIMIT]
    prefix = f"opencode returned HTTP {status}" if status else "opencode api failed"
    return OpencodeServerError(f"{prefix}: {detail}" if detail else prefix)


def _detail(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    mapping = cast(dict[str, object], payload)
    for key in ("message", "_tag"):
        value = mapping.get(key)
        if isinstance(value, str):
            return value
    return ""
