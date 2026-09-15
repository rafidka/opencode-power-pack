"""Access to OpenCode's own HTTP API for the mutating session commands.

Session deletion and archival are delegated to OpenCode so that cascading rules
(child sessions, messages, parts, todos, and the event log) stay owned by
OpenCode instead of being reimplemented against SQLite.

A local server already backed by the same database is reused when one is
running, which avoids a redundant process and lets its connected clients observe
the change. Otherwise a short-lived server is started for the duration of the
command.
"""

import os
import shutil
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import IO, cast

import httpx

from . import proc, system
from .errors import OpencodeServerError

_BINARY = "opencode"
_HOST = "127.0.0.1"
_STARTUP_TIMEOUT_SECONDS = 30.0
_STARTUP_POLL_SECONDS = 0.05
_SPAWN_ATTEMPTS = 3
_REQUEST_TIMEOUT_SECONDS = 60.0
_PROBE_TIMEOUT_SECONDS = 2.0
_PASSWORD_VARIABLE = "OPENCODE_SERVER_PASSWORD"
_PASSWORD_USER = "opencode"
_DATABASE_NAME = "opencode.db"
_DATA_DIRECTORY_NAME = "opencode"
_LISTENING_PREFIX = "opencode server listening on"
_PORT_CONFLICT_MARKERS = ("EADDRINUSE", "already in use", "ServeError")
_NOT_FOUND = 404
_UNAUTHORIZED = 401


class SessionNotFound(OpencodeServerError):
    """OpenCode reported that a session ID does not exist."""


def data_home_for_database(database: Path) -> Path:
    """Return the ``XDG_DATA_HOME`` value that makes OpenCode use ``database``.

    OpenCode resolves its database from its data directory rather than from a
    path option, so starting a server for a database requires the standard
    ``<data home>/opencode/opencode.db`` layout.
    """
    absolute = database.expanduser().absolute()
    if absolute.name != _DATABASE_NAME or absolute.parent.name != _DATA_DIRECTORY_NAME:
        msg = (
            f"cannot mutate sessions in {database}: the opencode server only reads "
            f"<data home>/{_DATA_DIRECTORY_NAME}/{_DATABASE_NAME}"
        )
        raise OpencodeServerError(msg)
    return absolute.parent.parent


class OpencodeServer:
    """A client for the session endpoints of a running ``opencode serve``.

    Requests are issued through ``httpx``, so an asynchronous counterpart needs
    only ``httpx.AsyncClient`` in place of the client built here.
    """

    def __init__(self, base_url: str, *, password: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._password = password

    def delete_session(self, session_id: str) -> None:
        """Delete a session and everything OpenCode cascades from it."""
        self._request("DELETE", f"/session/{session_id}", session_id=session_id)

    def archive_session(self, session_id: str, archived_at: int) -> None:
        """Mark a session archived at ``archived_at`` milliseconds."""
        self._request(
            "PATCH",
            f"/session/{session_id}",
            session_id=session_id,
            body={"time": {"archived": archived_at}},
        )

    def healthy(self) -> bool:
        """Return whether an OpenCode server answers this address.

        The response is inspected rather than only its status, because a reused
        address is discovered from the operating system and could belong to some
        other service the same process happens to be listening on.
        """
        try:
            payload = self._request(
                "GET", "/global/health", timeout=_PROBE_TIMEOUT_SECONDS
            )
        except OpencodeServerError:
            return False
        if not isinstance(payload, dict):
            return False
        return cast(dict[str, object], payload).get("healthy") is True

    def _client(self, timeout: float) -> httpx.Client:
        auth = (
            None
            if self._password is None
            else httpx.BasicAuth(_PASSWORD_USER, self._password)
        )
        return httpx.Client(base_url=self.base_url, timeout=timeout, auth=auth)

    def _request(
        self,
        method: str,
        path: str,
        *,
        session_id: str | None = None,
        body: Mapping[str, object] | None = None,
        timeout: float = _REQUEST_TIMEOUT_SECONDS,
    ) -> object:
        try:
            with self._client(timeout) as client:
                response = client.request(method, path, json=body)
        except httpx.HTTPError as error:
            msg = f"could not reach the opencode server at {self.base_url}: {error}"
            raise OpencodeServerError(msg) from error
        if response.is_error:
            raise self._failure(response, session_id)
        try:
            return response.json()
        except ValueError:
            return None

    def _failure(
        self, response: httpx.Response, session_id: str | None
    ) -> OpencodeServerError:
        if response.status_code == _NOT_FOUND and session_id is not None:
            return SessionNotFound(f"session {session_id} does not exist")
        if response.status_code == _UNAUTHORIZED:
            return OpencodeServerError(
                "the opencode server rejected the request credentials; "
                f"check {_PASSWORD_VARIABLE}"
            )
        detail = _error_detail(response)
        suffix = f": {detail}" if detail else ""
        return OpencodeServerError(
            f"opencode returned HTTP {response.status_code}{suffix}"
        )


@contextmanager
def opencode_server(database: Path) -> Generator[OpencodeServer]:
    """Provide a server backed by ``database`` for the duration of the block.

    A running local server on the same database is reused and left running.
    Otherwise ``opencode serve`` is started and stopped around the block.
    """
    target = _resolved(database)
    if not target.exists():
        msg = f"OpenCode database {database} does not exist"
        raise OpencodeServerError(msg)
    password = os.environ.get(_PASSWORD_VARIABLE) or None

    existing = _existing_server(target, password=password)
    if existing is not None:
        yield existing
        return

    with _spawned_server(target, password) as server:
        yield server


def _existing_server(
    database: Path, *, password: str | None = None
) -> OpencodeServer | None:
    """Return a healthy OpenCode server already backed by ``database``.

    Candidates are the listening sockets of this user's processes that hold
    ``database`` open, so a reused server is known to be writing to the intended
    file rather than merely answering on an expected port. Returns ``None`` where
    the operating system does not expose that mapping.
    """
    for port in system.ports_serving_file(database):
        server = OpencodeServer(f"http://{_HOST}:{port}", password=password)
        if server.healthy():
            return server
    return None


@contextmanager
def _spawned_server(target: Path, password: str | None) -> Generator[OpencodeServer]:
    binary = shutil.which(_BINARY)
    if binary is None:
        msg = (
            f"{_BINARY} was not found on PATH; deleting or archiving sessions "
            "requires the opencode CLI or an already running opencode server"
        )
        raise OpencodeServerError(msg)
    environment = dict(os.environ)
    environment["XDG_DATA_HOME"] = os.fspath(data_home_for_database(target))

    with proc.captured_output() as output:
        process: proc.Process | None = None
        server: OpencodeServer | None = None
        try:
            for attempt in range(1, _SPAWN_ATTEMPTS + 1):
                port = system.free_port()
                process = proc.launch(
                    [binary, "serve", f"--hostname={_HOST}", f"--port={port}"],
                    output=output,
                    environment=environment,
                    # The data directory keeps OpenCode from associating this
                    # short-lived server with the caller's working directory.
                    directory=target.parent,
                )
                try:
                    server = _await_ready(process, output, port, password)
                    break
                except OpencodeServerError as error:
                    proc.terminate(process)
                    process = None
                    if attempt == _SPAWN_ATTEMPTS or not _lost_the_port(error):
                        raise
                    proc.discard_output(output)
            assert server is not None
            yield server
        finally:
            if process is not None:
                proc.terminate(process)


def _await_ready(
    process: proc.Process,
    output: IO[str],
    port: int,
    password: str | None,
) -> OpencodeServer:
    """Wait for the announced URL, then for the server to answer requests.

    OpenCode accepts connections slightly before it serves them and leaves an
    early request open, so readiness is established from the announcement it
    prints and confirmed with short, retryable probes.
    """
    deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
    base_url = _await_announcement(process, output, deadline)
    server = OpencodeServer(base_url or f"http://{_HOST}:{port}", password=password)
    while not server.healthy():
        _fail_if_exited(process, output)
        if time.monotonic() >= deadline:
            raise _startup_error(
                "opencode serve did not answer requests within "
                f"{_STARTUP_TIMEOUT_SECONDS:.0f}s",
                output,
            )
        time.sleep(_STARTUP_POLL_SECONDS)
    return server


def _await_announcement(
    process: proc.Process, output: IO[str], deadline: float
) -> str | None:
    while True:
        _fail_if_exited(process, output)
        url = _announced_url(output)
        if url is not None:
            return url
        if time.monotonic() >= deadline:
            raise _startup_error(
                f"opencode serve did not start within {_STARTUP_TIMEOUT_SECONDS:.0f}s",
                output,
            )
        time.sleep(_STARTUP_POLL_SECONDS)


def _announced_url(output: IO[str]) -> str | None:
    for line in proc.output_lines(output):
        index = line.find(_LISTENING_PREFIX)
        if index == -1:
            continue
        remainder = line[index + len(_LISTENING_PREFIX) :].strip()
        candidate = remainder.split()[0] if remainder else ""
        if candidate.startswith("http"):
            return candidate
    return None


def _fail_if_exited(process: proc.Process, output: IO[str]) -> None:
    if process.poll() is None:
        return
    raise _startup_error(
        f"opencode serve exited with code {process.returncode}", output
    )


def _startup_error(message: str, output: IO[str]) -> OpencodeServerError:
    detail = proc.output_summary(output)
    return OpencodeServerError(f"{message}: {detail}" if detail else message)


def _lost_the_port(error: OpencodeServerError) -> bool:
    """Return whether a startup failure looks like the chosen port being taken."""
    message = str(error)
    return any(marker in message for marker in _PORT_CONFLICT_MARKERS)


def _resolved(database: Path) -> Path:
    return Path(os.path.realpath(database.expanduser()))


def _error_detail(response: httpx.Response) -> str:
    try:
        parsed = response.json()
    except ValueError:
        return " ".join(response.text.split())[:200]
    if isinstance(parsed, dict):
        mapping = cast(dict[str, object], parsed)
        data = mapping.get("data")
        if isinstance(data, dict):
            message = cast(dict[str, object], data).get("message")
            if isinstance(message, str):
                return message
        name = mapping.get("name")
        if isinstance(name, str):
            return name
    return ""
