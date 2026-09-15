"""Tests for reusing and starting OpenCode servers."""

import json
import os
import subprocess
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import IO, Any, ClassVar

import pytest

from opencode_power_pack import server as server_module
from opencode_power_pack.errors import OpencodeServerError
from opencode_power_pack.server import (
    OpencodeServer,
    _existing_server,
    data_home_for_database,
    opencode_server,
)


class _StubHandler(BaseHTTPRequestHandler):
    """Answers the endpoints the client uses, recording session mutations."""

    healthy = True
    port = 0
    requests: ClassVar[list[tuple[str, str, object]]] = []

    def _respond(self, status: int, payload: object) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _record(self) -> object:
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b""
        body = json.loads(raw) if raw else None
        type(self).requests.append((self.command, self.path, body))
        return body

    def do_GET(self) -> None:
        if self.path == "/global/health":
            self._respond(200, {"healthy": type(self).healthy, "version": "test"})
            return
        self._respond(404, {"name": "NotFoundError"})

    def do_DELETE(self) -> None:
        self._record()
        self._respond(200, True)

    def do_PATCH(self) -> None:
        self._record()
        self._respond(200, {"id": "ses_test"})

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def stub_server() -> Iterator[type[_StubHandler]]:
    """Serve the OpenCode endpoints on a loopback port owned by this process."""
    _StubHandler.requests = []
    _StubHandler.healthy = True
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    _StubHandler.port = int(httpd.server_address[1])
    try:
        yield _StubHandler
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture
def held_database(tmp_path: Path) -> Iterator[Path]:
    """A database file this process keeps open, as a running server would."""
    database = tmp_path / "share" / "opencode" / "opencode.db"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"")
    handle = database.open("rb")
    try:
        yield database
    finally:
        handle.close()


def _no_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("a server was started even though one was running")

    monkeypatch.setattr(subprocess, "Popen", fail)


def test_discovers_a_server_holding_the_database(
    held_database: Path, stub_server: type[_StubHandler]
) -> None:
    found = _existing_server(held_database)
    assert found is not None
    assert found.base_url.endswith(f":{stub_server.port}")


def test_ignores_servers_backed_by_another_database(
    held_database: Path, stub_server: type[_StubHandler], tmp_path: Path
) -> None:
    other = tmp_path / "other" / "opencode" / "opencode.db"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"")
    assert _existing_server(other) is None


def test_ignores_listeners_that_are_not_opencode(
    held_database: Path, stub_server: type[_StubHandler]
) -> None:
    stub_server.healthy = False
    assert _existing_server(held_database) is None


def test_reuses_a_running_server_without_starting_one(
    held_database: Path,
    stub_server: type[_StubHandler],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_spawn(monkeypatch)
    with opencode_server(held_database) as server:
        server.delete_session("ses_one")
        server.archive_session("ses_two", 1_700_000_000_000)
    assert stub_server.requests == [
        ("DELETE", "/session/ses_one", None),
        ("PATCH", "/session/ses_two", {"time": {"archived": 1_700_000_000_000}}),
    ]


def test_reuse_does_not_require_the_standard_layout(
    tmp_path: Path, stub_server: type[_StubHandler], monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "elsewhere.db"
    database.write_bytes(b"")
    handle = database.open("rb")
    try:
        _no_spawn(monkeypatch)
        with opencode_server(database) as server:
            server.delete_session("ses_one")
    finally:
        handle.close()
    assert stub_server.requests == [("DELETE", "/session/ses_one", None)]

    with pytest.raises(OpencodeServerError, match="data home"):
        data_home_for_database(database)


def test_reports_a_missing_database(tmp_path: Path) -> None:
    with pytest.raises(OpencodeServerError, match="does not exist"):
        with opencode_server(tmp_path / "share" / "opencode" / "opencode.db"):
            pass


class _FakeProcess:
    """A spawned server that never exits on its own."""

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        return 0


def _capture_spawns(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[list[str]], list[dict[str, Any]], list[_FakeProcess]]:
    commands: list[list[str]] = []
    options: list[dict[str, Any]] = []
    processes: list[_FakeProcess] = []

    def popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        commands.append(command)
        options.append(kwargs)
        process = _FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(server_module.shutil, "which", lambda _: "/usr/bin/opencode")
    monkeypatch.setattr(server_module, "_existing_server", lambda *_, **__: None)
    return commands, options, processes


def test_starts_a_server_in_the_data_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "share" / "opencode" / "opencode.db"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"")
    commands, options, processes = _capture_spawns(monkeypatch)

    def ready(
        process: object, output: IO[str], port: int, password: str | None
    ) -> OpencodeServer:
        return OpencodeServer(f"http://127.0.0.1:{port}")

    monkeypatch.setattr(server_module, "_await_ready", ready)

    with opencode_server(database) as server:
        assert server.base_url.startswith("http://127.0.0.1:")
    assert len(commands) == 1
    assert commands[0][1] == "serve"
    assert options[0]["cwd"] == os.fspath(database.parent)
    assert options[0]["env"]["XDG_DATA_HOME"] == os.fspath(tmp_path / "share")
    assert processes[0].terminated


def test_retries_once_when_the_port_was_taken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "share" / "opencode" / "opencode.db"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"")
    commands, _, processes = _capture_spawns(monkeypatch)
    attempts: list[int] = []

    def ready(
        process: object, output: IO[str], port: int, password: str | None
    ) -> OpencodeServer:
        attempts.append(port)
        if len(attempts) == 1:
            raise OpencodeServerError("opencode serve exited with code 1: ServeError")
        return OpencodeServer(f"http://127.0.0.1:{port}")

    monkeypatch.setattr(server_module, "_await_ready", ready)

    with opencode_server(database) as server:
        assert server.base_url.endswith(f":{attempts[1]}")
    assert len(commands) == 2
    assert attempts[0] != attempts[1]
    assert processes[0].terminated


def test_does_not_retry_other_startup_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "share" / "opencode" / "opencode.db"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"")
    commands, _, _ = _capture_spawns(monkeypatch)

    def ready(
        process: object, output: IO[str], port: int, password: str | None
    ) -> OpencodeServer:
        raise OpencodeServerError("opencode serve exited with code 1: boom")

    monkeypatch.setattr(server_module, "_await_ready", ready)

    with pytest.raises(OpencodeServerError, match="boom"):
        with opencode_server(database):
            pass
    assert len(commands) == 1


def test_requires_opencode_when_nothing_is_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "share" / "opencode" / "opencode.db"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"")
    monkeypatch.setattr(server_module, "_existing_server", lambda *_, **__: None)
    monkeypatch.setattr(server_module.shutil, "which", lambda _: None)

    with pytest.raises(OpencodeServerError, match="was not found on PATH"):
        with opencode_server(database):
            pass
