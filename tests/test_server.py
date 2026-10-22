"""Tests for reaching OpenCode's background service through ``opencode api``."""

import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from opencode_power_pack import server as server_module
from opencode_power_pack import system
from opencode_power_pack.errors import OpencodeServerError
from opencode_power_pack.server import OpencodeServer, SessionNotFound, opencode_server


def _completed(
    command: list[str], *, stdout: object = None, stderr: str = "", code: int = 0
) -> subprocess.CompletedProcess[str]:
    body = "" if stdout is None else json.dumps(stdout)
    return subprocess.CompletedProcess(command, code, body, stderr)


class _FakeApi:
    """Records ``opencode api`` invocations and answers them like the service."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.commands: list[list[str]] = []
        self.missing: set[str] = set()

    def __call__(
        self, command: list[str], **_: object
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        method, path = command[2], command[3]
        if (method, path) == ("get", "/api/info"):
            return _completed(command, stdout={"version": "2.0.0", "pid": self.pid})
        session_id = path.rsplit("/", 1)[-1]
        if method == "delete" and session_id in self.missing:
            return _completed(
                command,
                stdout={"_tag": "SessionNotFoundError", "message": "Session not found"},
                stderr="HTTP 404 Not Found\n",
                code=1,
            )
        if method == "delete":
            return _completed(command, stdout=True)
        return _completed(
            command,
            stdout={"_tag": "BadRequest", "message": "nope"},
            stderr="HTTP 400 Bad Request\n",
            code=1,
        )


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> _FakeApi:
    fake = _FakeApi(pid=os.getpid())
    monkeypatch.setattr(server_module.subprocess, "run", fake)
    monkeypatch.setattr(server_module.shutil, "which", lambda _: "/usr/bin/opencode")
    return fake


@pytest.fixture
def held_database(tmp_path: Path) -> Iterator[Path]:
    """A database file this process keeps open, as the service would."""
    database = tmp_path / "share" / "opencode" / "opencode.db"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"")
    with database.open("rb"):
        yield database


def test_deletes_through_the_v2_session_endpoint(
    held_database: Path, api: _FakeApi
) -> None:
    with opencode_server(held_database) as server:
        server.delete_session("ses_one")

    assert api.commands == [
        ["/usr/bin/opencode", "api", "get", "/api/info"],
        ["/usr/bin/opencode", "api", "delete", "/api/session/ses_one"],
    ]


def test_missing_session_is_reported_as_not_found(
    held_database: Path, api: _FakeApi
) -> None:
    api.missing.add("ses_gone")
    with opencode_server(held_database) as server:
        with pytest.raises(SessionNotFound, match="ses_gone does not exist"):
            server.delete_session("ses_gone")


def test_other_failures_carry_the_status_and_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return _completed(
            command,
            stdout={"_tag": "Boom", "message": "it broke"},
            stderr="HTTP 500 Internal Server Error\n",
            code=1,
        )

    monkeypatch.setattr(server_module.subprocess, "run", run)

    with pytest.raises(OpencodeServerError, match="HTTP 500: it broke"):
        OpencodeServer("/usr/bin/opencode").delete_session("ses_one")


def test_refuses_a_service_that_does_not_use_the_database(
    tmp_path: Path, api: _FakeApi
) -> None:
    database = tmp_path / "other" / "opencode" / "opencode.db"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"")

    with pytest.raises(OpencodeServerError, match="does not use"):
        with opencode_server(database):
            pass
    assert [command[2] for command in api.commands] == ["get"]


def test_falls_back_to_the_default_database_without_proc(
    tmp_path: Path, api: _FakeApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "data" / "opencode" / "opencode.db"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"")
    other = tmp_path / "other.db"
    other.write_bytes(b"")
    monkeypatch.setattr(system, "process_holds_file", lambda *_: None)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    with opencode_server(database):
        pass
    with pytest.raises(OpencodeServerError, match="does not use"):
        with opencode_server(other):
            pass


def test_reports_a_missing_database(tmp_path: Path) -> None:
    with pytest.raises(OpencodeServerError, match="does not exist"):
        with opencode_server(tmp_path / "share" / "opencode" / "opencode.db"):
            pass


def test_requires_opencode_on_path(
    held_database: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server_module.shutil, "which", lambda _: None)

    with pytest.raises(OpencodeServerError, match="was not found on PATH"):
        with opencode_server(held_database):
            pass


def test_process_holds_file_inspects_open_descriptors(
    held_database: Path, tmp_path: Path
) -> None:
    if system.process_holds_file(os.getpid(), held_database) is None:
        pytest.skip("open files are not observable on this platform")

    assert system.process_holds_file(os.getpid(), held_database) is True
    assert system.process_holds_file(os.getpid(), tmp_path / "closed.db") is False
