"""Tests for the sessions semantic index command."""

import json
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from typing import ClassVar

import httpx
import pytest
import typer
from cohere.core import ApiError
from rich.console import Console
from typer.testing import CliRunner

from opencode_power_pack.commands.sessions.index_sessions import register
from opencode_power_pack.errors import EmbeddingError
from opencode_power_pack.models import AppContext
from opencode_power_pack.semantic import (
    COHERE_DIMENSION,
    COHERE_MODEL,
    SemanticEmbeddingBatchCompleted,
    SemanticSessionSynchronized,
    SemanticSyncEvent,
    SemanticSyncPlanned,
    SemanticSyncResult,
)
from opencode_power_pack.semantic.cohere import CohereEmbedProvider


def _app(database: Path | None = None) -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def root(context: typer.Context) -> None:
        context.obj = AppContext(database=database)

    register(app)
    return app


class _Service:
    result: ClassVar[SemanticSyncResult] = SemanticSyncResult(
        no_op=False,
        indexed_session_ids=("session-1",),
        deleted_session_ids=("session-2",),
        source_changed=False,
    )
    calls: ClassVar[list[tuple[bool, bool]]] = []
    callbacks: ClassVar[list[Callable[[SemanticSyncEvent], None] | None]] = []
    events: ClassVar[list[SemanticSyncEvent]] = []
    sync_error: ClassVar[Exception | None] = None
    paths: ClassVar[list[tuple[Path, Path | None]]] = []

    def __init__(
        self, database: Path, embedder: object, sidecar_path: Path | None = None
    ) -> None:
        type(self).paths.append((database, sidecar_path))
        self.source_database = database
        self.sidecar_path = sidecar_path or Path("default-index.db")

    def sync(
        self,
        *,
        force: bool = False,
        rebuild: bool = False,
        on_progress: Callable[[SemanticSyncEvent], None] | None = None,
    ) -> SemanticSyncResult:
        type(self).calls.append((force, rebuild))
        type(self).callbacks.append(on_progress)
        for event in type(self).events:
            if on_progress is not None:
                on_progress(event)
        if type(self).sync_error is not None:
            raise type(self).sync_error
        return type(self).result


def _patch_command(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    from opencode_power_pack.commands.sessions import index_sessions

    providers: list[object] = []

    def create() -> object:
        provider = object()
        providers.append(provider)
        return provider

    monkeypatch.setattr(
        index_sessions,
        "_semantic_dependencies",
        lambda: (COHERE_MODEL, COHERE_DIMENSION, create, _Service),
    )
    monkeypatch.setattr(
        index_sessions, "resolve_database_path", lambda path: Path("source.db")
    )
    _Service.calls = []
    _Service.callbacks = []
    _Service.events = []
    _Service.sync_error = None
    _Service.paths = []
    _Service.result = SemanticSyncResult(
        no_op=False,
        indexed_session_ids=("session-1",),
        deleted_session_ids=("session-2",),
        source_changed=False,
    )
    return providers


def test_fixed_cohere_wiring_help_and_sidecar_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    providers = _patch_command(monkeypatch)
    runner = CliRunner()

    help_result = runner.invoke(_app(), ["index", "--help"])
    assert help_result.exit_code == 0
    assert "--semantic-index" in help_result.stdout
    for option in (
        "--embedding-provider",
        "--embedding-model",
        "--embedding-dimension",
    ):
        assert option not in help_result.stdout

    result = runner.invoke(_app(), ["index", "--semantic-index", "custom-index.db"])
    assert result.exit_code == 0
    assert len(providers) == 1
    assert _Service.paths == [(Path("source.db"), Path("custom-index.db"))]


def test_incremental_rebuild_force_validation_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_command(monkeypatch)
    runner = CliRunner()

    incremental = runner.invoke(_app(), ["index"])
    assert incremental.exit_code == 0
    assert "indexed 1 and removed 1" in incremental.stdout

    rebuilt = runner.invoke(_app(), ["index", "--rebuild"])
    assert rebuilt.exit_code == 0
    forced = runner.invoke(_app(), ["index", "--force"])
    assert forced.exit_code == 0
    assert _Service.calls == [(False, False), (False, True), (True, False)]

    invalid = runner.invoke(_app(), ["index", "--force", "--rebuild"])
    assert invalid.exit_code == 2
    assert "cannot be used" in invalid.stderr


def test_machine_output_and_missing_key_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_command(monkeypatch)
    _Service.result = SemanticSyncResult(
        no_op=True, indexed_session_ids=(), deleted_session_ids=(), source_changed=True
    )
    runner = CliRunner()

    machine = runner.invoke(_app(), ["index", "--format", "json"])
    assert machine.exit_code == 0
    assert json.loads(machine.stdout) == {
        "noOp": True,
        "indexedSessionIds": [],
        "deletedSessionIds": [],
        "indexedSessionCount": 0,
        "deletedSessionCount": 0,
        "sourceChanged": True,
        "provider": "cohere",
        "model": COHERE_MODEL,
        "dimension": COHERE_DIMENSION,
        "sourceDatabase": "source.db",
        "sidecarPath": "default-index.db",
    }

    from opencode_power_pack.commands.sessions import index_sessions

    def missing_key() -> object:
        raise EmbeddingError("COHERE_API_KEY is required")

    monkeypatch.setattr(
        index_sessions,
        "_semantic_dependencies",
        lambda: (COHERE_MODEL, COHERE_DIMENSION, missing_key, _Service),
    )
    error = runner.invoke(_app(), ["index"])
    assert error.exit_code == 1
    assert "COHERE_API_KEY is required" in error.stderr
    assert "Traceback" not in error.output


@pytest.mark.parametrize(
    ("provider_error", "expected"),
    [
        (
            ApiError(
                status_code=401,
                headers={"authorization": "secret-key-value"},
                body={"message": "private input"},
            ),
            "Error: Cohere authentication failed (HTTP 401): check COHERE_API_KEY",
        ),
        (
            ApiError(status_code=429, body={"message": "private input"}),
            "Error: Cohere rate limit exceeded (HTTP 429): retry later",
        ),
        (
            httpx.ConnectTimeout("timed out"),
            "Error: Cohere embedding request timed out; retry later",
        ),
    ],
)
def test_mocked_provider_errors_are_actionable_and_safe_at_cli(
    monkeypatch: pytest.MonkeyPatch, provider_error: Exception, expected: str
) -> None:
    _patch_command(monkeypatch)
    from opencode_power_pack.commands.sessions import index_sessions

    class Client:
        def embed(self, **_: object) -> object:
            raise provider_error

    class Service:
        def __init__(
            self, _: Path, embedder: CohereEmbedProvider, **__: object
        ) -> None:
            self.embedder = embedder

        def sync(self, **_: object) -> SemanticSyncResult:
            self.embedder.embed_documents(("private document text",))
            raise AssertionError("the mocked provider error should have stopped sync")

    monkeypatch.setattr(
        index_sessions,
        "_semantic_dependencies",
        lambda: (
            COHERE_MODEL,
            COHERE_DIMENSION,
            lambda: CohereEmbedProvider(client=Client()),
            Service,
        ),
    )

    result = CliRunner().invoke(_app(), ["index"])

    assert result.exit_code == 1
    assert expected in result.stderr
    assert "Traceback" not in result.output
    assert "secret-key-value" not in result.output
    assert "private input" not in result.output
    assert "private document text" not in result.output


def _force_tty_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> StringIO:
    from opencode_power_pack.commands.sessions import index_sessions

    stderr = StringIO()
    monkeypatch.setattr(
        index_sessions,
        "_stderr_console",
        lambda: Console(file=stderr, force_terminal=True, color_system=None),
    )
    return stderr


def test_tty_table_progress_uses_stderr_and_preserves_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_command(monkeypatch)
    stderr = _force_tty_progress(monkeypatch)
    _Service.events = [
        SemanticSyncPlanned(
            changed_session_count=1,
            deleted_session_count=1,
            document_count=2,
            batch_count=1,
        ),
        SemanticEmbeddingBatchCompleted(
            batch_number=1,
            batch_count=1,
            document_count=2,
            completed_documents=2,
            total_documents=2,
        ),
        SemanticSessionSynchronized(session_id="session-1"),
    ]
    from opencode_power_pack.semantic import SemanticSessionsDeleted

    _Service.events.append(SemanticSessionsDeleted(session_ids=("session-2",)))

    result = CliRunner().invoke(_app(), ["index"])

    assert result.exit_code == 0
    assert "Semantic index indexed 1 and removed 1 sessions." in result.stdout
    assert "Embedding documents" not in result.stdout
    assert "Embedding documents" in stderr.getvalue()
    assert "Synchronizing sessions" in stderr.getvalue()
    assert "Deleting sessions" in stderr.getvalue()
    assert _Service.callbacks[0] is not None


def test_non_tty_and_machine_formats_do_not_receive_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_command(monkeypatch)
    runner = CliRunner()

    table = runner.invoke(_app(), ["index"])
    assert table.exit_code == 0
    assert _Service.callbacks == [None]

    for output_format in ("json", "jsonl"):
        machine = runner.invoke(_app(), ["index", "--format", output_format])
        assert machine.exit_code == 0
        assert _Service.callbacks[-1] is None
        assert machine.stderr == ""


def test_no_op_does_not_render_progress_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_command(monkeypatch)
    stderr = _force_tty_progress(monkeypatch)
    _Service.result = SemanticSyncResult(
        no_op=True, indexed_session_ids=(), deleted_session_ids=(), source_changed=False
    )

    result = CliRunner().invoke(_app(), ["index"])

    assert result.exit_code == 0
    assert "already current" in result.stdout
    assert stderr.getvalue() == ""


def test_metadata_only_and_deletion_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_command(monkeypatch)
    stderr = _force_tty_progress(monkeypatch)
    _Service.result = SemanticSyncResult(
        no_op=False,
        indexed_session_ids=("session-1",),
        deleted_session_ids=("session-2",),
        source_changed=False,
    )
    from opencode_power_pack.semantic import SemanticSessionsDeleted

    _Service.events = [
        SemanticSyncPlanned(
            changed_session_count=1,
            deleted_session_count=1,
            document_count=0,
            batch_count=0,
        ),
        SemanticSessionSynchronized(session_id="session-1"),
        SemanticSessionsDeleted(session_ids=("session-2",)),
    ]

    result = CliRunner().invoke(_app(), ["index"])

    assert result.exit_code == 0
    assert "Synchronizing sessions" in stderr.getvalue()
    assert "Deleting sessions" in stderr.getvalue()
    assert "Embedding documents" not in stderr.getvalue()


def test_provider_error_stops_tty_progress_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_command(monkeypatch)
    stderr = _force_tty_progress(monkeypatch)
    _Service.events = [
        SemanticSyncPlanned(
            changed_session_count=1,
            deleted_session_count=0,
            document_count=1,
            batch_count=1,
        )
    ]
    _Service.sync_error = EmbeddingError("embedding failed")
    from opencode_power_pack.commands.sessions import index_sessions

    stopped = False
    original_stop = index_sessions.SyncProgressRenderer.stop

    def stop(renderer: index_sessions.SyncProgressRenderer) -> None:
        nonlocal stopped
        stopped = True
        original_stop(renderer)

    monkeypatch.setattr(index_sessions.SyncProgressRenderer, "stop", stop)
    result = CliRunner().invoke(_app(), ["index"])

    assert result.exit_code == 1
    assert stopped
    assert "embedding failed" in result.stderr
    assert "Traceback" not in result.output
    assert "Embedding documents" in stderr.getvalue()
