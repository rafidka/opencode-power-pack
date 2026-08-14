"""Tests for the sessions list command."""

import json
from pathlib import Path

import typer
from typer.testing import CliRunner

from opencode_power_pack.commands.sessions.list_sessions import register
from opencode_power_pack.models import AppContext, Repository, Session, TokenBreakdown


def _app(database: Path | None = None) -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def root(context: typer.Context) -> None:
        context.obj = AppContext(database=database)

    register(app)
    return app


class _Database:
    def __init__(self, sessions: list[Session], repositories: list[Repository]) -> None:
        self.sessions = sessions
        self.repositories = repositories
        self.query = None

    def __enter__(self) -> _Database:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def list_repositories(self) -> list[Repository]:
        return self.repositories

    def list_sessions(self, query: object) -> list[Session]:
        self.query = query
        return self.sessions


def _session(**changes: object) -> Session:
    values: dict[str, object] = {
        "id": "ses_1",
        "title": "A session",
        "created": 1_700_000_000_000,
        "updated": 1_700_000_100_000,
        "project_id": "project-1",
        "directory": "/work/project",
        "repository": "project",
        "agent": "build",
        "provider": "openai",
        "model": "gpt",
        "tokens": TokenBreakdown(
            input=1, output=2, reasoning=3, cache_read=4, cache_write=5
        ),
    }
    values.update(changes)
    return Session(**values)  # type: ignore[arg-type]


def _patch_database(monkeypatch: object, database: _Database) -> list[Path | None]:
    from opencode_power_pack.commands.sessions import list_sessions

    paths: list[Path | None] = []

    def resolve(path: Path | None) -> Path:
        paths.append(path)
        return Path("db")

    monkeypatch.setattr(list_sessions, "resolve_database_path", resolve)
    monkeypatch.setattr(list_sessions, "OpenCodeDatabase", lambda path: database)
    return paths


def test_maps_options_to_query(monkeypatch: object) -> None:
    database = _Database([_session()], [])
    _patch_database(monkeypatch, database)

    result = CliRunner().invoke(
        _app(),
        [
            "list",
            "-n",
            "3",
            "--include-children",
            "--include-archived",
            "--agent",
            "build",
            "--provider",
            "openai",
            "--model",
            "gpt",
            "--since",
            "2024-01-01T00:00:00Z",
            "--until",
            "2024-02-01T00:00:00Z",
            "--sort",
            "cost",
            "--reverse",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    assert database.query.limit == 3
    assert database.query.sort.value == "cost"
    assert database.query.direction.value == "asc"
    assert database.query.filters.include_children is True
    assert database.query.filters.archive_mode.value == "all"
    assert database.query.filters.agents == frozenset({"build"})
    assert database.query.filters.providers == frozenset({"openai"})
    assert database.query.filters.models == frozenset({"gpt"})


def test_json_and_jsonl_output(monkeypatch: object) -> None:
    database = _Database([_session()], [])
    _patch_database(monkeypatch, database)

    json_result = CliRunner().invoke(_app(), ["list", "--format", "json"])
    assert json_result.exit_code == 0
    assert json.loads(json_result.stdout) == [_session().as_dict()]

    jsonl_result = CliRunner().invoke(_app(), ["list", "--format", "jsonl"])
    assert jsonl_result.exit_code == 0
    assert [json.loads(line) for line in jsonl_result.stdout.splitlines()] == [
        _session().as_dict()
    ]


def test_table_empty_and_validation(monkeypatch: object) -> None:
    database = _Database([], [])
    _patch_database(monkeypatch, database)
    runner = CliRunner()

    empty = runner.invoke(_app(), ["list"])
    assert empty.exit_code == 0
    assert empty.stdout == "No sessions found.\n"

    database.sessions = [_session(parent_id="ses_parent")]
    table = runner.invoke(_app(), ["list", "--include-children"])
    assert table.exit_code == 0
    assert "Updated" in table.stdout
    assert "Session ID" in table.stdout
    assert "Parent" in table.stdout
    assert "ses_pare" in table.stdout

    invalid = runner.invoke(_app(), ["list", "--repo", "one", "--cwd"])
    assert invalid.exit_code == 2
    assert "--repo cannot be used with --cwd" in invalid.stderr


def test_global_context_and_repository_selection(monkeypatch: object) -> None:
    database = _Database(
        [],
        [
            Repository(
                display_name="project", project_id="project-1", path="/work/project"
            )
        ],
    )
    paths = _patch_database(monkeypatch, database)
    monkeypatch.setattr(
        "opencode_power_pack.commands.sessions.list_sessions.Path.cwd",
        lambda: Path("/work/project/subdirectory"),
    )
    runner = CliRunner()

    global_result = runner.invoke(
        _app(Path("configured.db")), ["list", "--format", "json"]
    )
    assert global_result.exit_code == 0
    assert paths == [Path("configured.db")]
    assert database.query.filters.repository_project_ids == frozenset()

    repository_result = runner.invoke(
        _app(), ["list", "--repo", "project", "--format", "json"]
    )
    assert repository_result.exit_code == 0
    assert database.query.filters.repository_project_ids == frozenset({"project-1"})

    cwd_result = runner.invoke(_app(), ["list", "--cwd", "--format", "json"])
    assert cwd_result.exit_code == 0
    assert database.query.filters.repository_project_ids == frozenset({"project-1"})
