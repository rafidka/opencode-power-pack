import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from opencode_power_pack import cli
from opencode_power_pack.commands.sessions import search_sessions
from opencode_power_pack.models import (
    SearchSessionMatch,
    SearchSource,
    Session,
)

runner = CliRunner()


@pytest.fixture
def search_api(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}
    match = SearchSessionMatch(
        session=Session(
            id="ses_123",
            title="A title\nwith whitespace",
            created=1,
            updated=1_700_000_000_000,
            project_id="project-1",
            directory="/work/example",
            repository="example",
        ),
        source=SearchSource.TEXT,
        role="assistant",
        snippet="A matching\nsnippet",
        match_count=2,
    )

    class Database:
        def __init__(self, path: Path) -> None:
            captured["path"] = path

        def __enter__(self) -> Database:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def list_repositories(self) -> tuple[str, ...]:
            return ()

        def search_sessions(self, query: object) -> tuple[SearchSessionMatch, ...]:
            captured["query"] = query
            return (match,)

    class Resolver:
        def __init__(self, repositories: object) -> None:
            captured["repositories"] = repositories

        def resolve_many(self, names: object) -> frozenset[str]:
            captured["repo_names"] = names
            return frozenset({f"id:{name}" for name in names})

        def resolve_cwd(self, cwd: Path) -> Session:
            captured["cwd"] = cwd
            return Session(
                id="",
                title=None,
                created=None,
                updated=None,
                project_id="cwd-project",
                directory=None,
            )

    def resolve(path: Path | None) -> Path:
        captured["explicit_path"] = path
        return Path("/resolved/database.db")

    monkeypatch.setattr(search_sessions, "OpenCodeDatabase", Database)
    monkeypatch.setattr(search_sessions, "RepositoryResolver", Resolver)
    monkeypatch.setattr(search_sessions, "resolve_database_path", resolve)
    captured["match"] = match
    return captured


def test_defaults_search_active_title_and_text(search_api: dict[str, object]) -> None:
    result = runner.invoke(
        cli.app, ["sessions", "search", "needle", "--format", "json"]
    )

    assert result.exit_code == 0
    query = search_api["query"]
    assert query.sources == frozenset({SearchSource.TITLE, SearchSource.TEXT})
    assert query.filters.archive_mode.value == "active"
    assert query.filters.repository_project_ids == frozenset()
    assert query.roles == frozenset({"user", "assistant"})


def test_search_collects_repeated_filters_and_all_sources(
    search_api: dict[str, object],
) -> None:
    result = runner.invoke(
        cli.app,
        [
            "sessions",
            "search",
            "needle",
            "-r",
            "one",
            "--repo",
            "two",
            "--role",
            "user",
            "--role",
            "assistant",
            "--agent",
            "build",
            "--provider",
            "openai",
            "--model",
            "model-x",
            "--in",
            "title",
            "--include-todos",
            "--include-reasoning",
            "--include-tool-inputs",
            "--include-tool-outputs",
            "--include-children",
            "--include-archived",
            "-n",
            "0",
            "--snippet-length",
            "10",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    query = search_api["query"]
    assert search_api["repo_names"] == ["one", "two"]
    assert query.filters.repository_project_ids == frozenset({"id:one", "id:two"})
    assert query.roles == frozenset({"user", "assistant"})
    assert query.sources == frozenset(
        {
            SearchSource.TITLE,
            SearchSource.TODO,
            SearchSource.REASONING,
            SearchSource.TOOL_INPUT,
            SearchSource.TOOL_OUTPUT,
        }
    )
    assert query.filters.include_children is True
    assert query.filters.archive_mode.value == "all"
    assert query.limit == 0
    assert query.snippet_length == 10


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["", "--format", "json"], "must not be empty"),
        (["needle", "--repo", "one", "--cwd"], "cannot be used"),
        (["needle", "--include-archived", "--archived-only"], "cannot be used"),
        (["needle", "-n", "-1"], "Invalid value"),
        (["needle", "--snippet-length", "0"], "Invalid value"),
        (
            [
                "needle",
                "--since",
                "2025-01-02T00:00:00Z",
                "--until",
                "2025-01-01T00:00:00Z",
            ],
            "must be earlier",
        ),
    ],
)
def test_search_validation_is_clean(arguments: list[str], message: str) -> None:
    result = runner.invoke(cli.app, ["sessions", "search", *arguments])

    assert result.exit_code == 2
    assert message in result.output
    assert "Traceback" not in result.output


def test_table_output_and_no_match(
    search_api: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    table = runner.invoke(cli.app, ["sessions", "search", "needle"])

    assert table.exit_code == 0
    assert "Updated" in table.output
    assert "A title" in table.output
    assert "matching" in table.output

    class EmptyDatabase:
        def __init__(self, _: Path) -> None:
            pass

        def __enter__(self) -> EmptyDatabase:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def list_repositories(self) -> tuple[str, ...]:
            return ()

        def search_sessions(self, _: object) -> tuple[()]:
            return ()

    monkeypatch.setattr(search_sessions, "OpenCodeDatabase", EmptyDatabase)
    empty = runner.invoke(cli.app, ["sessions", "search", "needle"])
    assert empty.exit_code == 0
    assert empty.output == "No matching sessions.\n"


def test_json_and_jsonl_are_exact_and_use_context_database(
    search_api: dict[str, object],
) -> None:
    arguments = ["--database", "/override.db", "sessions", "search", "needle"]
    json_result = runner.invoke(cli.app, [*arguments, "--format", "json"])
    jsonl_result = runner.invoke(cli.app, [*arguments, "--format", "jsonl"])
    expected = search_api["match"].as_dict()

    assert json_result.exit_code == 0
    assert json.loads(json_result.output) == [expected]
    assert jsonl_result.exit_code == 0
    assert json.loads(jsonl_result.output) == expected
    assert search_api["explicit_path"] == Path("/override.db")
