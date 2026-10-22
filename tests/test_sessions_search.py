import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from opencode_power_pack import cli
from opencode_power_pack.commands.sessions import search_sessions
from opencode_power_pack.errors import EmbeddingError
from opencode_power_pack.models import (
    SearchSessionMatch,
    SearchSource,
    Session,
)
from opencode_power_pack.semantic import SemanticSearchMatch

runner = CliRunner()

# The published JSON shape, written out rather than derived from the models, so a
# change to how they serialize cannot quietly change the contract.
_EXPECTED_SESSION = {
    "id": "ses_123",
    "title": "A title\nwith whitespace",
    "created": 1,
    "updated": 1_700_000_000_000,
    "projectId": "project-1",
    "directory": "/work/example",
    "parentId": None,
    "repository": "example-repo",
    "agent": None,
    "provider": None,
    "model": None,
    "cost": None,
    "tokens": None,
    "archived": None,
    "archivedAt": None,
}


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
            repository="example-repo",
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


def test_defaults_search_active_title_and_text(
    search_api: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
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
    table = runner.invoke(
        cli.app, ["sessions", "search", "needle"], env={"COLUMNS": "240"}
    )

    assert table.exit_code == 0
    assert "Updated" in table.output
    assert "Repository" in table.output
    assert "example-repo" in table.output
    assert "/work/example" in table.output
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


def test_table_abbreviates_home_directory(
    search_api: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(search_sessions.Path, "home", lambda: Path("/work"))

    table = runner.invoke(
        cli.app, ["sessions", "search", "needle"], env={"COLUMNS": "240"}
    )

    assert table.exit_code == 0
    assert "~/example" in table.output


def test_json_and_jsonl_are_exact_and_use_context_database(
    search_api: dict[str, object],
) -> None:
    arguments = ["--database", "/override.db", "sessions", "search", "needle"]
    json_result = runner.invoke(cli.app, [*arguments, "--format", "json"])
    jsonl_result = runner.invoke(cli.app, [*arguments, "--format", "jsonl"])
    expected = {
        "session": _EXPECTED_SESSION,
        "source": "text",
        "role": "assistant",
        "snippet": "A matching\nsnippet",
        "matchCount": 2,
    }

    assert json_result.exit_code == 0
    assert json.loads(json_result.output) == [expected]
    assert jsonl_result.exit_code == 0
    assert json.loads(jsonl_result.output) == expected
    assert search_api["explicit_path"] == Path("/override.db")


@pytest.fixture
def semantic_api(
    search_api: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    captured: dict[str, object] = {"search_api": search_api}
    session = search_api["match"].session
    match = SemanticSearchMatch(
        session=session,
        source=SearchSource.TEXT,
        role="assistant",
        snippet="A semantic match",
        score=0.98765,
        match_count=2,
    )

    def create() -> object:
        captured["provider_created"] = True
        return object()

    class Service:
        def __init__(
            self, database_path: Path, provider: object, index: Path | None = None
        ) -> None:
            captured["database_path"] = database_path
            captured["provider"] = provider
            captured["index"] = index

        def search(
            self, text: str, **options: object
        ) -> tuple[SemanticSearchMatch, ...]:
            captured["text"] = text
            captured["search_options"] = options
            return (match,)

    monkeypatch.setattr(
        search_sessions, "_semantic_dependencies", lambda: (create, Service)
    )
    captured["match"] = match
    return captured


def test_semantic_search_wires_defaults_filters_and_roles(
    semantic_api: dict[str, object],
) -> None:
    result = runner.invoke(
        cli.app,
        [
            "sessions",
            "search",
            "meaning",
            "--semantic",
            "--repo",
            "one",
            "--agent",
            "build",
            "--provider",
            "openai",
            "--model",
            "model-x",
            "--include-children",
            "--include-archived",
            "--role",
            "user",
            "--in",
            "title",
        ],
    )

    assert result.exit_code == 0
    assert semantic_api["provider_created"] is True
    options = semantic_api["search_options"]
    assert options["limit"] is None
    assert options["rerank"] is True
    assert options["sources"] == frozenset({SearchSource.TITLE})
    assert options["roles"] == frozenset({"user"})
    filters = options["filters"]
    assert filters.repository_project_ids == frozenset({"id:one"})
    assert filters.agents == frozenset({"build"})
    assert filters.providers == frozenset({"openai"})
    assert filters.models == frozenset({"model-x"})
    assert filters.include_children is True
    assert filters.archive_mode.value == "all"
    assert "Score" in result.output
    assert "0.988" in result.output


def test_semantic_search_uses_default_sidecar_and_explicit_limit(
    semantic_api: dict[str, object],
) -> None:
    result = runner.invoke(
        cli.app,
        [
            "sessions",
            "search",
            "meaning",
            "--semantic",
            "-n",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert semantic_api["index"] is None
    assert semantic_api["search_options"]["limit"] == 0


def test_no_rerank_is_passed_through_and_requires_semantic(
    semantic_api: dict[str, object],
) -> None:
    result = runner.invoke(
        cli.app, ["sessions", "search", "meaning", "--semantic", "--no-rerank"]
    )
    assert result.exit_code == 0, result.output
    assert semantic_api["search_options"]["rerank"] is False

    literal = runner.invoke(cli.app, ["sessions", "search", "meaning", "--no-rerank"])
    assert literal.exit_code == 2
    assert "--no-rerank requires --semantic" in literal.output


def test_semantic_json_output(semantic_api: dict[str, object]) -> None:
    result = runner.invoke(
        cli.app, ["sessions", "search", "meaning", "--semantic", "--format", "json"]
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == [
        {
            "session": _EXPECTED_SESSION,
            "source": "text",
            "role": "assistant",
            "snippet": "A semantic match",
            "score": 0.98765,
            "matchCount": 2,
        }
    ]


@pytest.mark.parametrize(
    "option",
    [
        "--include-reasoning",
        "--include-tool-inputs",
        "--include-tool-outputs",
        "--case-sensitive",
    ],
)
def test_semantic_rejects_unsupported_options(option: str) -> None:
    result = runner.invoke(
        cli.app, ["sessions", "search", "meaning", "--semantic", option]
    )

    assert result.exit_code == 2
    assert option in result.output
    assert "--semantic" in result.output
    assert "Traceback" not in result.output


def test_semantic_provider_errors_are_clean(
    search_api: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    def create() -> object:
        raise EmbeddingError("provider unavailable")

    monkeypatch.setattr(
        search_sessions, "_semantic_dependencies", lambda: (create, object)
    )
    result = runner.invoke(cli.app, ["sessions", "search", "meaning", "--semantic"])

    assert result.exit_code == 1
    assert "Error: provider unavailable" in result.output
    assert "Traceback" not in result.output
    assert "path" not in search_api


def test_search_help_omits_removed_semantic_configuration_options() -> None:
    result = runner.invoke(cli.app, ["sessions", "search", "--help"])

    assert result.exit_code == 0
    for option in (
        "--embedding-provider",
        "--embedding-model",
        "--embedding-dimension",
        "--semantic-index",
    ):
        assert option not in result.output
