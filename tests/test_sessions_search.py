import json
from pathlib import Path

import pytest
from sessions_contract import EXPECTED_SESSION
from typer.testing import CliRunner

from opencode_power_pack import cli
from opencode_power_pack.commands.sessions import search_sessions
from opencode_power_pack.errors import EmbeddingError
from opencode_power_pack.models import SearchSource
from opencode_power_pack.semantic import SemanticSearchMatch

runner = CliRunner()


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
            "-n",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert semantic_api["index"] is None
    assert semantic_api["search_options"]["limit"] == 0


def test_no_rerank_is_passed_through(semantic_api: dict[str, object]) -> None:
    result = runner.invoke(cli.app, ["sessions", "search", "meaning", "--no-rerank"])
    assert result.exit_code == 0, result.output
    assert semantic_api["search_options"]["rerank"] is False


def test_semantic_json_output(semantic_api: dict[str, object]) -> None:
    result = runner.invoke(
        cli.app, ["sessions", "search", "meaning", "--format", "json"]
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == [
        {
            "session": EXPECTED_SESSION,
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
        "--semantic",
    ],
)
def test_search_has_no_literal_only_options(option: str) -> None:
    result = runner.invoke(cli.app, ["sessions", "search", "meaning", option])

    assert result.exit_code == 2
    assert "No such option" in result.output
    assert "Traceback" not in result.output


def test_semantic_provider_errors_are_clean(
    search_api: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    def create() -> object:
        raise EmbeddingError("provider unavailable")

    monkeypatch.setattr(
        search_sessions, "_semantic_dependencies", lambda: (create, object)
    )
    result = runner.invoke(cli.app, ["sessions", "search", "meaning"])

    assert result.exit_code == 1
    assert "Error: provider unavailable" in result.output
    assert "`op2 sessions grep`" in result.output
    assert "Traceback" not in result.output


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


def _planned(sessions: int, chunks: int) -> object:
    from opencode_power_pack.semantic import SemanticSyncPlanned

    return SemanticSyncPlanned(
        changed_session_count=sessions,
        deleted_session_count=0,
        document_count=chunks,
        batch_count=1 if chunks else 0,
    )


@pytest.mark.parametrize(
    ("sessions", "chunks", "notice"),
    [
        (2, 5, "Indexing 2 new or changed sessions (5 chunks to embed)"),
        (1, 1, "Indexing 1 new or changed session (1 chunk to embed)"),
        (3, 0, None),
    ],
)
def test_search_announces_sessions_it_must_index_on_stderr(
    semantic_api: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    sessions: int,
    chunks: int,
    notice: str | None,
) -> None:
    service_type = search_sessions._semantic_dependencies()[1]

    class AnnouncingService(service_type):  # type: ignore[misc, valid-type]
        def search(self, text: str, **options: object) -> object:
            callback = options["on_progress"]
            assert callable(callback)
            callback(_planned(sessions, chunks))
            return super().search(text, **options)

    create = search_sessions._semantic_dependencies()[0]
    monkeypatch.setattr(
        search_sessions, "_semantic_dependencies", lambda: (create, AnnouncingService)
    )

    result = runner.invoke(
        cli.app, ["sessions", "search", "meaning", "--format", "json"]
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)[0]["score"] == 0.98765
    if notice is None:
        assert result.stderr == ""
    else:
        assert notice in result.stderr
        assert "before searching" in result.stderr
