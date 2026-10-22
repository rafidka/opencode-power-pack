"""The ``sessions search`` command: semantic search over session content."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console

from ...database import resolve_database_path
from ...errors import EmbeddingError, OpencodePowerPackError
from ...models import OutputFormat, SearchSource
from ...output import emit_json_array, emit_jsonl
from ._index_progress import SearchSyncNotice
from ._search_common import (
    Role,
    Source,
    emit_matches,
    selected_roles,
    session_filter,
    validate_query,
)

if TYPE_CHECKING:
    from ...semantic import CohereEmbedProvider, SemanticService


def _semantic_dependencies() -> tuple[type[CohereEmbedProvider], type[SemanticService]]:
    from ...semantic import CohereEmbedProvider, SemanticService

    return CohereEmbedProvider, SemanticService


def _stderr_console() -> Console:
    return Console(stderr=True)


def _search(
    context: typer.Context,
    query: Annotated[
        str,
        typer.Argument(
            help="A natural-language description of what to find.",
            callback=validate_query,
        ),
    ],
    repositories: Annotated[
        list[str] | None,
        typer.Option("-r", "--repo", help="Restrict results to a repository."),
    ] = None,
    cwd: Annotated[
        bool,
        typer.Option("--cwd", help="Restrict results to the current repository."),
    ] = False,
    max_count: Annotated[
        int | None,
        typer.Option(
            "-n", "--max-count", min=0, help="Return at most N sessions (default 5)."
        ),
    ] = None,
    include_children: Annotated[
        bool, typer.Option("--include-children", help="Include child sessions.")
    ] = False,
    include_archived: Annotated[
        bool, typer.Option("--include-archived", help="Include archived sessions.")
    ] = False,
    archived_only: Annotated[
        bool, typer.Option("--archived-only", help="Search archived sessions only.")
    ] = False,
    agents: Annotated[list[str] | None, typer.Option("--agent")] = None,
    providers: Annotated[list[str] | None, typer.Option("--provider")] = None,
    models: Annotated[list[str] | None, typer.Option("--model")] = None,
    since: Annotated[str | None, typer.Option("--since")] = None,
    until: Annotated[str | None, typer.Option("--until")] = None,
    roles: Annotated[list[Role] | None, typer.Option("--role")] = None,
    sources: Annotated[
        list[Source] | None,
        typer.Option("--in", help="Search session titles or message text."),
    ] = None,
    no_rerank: Annotated[
        bool,
        typer.Option("--no-rerank", help="Rank by embedding similarity alone."),
    ] = False,
    snippet_length: Annotated[
        int, typer.Option("--snippet-length", min=1, help="Maximum snippet length.")
    ] = 200,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", help="Output format.")
    ] = OutputFormat.TABLE,
) -> None:
    """Find sessions by meaning, using Cohere Embed and Rerank.

    New or changed sessions are embedded before searching; use
    ``op2 sessions grep`` for literal text matching instead.
    """
    from ...cli import get_app_context

    notice: SearchSyncNotice | None = None
    try:
        database_path = resolve_database_path(get_app_context(context).database)
        filters = session_filter(
            database_path,
            repositories=repositories or (),
            cwd=cwd,
            include_children=include_children,
            include_archived=include_archived,
            archived_only=archived_only,
            agents=agents or (),
            providers=providers or (),
            models=models or (),
            since=since,
            until=until,
        )
        provider_type, service_type = _semantic_dependencies()
        try:
            provider = provider_type()
        except EmbeddingError as error:
            raise EmbeddingError(
                f"{error}; `op2 sessions grep` searches literal text without it"
            ) from error
        console = _stderr_console()
        notice = SearchSyncNotice(
            console,
            show_progress=output_format is OutputFormat.TABLE and console.is_terminal,
        )
        matches = service_type(database_path, provider).search(
            query,
            filters=filters,
            sources=frozenset(SearchSource(source) for source in (sources or Source)),
            roles=selected_roles(roles),
            snippet_length=snippet_length,
            limit=max_count,
            rerank=not no_rerank,
            on_progress=notice,
        )
    except OpencodePowerPackError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(error.exit_code) from None
    finally:
        if notice is not None:
            notice.stop()

    if output_format is OutputFormat.JSON:
        emit_json_array(matches)
        return
    if output_format is OutputFormat.JSONL:
        emit_jsonl(matches)
        return
    if not matches:
        typer.echo("No matching sessions.")
        return
    emit_matches(
        "Score",
        (
            (match.session, match.source.value, match.snippet, f"{match.score:.3f}")
            for match in matches
        ),
    )


def register(app: typer.Typer) -> None:
    """Register the search command with a sessions application."""
    if not any(command.name == "search" for command in app.registered_commands):
        app.command("search")(_search)
