"""The ``sessions search`` command."""

import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from ...database import OpenCodeDatabase, resolve_database_path
from ...errors import OpencodePowerPackError, TimeParseError
from ...models import (
    ArchiveMode,
    OutputFormat,
    SearchSessionsQuery,
    SearchSource,
    SessionFilter,
)
from ...output import emit_json_array, emit_jsonl, emit_table
from ...repositories import RepositoryResolver
from ...timeparse import parse_time

_WHITESPACE = re.compile(r"\s+")


class _Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class _OrdinarySource(StrEnum):
    TITLE = "title"
    TEXT = "text"


def _parse_time_option(value: str | None, option_name: str) -> int | None:
    if value is None:
        return None
    try:
        return parse_time(value)
    except TimeParseError as error:
        raise typer.BadParameter(str(error), param_hint=option_name) from error


def _display(value: object | None) -> str:
    if value is None:
        return "—"
    text = _WHITESPACE.sub(" ", str(value)).strip()
    return text or "—"


def _display_directory(value: str | None) -> str:
    if value is None:
        return "—"
    path = Path(value)
    try:
        return f"~/{path.relative_to(Path.home())}"
    except ValueError:
        return _display(value)


def _display_repository(repository: str | None, directory: str | None) -> str:
    return f"{_display(repository)}\n{_display_directory(directory)}"


def _display_updated(value: int | None) -> str:
    if value is None:
        return "—"
    try:
        return datetime.fromtimestamp(value / 1_000, UTC).strftime("%Y-%m-%d %H:%M")
    except OverflowError, OSError, ValueError:
        return _display(value)


def _validate_query(value: str) -> str:
    if not value.strip():
        raise typer.BadParameter("must not be empty")
    return value


def _search(
    context: typer.Context,
    query: Annotated[
        str,
        typer.Argument(help="Text to search for.", callback=_validate_query),
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
        typer.Option("-n", "--max-count", min=0, help="Return at most N matches."),
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
    roles: Annotated[
        list[_Role] | None,
        typer.Option("--role"),
    ] = None,
    sources: Annotated[
        list[_OrdinarySource] | None,
        typer.Option("--in", help="Search title, text, or an opted-in content source."),
    ] = None,
    include_todos: Annotated[bool, typer.Option("--include-todos")] = False,
    include_reasoning: Annotated[bool, typer.Option("--include-reasoning")] = False,
    include_tool_inputs: Annotated[bool, typer.Option("--include-tool-inputs")] = False,
    include_tool_outputs: Annotated[
        bool, typer.Option("--include-tool-outputs")
    ] = False,
    case_sensitive: Annotated[bool, typer.Option("--case-sensitive")] = False,
    snippet_length: Annotated[
        int, typer.Option("--snippet-length", min=1, help="Maximum snippet length.")
    ] = 200,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", help="Output format.")
    ] = OutputFormat.TABLE,
) -> None:
    if repositories and cwd:
        raise typer.BadParameter("--repo cannot be used with --cwd")
    if include_archived and archived_only:
        raise typer.BadParameter(
            "--include-archived cannot be used with --archived-only"
        )

    since_ms = _parse_time_option(since, "--since")
    until_ms = _parse_time_option(until, "--until")
    if since_ms is not None and until_ms is not None and since_ms >= until_ms:
        raise typer.BadParameter("--since must be earlier than --until")

    selected_sources = {SearchSource(source) for source in (sources or _OrdinarySource)}
    if include_todos:
        selected_sources.add(SearchSource.TODO)
    if include_reasoning:
        selected_sources.add(SearchSource.REASONING)
    if include_tool_inputs:
        selected_sources.add(SearchSource.TOOL_INPUT)
    if include_tool_outputs:
        selected_sources.add(SearchSource.TOOL_OUTPUT)

    if archived_only:
        archive_mode = ArchiveMode.ARCHIVED
    elif include_archived:
        archive_mode = ArchiveMode.ALL
    else:
        archive_mode = ArchiveMode.ACTIVE

    try:
        from ...cli import get_app_context

        app_context = get_app_context(context)
        database_path = resolve_database_path(app_context.database)
        with OpenCodeDatabase(database_path) as database:
            resolver = RepositoryResolver(database.list_repositories())
            if cwd:
                repository_ids = frozenset(
                    {resolver.resolve_cwd(Path.cwd()).project_id}
                )
            else:
                repository_ids = resolver.resolve_many(repositories or ())
            search_query = SearchSessionsQuery(
                text=query,
                filters=SessionFilter(
                    repository_project_ids=repository_ids,
                    include_children=include_children,
                    archive_mode=archive_mode,
                    agents=frozenset(agents or ()),
                    providers=frozenset(providers or ()),
                    models=frozenset(models or ()),
                    since=since_ms,
                    until=until_ms,
                ),
                sources=frozenset(selected_sources),
                roles=frozenset(roles or (_Role.USER, _Role.ASSISTANT)),
                case_sensitive=case_sensitive,
                snippet_length=snippet_length,
                limit=max_count,
            )
            matches = database.search_sessions(search_query)
    except OpencodePowerPackError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(error.exit_code) from None

    if output_format is OutputFormat.JSON:
        emit_json_array(match.as_dict() for match in matches)
        return
    if output_format is OutputFormat.JSONL:
        emit_jsonl(match.as_dict() for match in matches)
        return
    if not matches:
        typer.echo("No matching sessions.")
        return
    emit_table(
        (
            "Updated",
            "Repository",
            "Title",
            "Source",
            "Snippet",
            "Matches",
            "Session ID",
        ),
        (
            (
                _display_updated(match.session.updated),
                _display_repository(
                    match.session.repository, match.session.directory
                ),
                _display(match.session.title),
                _display(match.source.value),
                _display(match.snippet),
                match.match_count,
                _display(match.session.id),
            )
            for match in matches
        ),
    )


def register(app: typer.Typer) -> None:
    """Register the search command with a sessions application."""
    if not any(command.name == "search" for command in app.registered_commands):
        app.command("search")(_search)
