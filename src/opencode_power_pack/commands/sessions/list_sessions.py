"""The ``sessions list`` command."""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from opencode_power_pack.database import OpenCodeDatabase, resolve_database_path
from opencode_power_pack.errors import OpencodePowerPackError, UsageError
from opencode_power_pack.models import (
    ArchiveMode,
    ListSessionsQuery,
    ListSort,
    OutputFormat,
    Session,
    SessionFilter,
    SortDirection,
)
from opencode_power_pack.output import emit_json_array, emit_jsonl, emit_table
from opencode_power_pack.repositories import RepositoryResolver
from opencode_power_pack.timeparse import parse_time


def register(app: typer.Typer) -> None:
    """Register the session listing command."""
    if not any(command.name == "list" for command in app.registered_commands):
        app.command("list")(list_sessions)


def list_sessions(
    context: typer.Context,
    repositories: Annotated[
        list[str] | None,
        typer.Option("--repo", "-r", metavar="TEXT", help="Filter by repository."),
    ] = None,
    cwd: Annotated[
        bool,
        typer.Option("--cwd", help="Filter by the current working directory."),
    ] = False,
    max_count: Annotated[
        int | None,
        typer.Option("--max-count", "-n", min=0, help="Maximum sessions to return."),
    ] = None,
    include_children: Annotated[
        bool,
        typer.Option("--include-children", help="Include child sessions."),
    ] = False,
    include_archived: Annotated[
        bool,
        typer.Option(
            "--include-archived", help="Include active and archived sessions."
        ),
    ] = False,
    archived_only: Annotated[
        bool,
        typer.Option("--archived-only", help="Include only archived sessions."),
    ] = False,
    agents: Annotated[
        list[str] | None,
        typer.Option("--agent", metavar="TEXT", help="Filter by agent."),
    ] = None,
    providers: Annotated[
        list[str] | None,
        typer.Option("--provider", metavar="TEXT", help="Filter by provider."),
    ] = None,
    models: Annotated[
        list[str] | None,
        typer.Option("--model", metavar="TEXT", help="Filter by model."),
    ] = None,
    since: Annotated[
        str | None,
        typer.Option(
            "--since", metavar="TIME", help="Only sessions updated at or after TIME."
        ),
    ] = None,
    until: Annotated[
        str | None,
        typer.Option(
            "--until", metavar="TIME", help="Only sessions updated before TIME."
        ),
    ] = None,
    sort: Annotated[
        ListSort,
        typer.Option(
            "--sort", help="Order by updated, created, cost, tokens, or title."
        ),
    ] = ListSort.UPDATED,
    reverse: Annotated[
        bool,
        typer.Option("--reverse", help="Reverse the default ordering direction."),
    ] = False,
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", help="Output format: table, json, or jsonl."),
    ] = OutputFormat.TABLE,
) -> None:
    """List sessions stored in the OpenCode database."""

    try:
        _run_list_sessions(
            context,
            repositories=repositories or (),
            cwd=Path.cwd() if cwd else None,
            max_count=max_count,
            include_children=include_children,
            include_archived=include_archived,
            archived_only=archived_only,
            agents=agents or (),
            providers=providers or (),
            models=models or (),
            since=since,
            until=until,
            sort=sort,
            reverse=reverse,
            output_format=output_format,
        )
    except OpencodePowerPackError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(error.exit_code) from None


def _run_list_sessions(
    context: typer.Context,
    *,
    repositories: Sequence[str],
    cwd: Path | None,
    max_count: int | None,
    include_children: bool,
    include_archived: bool,
    archived_only: bool,
    agents: Sequence[str],
    providers: Sequence[str],
    models: Sequence[str],
    since: str | None,
    until: str | None,
    sort: ListSort,
    reverse: bool,
    output_format: OutputFormat,
) -> None:
    if repositories and cwd is not None:
        raise UsageError("--repo cannot be used with --cwd")
    if include_archived and archived_only:
        raise UsageError("--include-archived cannot be used with --archived-only")

    since_timestamp = None if since is None else parse_time(since)
    until_timestamp = None if until is None else parse_time(until)
    if (
        since_timestamp is not None
        and until_timestamp is not None
        and since_timestamp >= until_timestamp
    ):
        raise UsageError("--since must be earlier than --until")

    if archived_only:
        archive_mode = ArchiveMode.ARCHIVED
    elif include_archived:
        archive_mode = ArchiveMode.ALL
    else:
        archive_mode = ArchiveMode.ACTIVE

    from opencode_power_pack.cli import get_app_context

    app_context = get_app_context(context)
    database_path = resolve_database_path(app_context.database)
    with OpenCodeDatabase(database_path) as database:
        project_ids = _resolve_project_ids(database, repositories, cwd)
        query = ListSessionsQuery(
            filters=SessionFilter(
                repository_project_ids=project_ids,
                include_children=include_children,
                archive_mode=archive_mode,
                agents=frozenset(agents),
                providers=frozenset(providers),
                models=frozenset(models),
                since=since_timestamp,
                until=until_timestamp,
            ),
            sort=sort,
            direction=SortDirection.ASC if reverse else SortDirection.DESC,
            limit=max_count,
        )
        sessions = database.list_sessions(query)

    _emit_sessions(sessions, output_format, include_children)


def _resolve_project_ids(
    database: OpenCodeDatabase,
    repositories: Sequence[str],
    cwd: Path | None,
) -> frozenset[str]:
    if not repositories and cwd is None:
        return frozenset()
    resolver = RepositoryResolver(database.list_repositories())
    if cwd is not None:
        return frozenset({resolver.resolve_cwd(cwd).project_id})
    return resolver.resolve_many(repositories)


def _emit_sessions(
    sessions: Sequence[Session],
    output_format: OutputFormat,
    include_children: bool,
) -> None:
    if output_format is OutputFormat.JSON:
        emit_json_array(sessions)
        return
    if output_format is OutputFormat.JSONL:
        emit_jsonl(sessions)
        return
    if not sessions:
        typer.echo("No sessions found.")
        return

    columns = ["Updated", "Repository", "Title", "Agent", "Model", "Session ID"]
    if include_children:
        columns.append("Parent")
    rows: list[list[str]] = []
    for session in sessions:
        row = [
            _format_timestamp(session.updated),
            _display(session.repository),
            _display(session.title),
            _display(session.agent),
            _display(session.model),
            session.id,
        ]
        if include_children:
            row.append(_display(session.parent_id))
        rows.append(row)
    emit_table(columns, rows)


def _format_timestamp(timestamp: int | None) -> str:
    if timestamp is None:
        return "-"
    return (
        datetime.fromtimestamp(timestamp / 1_000, UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _display(value: str | None) -> str:
    return "-" if value is None or value == "" else value
