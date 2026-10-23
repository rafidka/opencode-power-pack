"""The ``sessions grep`` command: literal text search over session content."""

from typing import Annotated

import typer

from ...database import OpenCodeDatabase, resolve_database_path
from ...errors import OpencodePowerPackError
from ...models import OutputFormat, SearchSessionsQuery, SearchSource
from ...output import emit_json_array, emit_jsonl
from ._search_common import (
    Role,
    Source,
    emit_matches,
    selected_roles,
    session_filter,
    validate_query,
)


def _grep(
    context: typer.Context,
    query: Annotated[
        str,
        typer.Argument(help="Literal text to search for.", callback=validate_query),
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
    include_trivial: Annotated[
        bool,
        typer.Option("--include-trivial", help="Include trivial sessions."),
    ] = False,
    trivial_only: Annotated[
        bool, typer.Option("--trivial-only", help="Search trivial sessions only.")
    ] = False,
    excluded: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude",
            metavar="SESSION",
            help="Leave out SESSION, its parents, and its sub-agent sessions.",
        ),
    ] = None,
    agents: Annotated[list[str] | None, typer.Option("--agent")] = None,
    providers: Annotated[list[str] | None, typer.Option("--provider")] = None,
    models: Annotated[list[str] | None, typer.Option("--model")] = None,
    since: Annotated[str | None, typer.Option("--since")] = None,
    until: Annotated[str | None, typer.Option("--until")] = None,
    roles: Annotated[list[Role] | None, typer.Option("--role")] = None,
    sources: Annotated[
        list[Source] | None,
        typer.Option("--in", help="Search title, text, or an opted-in content source."),
    ] = None,
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
    """Find sessions containing literal text.

    Matching is case-insensitive by default and never sends content anywhere;
    use ``op2 sessions search`` to find sessions by meaning instead.
    """
    selected_sources = {SearchSource(source) for source in (sources or Source)}
    if include_reasoning:
        selected_sources.add(SearchSource.REASONING)
    if include_tool_inputs:
        selected_sources.add(SearchSource.TOOL_INPUT)
    if include_tool_outputs:
        selected_sources.add(SearchSource.TOOL_OUTPUT)

    try:
        from ...cli import get_app_context

        database_path = resolve_database_path(get_app_context(context).database)
        filters = session_filter(
            database_path,
            repositories=repositories or (),
            cwd=cwd,
            include_children=include_children,
            include_archived=include_archived,
            archived_only=archived_only,
            include_trivial=include_trivial,
            trivial_only=trivial_only,
            excluded=excluded or (),
            agents=agents or (),
            providers=providers or (),
            models=models or (),
            since=since,
            until=until,
        )
        search_query = SearchSessionsQuery(
            text=query,
            filters=filters,
            sources=frozenset(selected_sources),
            roles=selected_roles(roles),
            case_sensitive=case_sensitive,
            snippet_length=snippet_length,
            limit=max_count,
        )
        with OpenCodeDatabase(database_path) as database:
            matches = database.search_sessions(search_query)
    except OpencodePowerPackError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(error.exit_code) from None

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
        "Matches",
        (
            (match.session, match.source.value, match.snippet or "", match.match_count)
            for match in matches
        ),
    )


def register(app: typer.Typer) -> None:
    """Register the literal search command with a sessions application."""
    if not any(command.name == "grep" for command in app.registered_commands):
        app.command("grep")(_grep)
