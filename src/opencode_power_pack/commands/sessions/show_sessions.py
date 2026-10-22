"""The ``sessions show`` command."""

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Annotated

import typer

from ...database import OpenCodeDatabase, resolve_database_path
from ...errors import RuntimeCommandError, UsageError
from ...models import TranscriptQuery
from ...session_selectors import (
    SessionResolver,
    selectors_from_arguments,
    selectors_from_stdin,
)
from ...transcript import TranscriptOptions, render_transcripts
from ._mutations import handle_expected_errors


def register(app: typer.Typer) -> None:
    """Register the session transcript command."""
    if not any(command.name == "show" for command in app.registered_commands):
        app.command("show")(show_sessions)


def show_sessions(
    context: typer.Context,
    sessions: Annotated[
        list[str] | None,
        typer.Argument(
            metavar="[SESSION]...",
            help="Session IDs, or any prefix identifying exactly one session.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            metavar="PATH",
            help="Write the transcript to PATH instead of standard output.",
        ),
    ] = None,
    include_reasoning: Annotated[
        bool,
        typer.Option("--include-reasoning", help="Include assistant reasoning blocks."),
    ] = False,
    include_tool_inputs: Annotated[
        bool,
        typer.Option(
            "--include-tool-inputs", help="Include the full input of every tool call."
        ),
    ] = False,
    include_tool_outputs: Annotated[
        bool,
        typer.Option(
            "--include-tool-outputs",
            help="Include the full output of every tool call.",
        ),
    ] = False,
    include_children: Annotated[
        bool,
        typer.Option(
            "--include-children", help="Inline sub-agent session transcripts."
        ),
    ] = False,
    max_tool_chars: Annotated[
        int,
        typer.Option(
            "--max-tool-chars",
            min=0,
            help=(
                "Truncate each included tool input or output to this many "
                "characters; 0 means unlimited."
            ),
        ),
    ] = 2000,
) -> None:
    """Render sessions as a tagged transcript, suitable for referencing in
    another conversation or feeding to a summarizer.

    Conversation content is reproduced verbatim; the transcript's own
    structure is XML tags, so a heading or rule inside the conversation can
    never be confused with the transcript's own structure. Session IDs may
    also be piped in, for example from ``op2 sessions list``. User and
    assistant text plus one line per tool call are included by default,
    since tool inputs and outputs dominate transcript size.
    """
    handle_expected_errors(
        lambda: _run_show_sessions(
            context,
            sessions=sessions or (),
            output=output,
            include_reasoning=include_reasoning,
            include_tool_inputs=include_tool_inputs,
            include_tool_outputs=include_tool_outputs,
            include_children=include_children,
            max_tool_chars=max_tool_chars,
        )
    )


def _run_show_sessions(
    context: typer.Context,
    *,
    sessions: Sequence[str],
    output: Path | None,
    include_reasoning: bool,
    include_tool_inputs: bool,
    include_tool_outputs: bool,
    include_children: bool,
    max_tool_chars: int,
    stdin: IO[str] | None = None,
) -> None:
    """Resolve selectors, fetch transcripts, and write the rendered document."""
    from ...cli import get_app_context

    stream = sys.stdin if stdin is None else stdin
    database_path = resolve_database_path(get_app_context(context).database)
    with OpenCodeDatabase(database_path) as database:
        resolver = SessionResolver(database.all_sessions())

        selectors = list(selectors_from_arguments(sessions))
        piped = False
        if not selectors:
            # Only consume stdin when no session was named, so that an argument
            # form never blocks on a stdin that stays open without ever
            # reaching EOF.
            piped = not stream.isatty()
            if piped:
                selectors.extend(selectors_from_stdin(stream, resolver))
        if not selectors:
            if piped:
                typer.echo("No sessions to show.", err=True)
                return
            raise UsageError(
                "provide one or more session IDs as arguments or pipe "
                "`sessions list` output into `sessions show`"
            )

        targets = resolver.resolve_all(selectors)
        query = TranscriptQuery(
            session_ids=tuple(session.id for session in targets),
            include_children=include_children,
        )
        transcripts = database.fetch_transcripts(query)

    document = render_transcripts(
        transcripts,
        TranscriptOptions(
            include_reasoning=include_reasoning,
            include_tool_inputs=include_tool_inputs,
            include_tool_outputs=include_tool_outputs,
            max_tool_chars=max_tool_chars,
        ),
    )

    if output is None:
        sys.stdout.write(document)
        return

    try:
        output.write_text(document, encoding="utf-8")
    except OSError as error:
        raise RuntimeCommandError(f"could not write {output}: {error}") from None

    noun = "session" if len(targets) == 1 else "sessions"
    typer.echo(f"Wrote {len(targets)} {noun} to {output}.", err=True)
