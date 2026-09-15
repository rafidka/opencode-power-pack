"""The ``sessions archive`` command."""

from typing import Annotated

import typer

from opencode_power_pack.models import OutputFormat, Session
from opencode_power_pack.server import OpencodeServer

from ._mutations import (
    MutationVerb,
    handle_expected_errors,
    now_milliseconds,
    run_mutation,
)


def register(app: typer.Typer) -> None:
    """Register the session archival command."""
    if not any(command.name == "archive" for command in app.registered_commands):
        app.command("archive")(archive_sessions)


def archive_sessions(
    context: typer.Context,
    sessions: Annotated[
        list[str] | None,
        typer.Argument(
            metavar="[SESSION]...",
            help="Session IDs, or any prefix identifying exactly one session.",
        ),
    ] = None,
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", help="Output format: table, json, or jsonl."),
    ] = OutputFormat.TABLE,
) -> None:
    """Archive sessions so that listings and searches skip them by default.

    Archiving is performed by OpenCode itself and keeps all session content.
    Session IDs may also be piped in, for example from ``op2 sessions list``.
    """
    archived_at = now_milliseconds()

    def apply(server: OpencodeServer, session: Session) -> None:
        server.archive_session(session.id, archived_at)

    handle_expected_errors(
        lambda: run_mutation(
            context,
            sessions=sessions or (),
            output_format=output_format,
            verb=MutationVerb.ARCHIVE,
            apply=apply,
            confirm=False,
        )
    )
