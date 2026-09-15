"""The ``sessions delete`` command."""

from typing import Annotated

import typer

from opencode_power_pack.models import OutputFormat, Session
from opencode_power_pack.server import OpencodeServer

from ._mutations import MutationVerb, handle_expected_errors, run_mutation


def register(app: typer.Typer) -> None:
    """Register the session deletion command."""
    if not any(command.name == "delete" for command in app.registered_commands):
        app.command("delete")(delete_sessions)


def delete_sessions(
    context: typer.Context,
    sessions: Annotated[
        list[str] | None,
        typer.Argument(
            metavar="[SESSION]...",
            help="Session IDs, or any prefix identifying exactly one session.",
        ),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Delete without asking for confirmation."),
    ] = False,
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", help="Output format: table, json, or jsonl."),
    ] = OutputFormat.TABLE,
) -> None:
    """Delete sessions, including their child sessions and content.

    Deletion is performed by OpenCode itself and cannot be undone. Session IDs may
    also be piped in, for example from ``op2 sessions list``.
    """

    def apply(server: OpencodeServer, session: Session) -> None:
        server.delete_session(session.id)

    handle_expected_errors(
        lambda: run_mutation(
            context,
            sessions=sessions or (),
            output_format=output_format,
            verb=MutationVerb.DELETE,
            apply=apply,
            confirm=not yes,
        )
    )
