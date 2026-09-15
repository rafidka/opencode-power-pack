"""Shared plumbing for the mutating session commands.

``sessions delete`` and ``sessions archive`` differ only in the request they send
to OpenCode and in whether they confirm first, so selector collection, ordering,
reporting, and exit status live here.
"""

import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import IO

import typer
from pydantic import Field

from ...database import OpenCodeDatabase, resolve_database_path
from ...errors import OpencodePowerPackError, OpencodeServerError, UsageError
from ...models import OutputFormat, RecordModel, Session
from ...output import (
    emit_json_array,
    emit_jsonl,
    emit_table,
    is_stdout_terminal,
)
from ...server import OpencodeServer, opencode_server
from ...session_selectors import (
    SessionResolver,
    selectors_from_arguments,
    selectors_from_stdin,
)


class MutationStatus(StrEnum):
    """Whether OpenCode applied the request for one session."""

    APPLIED = "applied"
    FAILED = "failed"


class MutationVerb(StrEnum):
    """A mutation a session command can ask OpenCode to perform."""

    DELETE = "delete"
    ARCHIVE = "archive"

    @property
    def past_tense(self) -> str:
        """Return the past tense used when reporting what was done."""
        return _PAST_TENSE[self]


_PAST_TENSE = {
    MutationVerb.DELETE: "deleted",
    MutationVerb.ARCHIVE: "archived",
}


class MutationOutcome(RecordModel):
    """The result of mutating one session.

    The fields are the reported record, so no separate output conversion is
    needed: ``session_id`` is published as ``id`` alongside the session details
    worth showing next to it.
    """

    session_id: str = Field(serialization_alias="id")
    title: str | None
    repository: str | None
    status: MutationStatus
    error: str | None = None

    @classmethod
    def of(
        cls, session: Session, status: MutationStatus, error: str | None = None
    ) -> MutationOutcome:
        """Return the outcome of mutating ``session``."""
        return cls(
            session_id=session.id,
            title=session.title,
            repository=session.repository,
            status=status,
            error=error,
        )


def run_mutation(
    context: typer.Context,
    *,
    sessions: Sequence[str],
    output_format: OutputFormat,
    verb: MutationVerb,
    apply: Callable[[OpencodeServer, Session], None],
    confirm: bool,
    stdin: IO[str] | None = None,
) -> None:
    """Resolve selectors, apply ``apply`` to each session, and report outcomes."""
    from ...cli import get_app_context

    stream = sys.stdin if stdin is None else stdin
    database_path = resolve_database_path(get_app_context(context).database)
    with OpenCodeDatabase(database_path) as database:
        resolver = SessionResolver(database.all_sessions())

    selectors = list(selectors_from_arguments(sessions))
    piped = False
    if not selectors:
        # Only consume stdin when no session was named, so that an argument form
        # never blocks on a stdin that stays open without ever reaching EOF.
        piped = not stream.isatty()
        if piped:
            selectors.extend(selectors_from_stdin(stream, resolver))
    if not selectors:
        if piped:
            typer.echo(f"No sessions to {verb}.", err=True)
            return
        raise UsageError(
            f"provide one or more session IDs as arguments or pipe "
            f"`sessions list` output into `sessions {verb}`"
        )

    targets = resolver.resolve_all(selectors)
    if confirm:
        _confirm(targets, verb=verb, piped=piped)

    outcomes = _apply_all(database_path, targets, apply)
    _emit(outcomes, output_format=output_format, verb=verb)
    if any(outcome.status is MutationStatus.FAILED for outcome in outcomes):
        raise typer.Exit(1)


def _apply_all(
    database_path: Path,
    targets: Sequence[Session],
    apply: Callable[[OpencodeServer, Session], None],
) -> tuple[MutationOutcome, ...]:
    outcomes: list[MutationOutcome] = []
    with opencode_server(database_path) as server:
        for session in targets:
            try:
                apply(server, session)
            except OpencodeServerError as error:
                outcomes.append(
                    MutationOutcome.of(session, MutationStatus.FAILED, str(error))
                )
            else:
                outcomes.append(MutationOutcome.of(session, MutationStatus.APPLIED))
    return tuple(outcomes)


def _confirm(targets: Sequence[Session], *, verb: MutationVerb, piped: bool) -> None:
    count = len(targets)
    noun = "session" if count == 1 else "sessions"
    if piped or not is_stdout_terminal():
        raise UsageError(
            f"refusing to {verb} {count} {noun} without confirmation; pass --yes"
        )
    for session in targets:
        typer.echo(f"  {session.id}  {_display(session.title)}", err=True)
    if not typer.confirm(f"{verb.capitalize()} {count} {noun}?"):
        raise typer.Abort()


def _emit(
    outcomes: Sequence[MutationOutcome],
    *,
    output_format: OutputFormat,
    verb: MutationVerb,
) -> None:
    if output_format is OutputFormat.JSON:
        emit_json_array(outcomes)
        return
    if output_format is OutputFormat.JSONL:
        emit_jsonl(outcomes)
        return
    applied = [
        outcome for outcome in outcomes if outcome.status is MutationStatus.APPLIED
    ]
    failed = [
        outcome for outcome in outcomes if outcome.status is MutationStatus.FAILED
    ]
    if is_stdout_terminal():
        emit_table(
            ("Session ID", "Title", "Status"),
            (
                (
                    outcome.session_id,
                    _display(outcome.title),
                    outcome.error or outcome.status.value,
                )
                for outcome in outcomes
            ),
        )
    else:
        for outcome in applied:
            typer.echo(outcome.session_id)
    noun = "session" if len(applied) == 1 else "sessions"
    typer.echo(f"{verb.past_tense.capitalize()} {len(applied)} {noun}.", err=True)
    for outcome in failed:
        typer.echo(f"Error: {outcome.session_id}: {outcome.error}", err=True)


def _display(value: str | None) -> str:
    return "-" if value is None or not value.strip() else value.strip()


def now_milliseconds() -> int:
    """Return the current UTC time in milliseconds since the epoch."""
    return int(datetime.now(UTC).timestamp() * 1_000)


def handle_expected_errors(run: Callable[[], None]) -> None:
    """Convert expected errors into a stderr message and an exit code."""
    try:
        run()
    except OpencodePowerPackError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(error.exit_code) from None
