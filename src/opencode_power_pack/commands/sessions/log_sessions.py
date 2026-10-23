"""The ``sessions log`` command."""

import os
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import typer

from ...database import OpenCodeDatabase, resolve_database_path
from ...errors import OpencodePowerPackError, TimeParseError
from ...models import ActivityQuery, SessionActivity
from ...output import emit_json_array, emit_jsonl
from ...timeparse import parse_time
from ._search_common import session_filter

_NO_TICKET = "No ticket"
_LOCALTIME = "/etc/localtime"


class LogFormat(StrEnum):
    MARKDOWN = "markdown"
    JSON = "json"
    JSONL = "jsonl"


def register(app: typer.Typer) -> None:
    """Register the session activity log command."""
    if not any(command.name == "log" for command in app.registered_commands):
        app.command("log")(log_sessions)


def log_sessions(
    context: typer.Context,
    since: Annotated[
        str,
        typer.Option("--since", metavar="TIME", help="Start of the window, inclusive."),
    ] = "7d",
    until: Annotated[
        str | None,
        typer.Option("--until", metavar="TIME", help="End of the window, exclusive."),
    ] = None,
    repositories: Annotated[
        list[str] | None,
        typer.Option("--repo", "-r", metavar="TEXT", help="Filter by repository."),
    ] = None,
    cwd: Annotated[
        bool, typer.Option("--cwd", help="Filter by the current working directory.")
    ] = False,
    excluded_repositories: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude-repo",
            metavar="NAME",
            help="Leave out sessions of the repository named NAME. Repeatable.",
        ),
    ] = None,
    time_zone: Annotated[
        str | None,
        typer.Option(
            "--tz",
            metavar="ZONE",
            help="IANA time zone for active days. Defaults to the system zone.",
        ),
    ] = None,
    ticket_prefixes: Annotated[
        list[str] | None,
        typer.Option(
            "--ticket-prefix",
            metavar="KEY",
            help="Only count tickets with this key, such as PROJ. Repeatable.",
        ),
    ] = None,
    include_trivial: Annotated[
        bool, typer.Option("--include-trivial", help="Include trivial sessions.")
    ] = False,
    output_format: Annotated[
        LogFormat,
        typer.Option("--format", help="Output format: markdown, json, or jsonl."),
    ] = LogFormat.MARKDOWN,
) -> None:
    """Summarize what sessions did within a time window.

    Each root session, with its sub-agent sessions folded in, is reported with
    the days it was used, the tickets it mentioned, and the pull requests it
    created or referenced. Only messages sent within the window count.
    """
    from ...cli import get_app_context

    zone_name = time_zone or system_time_zone()
    try:
        ZoneInfo(zone_name)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise typer.BadParameter(
            f"unknown time zone {zone_name!r}", param_hint="--tz"
        ) from error
    since_ms = _parse(since, "--since")
    until_ms = None if until is None else _parse(until, "--until")
    if until_ms is not None and since_ms >= until_ms:
        raise typer.BadParameter("--since must be earlier than --until")

    try:
        database_path = resolve_database_path(get_app_context(context).database)
        filters = session_filter(
            database_path,
            repositories=repositories or (),
            cwd=cwd,
            include_children=False,
            include_archived=False,
            archived_only=False,
            include_trivial=include_trivial,
            trivial_only=False,
            agents=(),
            providers=(),
            models=(),
            since=None,
            until=None,
        )
        query = ActivityQuery(
            filters=filters,
            since=since_ms,
            until=until_ms,
            time_zone=zone_name,
            ticket_prefixes=frozenset(ticket_prefixes or ()),
        )
        with OpenCodeDatabase(database_path) as database:
            activities = database.session_activity(query)
        excluded = set(excluded_repositories or ())
        activities = tuple(
            activity
            for activity in activities
            if activity.session.repository not in excluded
        )
    except OpencodePowerPackError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(error.exit_code) from None

    if output_format is LogFormat.JSON:
        emit_json_array(activities)
    elif output_format is LogFormat.JSONL:
        emit_jsonl(activities)
    else:
        typer.echo(render_markdown(activities, query), nl=False)


def render_markdown(activities: Sequence[SessionActivity], query: ActivityQuery) -> str:
    """Render activities grouped by their main ticket, then by repository."""
    zone = ZoneInfo(query.time_zone)
    start = _date(query.since, zone)
    end = "now" if query.until is None else _date(query.until, zone)
    lines = [f"# Work log: {start} to {end} ({query.time_zone})", ""]
    if not activities:
        lines.append("No sessions in this window.")
        return "\n".join(lines) + "\n"

    groups: dict[str, list[SessionActivity]] = {}
    for activity in activities:
        key = activity.tickets[0] if activity.tickets else _NO_TICKET
        groups.setdefault(key, []).append(activity)
    for key in sorted(groups, key=lambda name: (name == _NO_TICKET, name)):
        lines.append(f"## {key}")
        lines.append("")
        if key == _NO_TICKET:
            by_repository: dict[str, list[SessionActivity]] = {}
            for activity in groups[key]:
                name = activity.session.repository or "unknown repository"
                by_repository.setdefault(name, []).append(activity)
            for name in sorted(by_repository):
                lines.append(f"### {name}")
                lines.append("")
                lines.extend(_entry(activity) for activity in by_repository[name])
                lines.append("")
        else:
            lines.extend(_entry(activity) for activity in groups[key])
            lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def _entry(activity: SessionActivity) -> str:
    session = activity.session
    title = session.title or "Untitled"
    parts = [
        f"- **{title}**",
        session.repository or "unknown repository",
        ", ".join(activity.active_days),
        f"`{session.id}`",
    ]
    lines = [" · ".join(parts)]
    for pr in activity.pull_requests_created:
        label = pr.url or f"#{pr.number} {pr.branch}"
        where = (
            f" (in {_home(pr.directory)})" if pr.url is None and pr.directory else ""
        )
        lines.append(f"  - Created {label}{where}")
    if activity.pull_requests_referenced:
        lines.append("  - Referenced " + ", ".join(activity.pull_requests_referenced))
    if len(activity.tickets) > 1:
        lines.append("  - Also mentions " + ", ".join(activity.tickets[1:]))
    return "\n".join(lines)


def _home(directory: str) -> str:
    path = Path(directory)
    try:
        return f"~/{path.relative_to(Path.home())}"
    except ValueError:
        return directory


def _date(timestamp: int, zone: ZoneInfo) -> str:
    return (
        datetime.fromtimestamp(timestamp / 1_000, UTC)
        .astimezone(zone)
        .date()
        .isoformat()
    )


def system_time_zone() -> str:
    """Return the IANA name of the system time zone: ``$TZ``, then
    ``/etc/localtime``, then UTC."""
    zone = os.environ.get("TZ", "").removeprefix(":")
    if zone:
        return zone
    try:
        target = Path(os.readlink(_LOCALTIME))
    except OSError:
        return "UTC"
    parts = target.parts
    if "zoneinfo" in parts:
        return "/".join(parts[parts.index("zoneinfo") + 1 :]) or "UTC"
    return "UTC"


def _parse(value: str, option: str) -> int:
    try:
        return parse_time(value)
    except TimeParseError as error:
        raise typer.BadParameter(str(error), param_hint=option) from error
