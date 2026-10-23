"""Options, filter resolution, and table output shared by the search commands.

``sessions search`` (semantic) and ``sessions grep`` (lexical) accept the same
session filters and print the same kind of result table, differing only in how
they rank sessions and what their last column means.
"""

import re
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import typer

from ...database import OpenCodeDatabase
from ...errors import TimeParseError
from ...models import ArchiveMode, Session, SessionFilter, TrivialMode
from ...output import emit_table
from ...repositories import RepositoryResolver
from ...session_selectors import SessionResolver, selectors_from_arguments
from ...timeparse import parse_time

_WHITESPACE = re.compile(r"\s+")


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class Source(StrEnum):
    TITLE = "title"
    TEXT = "text"


def validate_query(value: str) -> str:
    if not value.strip():
        raise typer.BadParameter("must not be empty")
    return value


def session_filter(
    database_path: Path,
    *,
    repositories: Sequence[str],
    cwd: bool,
    include_children: bool,
    include_archived: bool,
    archived_only: bool,
    include_trivial: bool,
    trivial_only: bool,
    excluded: Sequence[str] = (),
    agents: Iterable[str],
    providers: Iterable[str],
    models: Iterable[str],
    since: str | None,
    until: str | None,
) -> SessionFilter:
    """Validate the shared filter options and resolve them against the database."""
    if repositories and cwd:
        raise typer.BadParameter("--repo cannot be used with --cwd")
    if include_archived and archived_only:
        raise typer.BadParameter(
            "--include-archived cannot be used with --archived-only"
        )
    if include_trivial and trivial_only:
        raise typer.BadParameter("--include-trivial cannot be used with --trivial-only")
    since_ms = _parse_time_option(since, "--since")
    until_ms = _parse_time_option(until, "--until")
    if since_ms is not None and until_ms is not None and since_ms >= until_ms:
        raise typer.BadParameter("--since must be earlier than --until")
    if archived_only:
        archive_mode = ArchiveMode.ARCHIVED
    elif include_archived:
        archive_mode = ArchiveMode.ALL
    else:
        archive_mode = ArchiveMode.ACTIVE
    with OpenCodeDatabase(database_path) as database:
        resolver = RepositoryResolver(database.list_repositories())
        if cwd:
            repository_ids = frozenset({resolver.resolve_cwd(Path.cwd()).project_id})
        else:
            repository_ids = resolver.resolve_many(repositories)
        excluded_ids = (
            session_trees(database.all_sessions(), excluded)
            if excluded
            else frozenset[str]()
        )
    return SessionFilter(
        repository_project_ids=repository_ids,
        include_children=include_children,
        archive_mode=archive_mode,
        trivial_mode=trivial_mode(include_trivial, trivial_only),
        excluded_session_ids=excluded_ids,
        agents=frozenset(agents),
        providers=frozenset(providers),
        models=frozenset(models),
        since=since_ms,
        until=until_ms,
    )


def session_trees(
    sessions: Iterable[Session], selectors: Sequence[str]
) -> frozenset[str]:
    """Return every session in the trees containing the selected sessions.

    A sub-agent's search excludes the conversation that started it, and a root
    session's search excludes the sub-agents it started.
    """
    known = tuple(sessions)
    parents = {session.id: session.parent_id for session in known}
    children: dict[str, list[str]] = {}
    for session in known:
        if session.parent_id is not None:
            children.setdefault(session.parent_id, []).append(session.id)
    resolver = SessionResolver(known)
    tree: set[str] = set()
    for selector in selectors_from_arguments(selectors):
        root = resolver.resolve(selector).id
        while (parent := parents.get(root)) is not None and parent in parents:
            root = parent
        pending = [root]
        while pending:
            current = pending.pop()
            if current not in tree:
                tree.add(current)
                pending.extend(children.get(current, ()))
    return frozenset(tree)


def trivial_mode(include_trivial: bool, trivial_only: bool) -> TrivialMode:
    """Map the trivial-session options to a filter mode."""
    if trivial_only:
        return TrivialMode.TRIVIAL
    if include_trivial:
        return TrivialMode.ALL
    return TrivialMode.SUBSTANTIVE


def selected_roles(roles: Sequence[Role] | None) -> frozenset[str]:
    return frozenset(roles or (Role.USER, Role.ASSISTANT))


def emit_matches(
    last_column: str,
    rows: Iterable[tuple[Session, str, str, object]],
) -> None:
    """Print ``(session, source, snippet, last column value)`` rows as a table."""
    emit_table(
        (
            "Updated",
            "Repository",
            "Title",
            "Source",
            "Snippet",
            last_column,
            "Session ID",
        ),
        (
            (
                _display_updated(session.updated),
                f"{_display(session.repository)}\n"
                f"{_display_directory(session.directory)}",
                _display(session.title),
                _display(source),
                _display(snippet),
                value,
                _display(session.id),
            )
            for session, source, snippet, value in rows
        ),
    )


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


def _display_updated(value: int | None) -> str:
    if value is None:
        return "—"
    try:
        return datetime.fromtimestamp(value / 1_000, UTC).strftime("%Y-%m-%d %H:%M")
    except OverflowError, OSError, ValueError:
        return _display(value)
