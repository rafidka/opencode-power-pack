"""Session conversion, filtering, and list ordering."""

import os
import sqlite3

from ..models import (
    ArchiveMode,
    ListSessionsQuery,
    ListSort,
    Session,
    SessionFilter,
    SortDirection,
    TokenBreakdown,
)
from ._utils import json_object, string


def filtered_sessions(
    connection: sqlite3.Connection, filters: SessionFilter
) -> list[Session]:
    """Load session summaries and retain those matching common filters."""
    projects = {
        row["id"]: row["name"] or os.path.basename(row["worktree"])
        for row in connection.execute("SELECT id, worktree, name FROM project")
    }
    rows = connection.execute("SELECT * FROM session").fetchall()
    sessions = [session_from_row(row, projects.get(row["project_id"])) for row in rows]
    return [session for session in sessions if matches_filters(session, filters)]


def session_from_row(row: sqlite3.Row, repository: str | None) -> Session:
    """Convert a native session row to its storage-independent summary."""
    model_data = json_object(row["model"])
    tokens = TokenBreakdown(
        input=row["tokens_input"],
        output=row["tokens_output"],
        reasoning=row["tokens_reasoning"],
        cache_read=row["tokens_cache_read"],
        cache_write=row["tokens_cache_write"],
    )
    return Session(
        id=row["id"],
        title=row["title"],
        created=row["time_created"],
        updated=row["time_updated"],
        project_id=row["project_id"],
        directory=row["directory"],
        parent_id=row["parent_id"],
        repository=repository,
        agent=row["agent"],
        provider=string(model_data.get("providerID")),
        model=string(model_data.get("id")),
        cost=row["cost"],
        tokens=tokens,
        archived=row["time_archived"] is not None,
        archived_at=row["time_archived"],
    )


def matches_filters(session: Session, filters: SessionFilter) -> bool:
    """Return whether a session satisfies common list and search filters."""
    if not filters.include_children and session.parent_id is not None:
        return False
    if filters.archive_mode is ArchiveMode.ACTIVE and session.archived:
        return False
    if filters.archive_mode is ArchiveMode.ARCHIVED and not session.archived:
        return False
    if (
        filters.repository_project_ids
        and session.project_id not in filters.repository_project_ids
    ):
        return False
    if filters.agents and session.agent not in filters.agents:
        return False
    if filters.providers and session.provider not in filters.providers:
        return False
    if filters.models and session.model not in filters.models:
        return False
    return not (
        (
            filters.since is not None
            and (session.updated is None or session.updated < filters.since)
        )
        or (
            filters.until is not None
            and (session.updated is None or session.updated >= filters.until)
        )
    )


def sort_and_limit_sessions(
    sessions: list[Session], query: ListSessionsQuery
) -> tuple[Session, ...]:
    """Apply deterministic list ordering and an optional limit."""
    sessions.sort(
        key=lambda session: session.id, reverse=query.direction is SortDirection.DESC
    )
    sessions.sort(
        key=lambda session: list_value(session, query),
        reverse=query.direction is SortDirection.DESC,
    )
    if query.limit is not None:
        sessions = sessions[: query.limit]
    return tuple(sessions)


def list_value(session: Session, query: ListSessionsQuery) -> int | float | str:
    """Return the primary sort value for a session list query."""
    value: object
    if query.sort is ListSort.UPDATED:
        value = session.updated if session.updated is not None else -1
    elif query.sort is ListSort.CREATED:
        value = session.created if session.created is not None else -1
    elif query.sort is ListSort.COST:
        value = session.cost if session.cost is not None else -1.0
    elif query.sort is ListSort.TOKENS:
        value = (
            session.tokens.total
            if session.tokens and session.tokens.total is not None
            else -1
        )
    else:
        value = (session.title or "").casefold()
    assert isinstance(value, int | float | str)
    return value
