"""Read-only SQLite client and repository-loading orchestration."""

import os
import sqlite3
from collections.abc import Callable, Collection, Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

from ..errors import DatabaseError
from ..models import (
    ListSessionsQuery,
    Repository,
    SearchSessionMatch,
    SearchSessionsQuery,
    Session,
)
from ..semantic.documents import (
    SemanticDocument,
    SemanticDocumentSource,
    SessionDocumentBatch,
    SessionRevision,
    make_semantic_documents,
)
from ._utils import json_object, string
from .search import search_session, sort_and_limit_matches
from .sessions import filtered_sessions, sort_and_limit_sessions

_REQUIRED_COLUMNS = {
    "project": {"id", "worktree", "name"},
    "project_directory": {"project_id", "directory"},
    "session": {
        "id",
        "project_id",
        "parent_id",
        "title",
        "directory",
        "cost",
        "agent",
        "model",
        "time_created",
        "time_updated",
        "time_archived",
        "tokens_input",
        "tokens_output",
        "tokens_reasoning",
        "tokens_cache_read",
        "tokens_cache_write",
    },
    "message": {"id", "session_id", "data"},
    "part": {"message_id", "session_id", "data"},
    "todo": {"session_id", "content"},
}

_SQLITE_SESSION_ID_BIND_BATCH_SIZE = 900
"""IDs per semantic query, leaving headroom below SQLite's legacy 999-bind limit."""


def resolve_database_path(explicit: Path | None = None) -> Path:
    """Return the explicit database path or OpenCode's data-directory default."""
    if explicit is not None:
        return explicit
    data_home = os.environ.get("XDG_DATA_HOME")
    if data_home is None:
        data_home = os.fspath(Path.home() / ".local" / "share")
    return Path(data_home) / "opencode" / "opencode.db"


class OpenCodeDatabase:
    """A validated, read-only connection to an OpenCode database."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._connection: sqlite3.Connection | None = None
        self._context_transaction = False
        self._part_schema_has_id_column = False

    def __enter__(self) -> OpenCodeDatabase:
        self.open()
        assert self._connection is not None
        self._connection.execute("BEGIN")
        self._context_transaction = True
        return self

    def __exit__(self, *_: object) -> None:
        if self._connection is not None and self._connection.in_transaction:
            self._connection.rollback()
        self._context_transaction = False
        self.close()

    def open(self) -> None:
        """Open and validate the database without granting write access."""
        if self._connection is not None:
            return
        absolute = self.path.absolute().as_posix()
        uri = f"file:{quote(absolute)}?mode=ro"
        try:
            connection = sqlite3.connect(uri, uri=True, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            self._connection = connection
            self._validate_schema()
        except DatabaseError:
            self.close()
            raise
        except sqlite3.Error as error:
            self.close()
            msg = f"unable to open OpenCode database {self.path}: {error}"
            raise DatabaseError(msg) from error

    def close(self) -> None:
        """Close the underlying connection, if one is open."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def list_repositories(self) -> tuple[Repository, ...]:
        """Return projects with directory and observed-session aliases."""
        with self._read_transaction():
            rows = (
                self._connection_or_error()
                .execute("SELECT id, worktree, name FROM project ORDER BY id")
                .fetchall()
            )
            aliases = self._repository_aliases()
        return tuple(
            Repository(
                display_name=row["name"] or os.path.basename(row["worktree"]),
                project_id=row["id"],
                path=row["worktree"],
                aliases=tuple(
                    alias
                    for alias in aliases.get(row["id"], ())
                    if alias != row["worktree"]
                ),
            )
            for row in rows
        )

    def list_sessions(self, query: ListSessionsQuery) -> tuple[Session, ...]:
        """List session summaries matching ``query``."""
        with self._read_transaction():
            return sort_and_limit_sessions(
                filtered_sessions(self._connection_or_error(), query.filters), query
            )

    def search_sessions(
        self, query: SearchSessionsQuery
    ) -> tuple[SearchSessionMatch, ...]:
        """Search current title, message part, and todo materialized records."""
        with self._read_transaction():
            connection = self._connection_or_error()
            matches = [
                search_session(connection, session, query)
                for session in filtered_sessions(connection, query.filters)
            ]
        return sort_and_limit_matches(
            [match for match in matches if match is not None], query
        )

    def list_semantic_session_revisions(self) -> Collection[SessionRevision]:
        """Return a lazy, sized view of session revisions.

        Iteration streams one query rather than building a tuple for every session.
        Consume the view while this database remains open; inside a database context
        it shares that context's read transaction, preserving a consistent scan.
        """
        return _SemanticSessionRevisions(
            self._iter_semantic_session_revisions,
            self._semantic_session_revision_count,
        )

    def fetch_semantic_documents(
        self, session_ids: Collection[str]
    ) -> tuple[SessionDocumentBatch, ...]:
        """Orchestrate batched SQL reads into per-session safe document batches.

        The public method accepts an arbitrary collection, normalizes it, and joins
        results from bounded SQL batches. Its private counterpart executes one such
        normalized SQL batch.
        """
        selected = tuple(sorted(set(session_ids)))
        if not selected:
            return ()
        batches: list[SessionDocumentBatch] = []
        with self._read_transaction():
            for session_id_group in _groups(
                selected, _SQLITE_SESSION_ID_BIND_BATCH_SIZE
            ):
                batches.extend(self._fetch_semantic_document_batch(session_id_group))
        return tuple(batches)

    def _fetch_semantic_document_batch(
        self, session_ids: tuple[str, ...]
    ) -> tuple[SessionDocumentBatch, ...]:
        """Execute the three semantic source queries for one bounded ID batch."""
        connection = self._connection_or_error()
        placeholders = ", ".join("?" for _ in session_ids)
        sessions = connection.execute(
            "SELECT id, title, time_updated FROM session "
            f"WHERE id IN ({placeholders}) ORDER BY id",
            session_ids,
        ).fetchall()
        messages = connection.execute(
            "SELECT id, session_id, data FROM message "
            f"WHERE session_id IN ({placeholders}) ORDER BY session_id, id",
            session_ids,
        ).fetchall()
        roles = {
            row["id"]: string(json_object(row["data"]).get("role")) for row in messages
        }
        part_select = (
            "SELECT id AS source_record_id, session_id, message_id, data FROM part "
            if self._part_schema_has_id_column
            else (
                "SELECT rowid AS source_record_id, session_id, message_id, data "
                "FROM part "
            )
        )
        parts = connection.execute(
            part_select
            + f"WHERE session_id IN ({placeholders}) "
            + "ORDER BY session_id, message_id, data",
            session_ids,
        ).fetchall()
        documents_by_session: dict[str, list[SemanticDocument]] = {
            row["id"]: [] for row in sessions
        }
        for row in sessions:
            title = row["title"]
            if isinstance(title, str) and title:
                documents_by_session[row["id"]].extend(
                    make_semantic_documents(
                        session_id=row["id"],
                        source=SemanticDocumentSource.TITLE,
                        role=None,
                        source_record_id="title",
                        source_ordinal=0,
                        content=title,
                    )
                )
        for row in parts:
            role = roles.get(row["message_id"])
            data = json_object(row["data"])
            text = string(data.get("text"))
            if (
                role not in {"user", "assistant"}
                or string(data.get("type")) != "text"
                or not text
            ):
                continue
            documents_by_session[row["session_id"]].extend(
                make_semantic_documents(
                    session_id=row["session_id"],
                    source=SemanticDocumentSource.TEXT,
                    role=role,
                    source_record_id=str(row["source_record_id"]),
                    source_ordinal=0,
                    content=text,
                )
            )
        return tuple(
            SessionDocumentBatch(
                revision=SessionRevision(
                    session_id=row["id"], time_updated=row["time_updated"]
                ),
                documents=tuple(documents_by_session[row["id"]]),
            )
            for row in sessions
        )

    def _validate_schema(self) -> None:
        connection = self._connection_or_error()
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        missing: list[str] = []
        part_columns: set[str] = set()
        for table, required in _REQUIRED_COLUMNS.items():
            if table not in tables:
                missing.append(table)
                continue
            actual = {
                row["name"] for row in connection.execute(f"PRAGMA table_info({table})")
            }
            if table == "part":
                part_columns = actual
            absent = sorted(required - actual)
            if absent:
                missing.append(f"{table} ({', '.join(absent)})")
        if missing:
            msg = "incompatible OpenCode database schema: missing " + "; ".join(missing)
            raise DatabaseError(msg)
        self._part_schema_has_id_column = "id" in part_columns

    @contextmanager
    def _read_transaction(self) -> Generator[None]:
        self.open()
        connection = self._connection_or_error()
        started = not connection.in_transaction
        if started:
            connection.execute("BEGIN")
        try:
            yield
        finally:
            if started and connection.in_transaction:
                connection.rollback()

    def _repository_aliases(self) -> dict[str, tuple[str, ...]]:
        connection = self._connection_or_error()
        aliases: dict[str, list[str]] = {}
        for row in connection.execute(
            "SELECT project_id, directory FROM project_directory"
        ):
            if row["directory"] is not None:
                aliases.setdefault(row["project_id"], []).append(row["directory"])
        for row in connection.execute(
            "SELECT project_id, directory FROM session "
            "WHERE project_id IS NOT NULL AND directory IS NOT NULL"
        ):
            aliases.setdefault(row["project_id"], []).append(row["directory"])
        return {
            project_id: tuple(dict.fromkeys(paths))
            for project_id, paths in aliases.items()
        }

    def _iter_semantic_session_revisions(self) -> Iterator[SessionRevision]:
        with self._read_transaction():
            rows = self._connection_or_error().execute(
                "SELECT id, time_updated FROM session ORDER BY id"
            )
            yield from (
                SessionRevision(session_id=row["id"], time_updated=row["time_updated"])
                for row in rows
            )

    def _semantic_session_revision_count(self) -> int:
        with self._read_transaction():
            row = (
                self._connection_or_error()
                .execute("SELECT COUNT(*) AS count FROM session")
                .fetchone()
            )
        assert row is not None
        return int(row["count"])

    def _connection_or_error(self) -> sqlite3.Connection:
        if self._connection is None:
            msg = "OpenCode database connection is not open"
            raise DatabaseError(msg)
        return self._connection


def _groups(values: tuple[str, ...], size: int) -> Generator[tuple[str, ...]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


class _SemanticSessionRevisions(Collection[SessionRevision]):
    """Lazy revision view whose queries run while the database is open."""

    def __init__(
        self,
        iterate: Callable[[], Iterator[SessionRevision]],
        count: Callable[[], int],
    ) -> None:
        self._iterate = iterate
        self._count = count

    def __iter__(self) -> Iterator[SessionRevision]:
        return self._iterate()

    def __len__(self) -> int:
        return self._count()

    def __contains__(self, value: object) -> bool:
        return any(revision == value for revision in self)
