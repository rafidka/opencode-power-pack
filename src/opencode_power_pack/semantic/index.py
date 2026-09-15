"""Persistent exact vector storage, separate from OpenCode's source database."""

import hashlib
import math
import os
import sqlite3
import struct
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Self

from pydantic import SkipValidation, ValidationError

from ..models import ValueModel
from .cohere import (
    COHERE_DIMENSION,
    COHERE_MODEL,
    CohereEmbeddingContract,
    EmbeddingVector,
)

SCHEMA_VERSION = 2
_SQLITE_BIND_BATCH_SIZE = 900
"""Values per ``IN`` query, below SQLite's legacy 999-bind limit."""


class SemanticIndexError(Exception):
    """An index cannot be safely opened or used."""


class FileFingerprint(ValueModel):
    exists: bool
    size: int | None
    mtime_ns: int | None


class SourceFingerprint(ValueModel):
    """Stable source-database change signals used to skip synchronization."""

    main: FileFingerprint
    wal: FileFingerprint

    def to_json(self) -> str:
        """Return the private cache representation of this fingerprint."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, value: str) -> Self | None:
        """Return the stored fingerprint, or ``None`` when it cannot be read.

        A fingerprint only decides whether synchronization can be skipped, so an
        unreadable one is treated as absent and provokes a rescan rather than
        failing the command. That keeps a stored cache from outliving the shape
        it was written in.
        """
        try:
            return cls.model_validate_json(value)
        except ValidationError:
            return None


class SemanticChunk(ValueModel):
    document_id: str
    session_id: str
    source: str
    role: str | None
    content: str
    # Vectors are produced by the embedding provider or decoded from this index,
    # and their dimension and finiteness are checked explicitly where they enter.
    # Re-validating a thousand floats per chunk would dominate indexing.
    embedding: Annotated[EmbeddingVector, SkipValidation]
    metadata: str | None = None


class SearchHit(ValueModel):
    chunk: SemanticChunk
    score: float


class SessionSearchResult(ValueModel):
    session_id: str
    hits: tuple[SearchHit, ...]

    @property
    def score(self) -> float:
        return self.hits[0].score


class IndexStatus(ValueModel):
    path: Path
    contract: CohereEmbeddingContract
    source_fingerprint: SourceFingerprint | None
    session_count: int
    chunk_count: int


def resolve_sidecar_path(
    source_database: str | Path, *, cache_home: str | Path | None = None
) -> Path:
    source = Path(source_database).expanduser().resolve(strict=False)
    digest = hashlib.sha256(os.fsencode(str(source))).hexdigest()
    root = (
        Path(cache_home).expanduser()
        if cache_home is not None
        else Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
    )
    return root / "opencode-power-pack" / f"semantic-{digest}.sqlite3"


def source_fingerprint(source_database: str | Path) -> SourceFingerprint:
    database = Path(source_database).expanduser().resolve(strict=False)

    def fingerprint(path: Path) -> FileFingerprint:
        try:
            result = path.stat()
        except FileNotFoundError:
            return FileFingerprint(exists=False, size=None, mtime_ns=None)
        return FileFingerprint(
            exists=True, size=result.st_size, mtime_ns=result.st_mtime_ns
        )

    return SourceFingerprint(
        main=fingerprint(database), wal=fingerprint(Path(f"{database}-wal"))
    )


class SemanticIndex:
    """A writable sidecar index for one embedding identity."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.contract = CohereEmbeddingContract()
        self._connection: sqlite3.Connection | None = None

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        replace_identity: bool = False,
    ) -> Self:
        """Open an index, optionally replacing its identity-derived contents."""
        index = cls(path)
        index.initialize(replace_identity=replace_identity)
        return index

    def initialize(self, *, replace_identity: bool = False) -> None:
        if self._connection is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        self._connection = connection
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            with self._transaction():
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS metadata "
                    "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS sessions "
                    "(session_id TEXT PRIMARY KEY, revision TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS chunks ("
                    "document_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
                    "source TEXT NOT NULL, role TEXT, content TEXT NOT NULL, "
                    "embedding BLOB NOT NULL, norm REAL NOT NULL, metadata TEXT, "
                    "FOREIGN KEY(session_id) REFERENCES sessions(session_id))"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS chunks_session_idx "
                    "ON chunks(session_id)"
                )
                self._initialize_metadata(connection, replace_identity=replace_identity)
        except BaseException:
            connection.close()
            self._connection = None
            raise

    def _initialize_metadata(
        self, connection: sqlite3.Connection, *, replace_identity: bool
    ) -> None:
        existing = dict(connection.execute("SELECT key, value FROM metadata"))
        valid_versions = (None, str(SCHEMA_VERSION))
        if (version := existing.get("schema_version")) not in valid_versions:
            if not replace_identity:
                raise SemanticIndexError(
                    "unsupported semantic index schema "
                    f"{version}; expected {SCHEMA_VERSION}; rebuild the semantic index"
                )
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM sessions")
            connection.execute("DELETE FROM metadata")
            existing = {}
        keys = ("cohere_model", "cohere_dimension")
        expected = (COHERE_MODEL, str(COHERE_DIMENSION))
        stored = tuple(existing.get(key) for key in keys)
        has_legacy_identity = any(
            key in existing
            for key in ("embedding_provider", "embedding_model", "embedding_dimension")
        )
        if (has_legacy_identity or any(value is not None for value in stored)) and (
            has_legacy_identity or stored != expected
        ):
            if not replace_identity:
                raise SemanticIndexError(
                    "Cohere embedding contract mismatch; rebuild the semantic index"
                )
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM sessions")
            connection.execute("DELETE FROM metadata")
        for key, value in zip(
            ("schema_version", *keys), (str(SCHEMA_VERSION), *expected), strict=True
        ):
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                (key, value),
            )

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> Self:
        self.initialize()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._connection is not None

    def status(self) -> IndexStatus:
        connection = self._require_connection()
        return IndexStatus(
            path=self.path,
            contract=self.contract,
            source_fingerprint=self.stored_fingerprint(),
            session_count=int(
                connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            ),
            chunk_count=int(
                connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            ),
        )

    def stored_fingerprint(self) -> SourceFingerprint | None:
        row = (
            self._require_connection()
            .execute("SELECT value FROM metadata WHERE key = 'source_fingerprint'")
            .fetchone()
        )
        return None if row is None else SourceFingerprint.from_json(str(row[0]))

    def indexed_revisions(self) -> dict[str, str]:
        return dict(
            self._require_connection().execute(
                "SELECT session_id, revision FROM sessions"
            )
        )

    def document_ids_by_session(
        self, session_ids: Iterable[str]
    ) -> dict[str, frozenset[str]]:
        """Return stored document IDs for each requested session."""
        ids = tuple(dict.fromkeys(session_ids))
        if not ids:
            return {}
        result: dict[str, set[str]] = {session_id: set() for session_id in ids}
        connection = self._require_connection()
        for start in range(0, len(ids), _SQLITE_BIND_BATCH_SIZE):
            batch = ids[start : start + _SQLITE_BIND_BATCH_SIZE]
            placeholders = ",".join("?" for _ in batch)
            rows = connection.execute(
                "SELECT session_id, document_id FROM chunks "
                f"WHERE session_id IN ({placeholders})",
                batch,
            )
            for row in rows:
                result[str(row["session_id"])].add(str(row["document_id"]))
        return {
            session_id: frozenset(document_ids)
            for session_id, document_ids in result.items()
        }

    def synchronize_session(
        self,
        session_id: str,
        revision: str,
        current_document_ids: Iterable[str],
        new_chunks: Sequence[SemanticChunk],
    ) -> None:
        """Atomically retain current chunks, add new chunks, and update revision."""
        if not session_id or not revision:
            raise SemanticIndexError("session ID and revision must be non-empty")
        current_ids = tuple(current_document_ids)
        if len(set(current_ids)) != len(current_ids) or any(
            not document_id for document_id in current_ids
        ):
            raise SemanticIndexError(
                "current document IDs must be unique and non-empty"
            )
        prepared = [self._prepare_chunk(chunk, session_id) for chunk in new_chunks]
        new_ids = {chunk.document_id for chunk, _, _ in prepared}
        if len(new_ids) != len(prepared):
            raise SemanticIndexError("new chunks contain duplicate document IDs")
        if not new_ids <= set(current_ids):
            raise SemanticIndexError("new chunks must be current session documents")
        with self._transaction() as connection:
            if current_ids:
                placeholders = ",".join("?" for _ in current_ids)
                connection.execute(
                    "DELETE FROM chunks WHERE session_id = ? "
                    f"AND document_id NOT IN ({placeholders})",
                    (session_id, *current_ids),
                )
            else:
                connection.execute(
                    "DELETE FROM chunks WHERE session_id = ?", (session_id,)
                )
            connection.execute(
                "INSERT INTO sessions(session_id, revision) VALUES (?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET revision = excluded.revision",
                (session_id, revision),
            )
            connection.executemany(
                "INSERT INTO chunks(document_id, session_id, source, role, content, "
                "embedding, norm, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        chunk.document_id,
                        chunk.session_id,
                        chunk.source,
                        chunk.role,
                        chunk.content,
                        vector,
                        norm,
                        chunk.metadata,
                    )
                    for chunk, vector, norm in prepared
                ],
            )

    def replace_session(
        self, session_id: str, revision: str, chunks: Sequence[SemanticChunk]
    ) -> None:
        if not session_id or not revision:
            raise SemanticIndexError("session ID and revision must be non-empty")
        prepared = [self._prepare_chunk(chunk, session_id) for chunk in chunks]
        if len({chunk.document_id for chunk, _, _ in prepared}) != len(prepared):
            raise SemanticIndexError(
                "a session replacement contains duplicate document IDs"
            )
        with self._transaction() as connection:
            connection.execute("DELETE FROM chunks WHERE session_id = ?", (session_id,))
            connection.execute(
                "INSERT INTO sessions(session_id, revision) VALUES (?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET revision = excluded.revision",
                (session_id, revision),
            )
            connection.executemany(
                "INSERT INTO chunks(document_id, session_id, source, role, content, "
                "embedding, norm, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        chunk.document_id,
                        chunk.session_id,
                        chunk.source,
                        chunk.role,
                        chunk.content,
                        vector,
                        norm,
                        chunk.metadata,
                    )
                    for chunk, vector, norm in prepared
                ],
            )

    def delete_sessions(self, session_ids: Iterable[str]) -> None:
        ids = tuple(session_ids)
        with self._transaction() as connection:
            connection.executemany(
                "DELETE FROM chunks WHERE session_id = ?",
                ((session_id,) for session_id in ids),
            )
            connection.executemany(
                "DELETE FROM sessions WHERE session_id = ?",
                ((session_id,) for session_id in ids),
            )

    def clear(self) -> None:
        with self._transaction() as connection:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM sessions")
            connection.execute("DELETE FROM metadata WHERE key = 'source_fingerprint'")

    def clear_fingerprint(self) -> None:
        """Mark the source as needing another synchronization check."""
        with self._transaction() as connection:
            connection.execute("DELETE FROM metadata WHERE key = 'source_fingerprint'")

    def rebuild(self) -> None:
        self.clear()

    def record_fingerprint(self, fingerprint: SourceFingerprint) -> None:
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('source_fingerprint', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (fingerprint.to_json(),),
            )

    def search(
        self,
        query: Sequence[float],
        *,
        session_ids: Iterable[str] | None = None,
        sources: Iterable[str] | None = None,
        roles: Iterable[str | None] | None = None,
        chunks_per_session: int = 1,
        limit_sessions: int | None = None,
    ) -> tuple[SessionSearchResult, ...]:
        if chunks_per_session <= 0 or (
            limit_sessions is not None and limit_sessions <= 0
        ):
            raise SemanticIndexError("search limits must be positive")
        query_values, query_norm = self._normalize_vector(query)
        clauses, parameters = self._filters(
            (("session_id", session_ids), ("source", sources), ("role", roles))
        )
        if clauses is None:
            return ()
        statement = "SELECT * FROM chunks" + (
            " WHERE " + " AND ".join(clauses) if clauses else ""
        )
        hits: list[SearchHit] = []
        for row in self._require_connection().execute(statement, parameters):
            chunk, norm = self._chunk_from_row(row)
            score = (
                0.0
                if query_norm == 0.0 or norm == 0.0
                else sum(
                    left * right
                    for left, right in zip(query_values, chunk.embedding, strict=True)
                )
                / (query_norm * norm)
            )
            hits.append(SearchHit(chunk=chunk, score=score))
        hits.sort(
            key=lambda hit: (-hit.score, hit.chunk.session_id, hit.chunk.document_id)
        )
        grouped: dict[str, list[SearchHit]] = {}
        for hit in hits:
            if len(grouped.setdefault(hit.chunk.session_id, [])) < chunks_per_session:
                grouped[hit.chunk.session_id].append(hit)
        results = [
            SessionSearchResult(session_id=session_id, hits=tuple(session_hits))
            for session_id, session_hits in grouped.items()
        ]
        results.sort(key=lambda result: (-result.score, result.session_id))
        return tuple(results if limit_sessions is None else results[:limit_sessions])

    def _filters(
        self, filters: Sequence[tuple[str, Iterable[str | None] | None]]
    ) -> tuple[list[str] | None, list[str]]:
        clauses: list[str] = []
        parameters: list[str] = []
        for column, values in filters:
            if values is None:
                continue
            selected = tuple(values)
            if not selected:
                return None, []
            non_null = tuple(value for value in selected if value is not None)
            parts: list[str] = []
            if any(value is None for value in selected):
                parts.append(f"{column} IS NULL")
            if non_null:
                parts.append(f"{column} IN ({','.join('?' for _ in non_null)})")
                parameters.extend(non_null)
            clauses.append("(" + " OR ".join(parts) + ")")
        return clauses, parameters

    def _prepare_chunk(
        self, chunk: SemanticChunk, session_id: str
    ) -> tuple[SemanticChunk, bytes, float]:
        if chunk.session_id != session_id:
            raise SemanticIndexError(
                "all replacement chunks must belong to the target session"
            )
        if not all((chunk.document_id, chunk.source, chunk.content)):
            raise SemanticIndexError(
                "chunk document ID, source, and content must be non-empty"
            )
        values, norm = self._normalize_vector(chunk.embedding)
        return (
            SemanticChunk(
                document_id=chunk.document_id,
                session_id=chunk.session_id,
                source=chunk.source,
                role=chunk.role,
                content=chunk.content,
                embedding=values,
                metadata=chunk.metadata,
            ),
            struct.pack(f"<{COHERE_DIMENSION}f", *values),
            norm,
        )

    def _normalize_vector(
        self, vector: Sequence[float]
    ) -> tuple[EmbeddingVector, float]:
        if len(vector) != COHERE_DIMENSION:
            raise SemanticIndexError(
                f"invalid vector dimension; expected {COHERE_DIMENSION}"
            )
        try:
            values = tuple(
                struct.unpack("<f", struct.pack("<f", float(value)))[0]
                for value in vector
            )
        except (OverflowError, TypeError, ValueError, struct.error) as error:
            raise SemanticIndexError(
                "embedding vector must contain float32 values"
            ) from error
        if not all(math.isfinite(value) for value in values):
            raise SemanticIndexError("embedding vector contains a non-finite value")
        return values, math.sqrt(sum(value * value for value in values))

    def _chunk_from_row(self, row: sqlite3.Row) -> tuple[SemanticChunk, float]:
        data = bytes(row["embedding"])
        if len(data) != COHERE_DIMENSION * 4:
            raise SemanticIndexError("stored embedding has an invalid dimension")
        embedding = tuple(struct.unpack(f"<{COHERE_DIMENSION}f", data))
        return SemanticChunk(
            document_id=row["document_id"],
            session_id=row["session_id"],
            source=row["source"],
            role=row["role"],
            content=row["content"],
            embedding=embedding,
            metadata=row["metadata"],
        ), float(row["norm"])

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection]:
        connection = self._require_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise SemanticIndexError(
                "semantic index is closed; call initialize() or open()"
            )
        return self._connection
