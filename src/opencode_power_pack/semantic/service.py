"""Cohere-backed synchronization and search over semantic sidecar indexes."""

import re
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ..errors import OpencodePowerPackError, RuntimeCommandError
from ..models import (
    ListSessionsQuery,
    RecordModel,
    SearchSource,
    Session,
    SessionFilter,
    ValueModel,
)
from .cohere import (
    CohereDocumentBatchCompleted,
    CohereEmbedProvider,
    EmbeddingBatch,
    document_batch_count,
)
from .documents import SemanticDocument, SessionDocumentBatch
from .index import (
    SemanticChunk,
    SemanticIndex,
    SemanticIndexError,
    SessionSearchResult,
    resolve_sidecar_path,
    source_fingerprint,
)

if TYPE_CHECKING:
    from ..database.client import OpenCodeDatabase

_DEFAULT_SOURCES = frozenset({SearchSource.TITLE, SearchSource.TEXT})
_SUPPORTED_SOURCES = _DEFAULT_SOURCES
_SNIPPET_WHITESPACE = re.compile(r"\s+")
_DEFAULT_FILTERS = SessionFilter()

SEMANTIC_RESULT_LIMIT = 5
"""Sessions a semantic search returns when no count is requested."""

SEMANTIC_RERANK_CANDIDATES = 50
"""Chunks taken from cosine pre-ranking and rescored by Cohere Rerank."""

SEMANTIC_CHUNKS_PER_SESSION = 3
"""Most chunks one session contributes, so a long session cannot crowd out others."""


class SemanticServiceError(RuntimeCommandError):
    """A semantic sidecar could not be synchronized or searched safely."""


class SemanticSyncPlanned(ValueModel):
    changed_session_count: int
    deleted_session_count: int
    document_count: int
    batch_count: int


class SemanticEmbeddingBatchCompleted(ValueModel):
    batch_number: int
    batch_count: int
    document_count: int
    completed_documents: int
    total_documents: int


class SemanticSessionSynchronized(ValueModel):
    session_id: str


class SemanticSessionsDeleted(ValueModel):
    session_ids: tuple[str, ...]


type SemanticSyncEvent = (
    SemanticSyncPlanned
    | SemanticEmbeddingBatchCompleted
    | SemanticSessionSynchronized
    | SemanticSessionsDeleted
)
type SemanticSyncProgressCallback = Callable[[SemanticSyncEvent], None]


class _ProgressCallbackError(Exception):
    def __init__(self, error: Exception) -> None:
        self.error = error


class SemanticSyncResult(ValueModel):
    """The completed work and freshness state of one synchronization."""

    no_op: bool
    indexed_session_ids: tuple[str, ...]
    deleted_session_ids: tuple[str, ...]
    source_changed: bool

    @property
    def indexed_session_count(self) -> int:
        return len(self.indexed_session_ids)

    @property
    def deleted_session_count(self) -> int:
        return len(self.deleted_session_ids)


class SemanticSearchMatch(RecordModel):
    """One ranked semantic result, grouped from its best retained chunks.

    ``score`` is Cohere Rerank's relevance score, a calibrated probability that
    the best chunk is useful, or the cosine similarity when reranking is off.
    """

    session: Session
    source: SearchSource
    role: str | None
    snippet: str
    score: float
    match_count: int


class SemanticService:
    """Synchronize one OpenCode database with a Cohere semantic sidecar."""

    def __init__(
        self,
        source_database: str | Path,
        embedder: CohereEmbedProvider,
        sidecar_path: str | Path | None = None,
    ) -> None:
        self.source_database = Path(source_database).expanduser()
        self._embedder = embedder
        self.sidecar_path = (
            Path(sidecar_path).expanduser()
            if sidecar_path is not None
            else resolve_sidecar_path(self.source_database)
        )
        source_path = self.source_database.resolve(strict=False)
        unsafe_paths = {
            source_path,
            Path(f"{source_path}-journal"),
            Path(f"{source_path}-shm"),
            Path(f"{source_path}-wal"),
        }
        if self.sidecar_path.resolve(strict=False) in unsafe_paths:
            raise SemanticServiceError(
                "semantic index path must not be the OpenCode source database or "
                "its SQLite sidecars"
            )

    def sync(
        self,
        *,
        force: bool = False,
        rebuild: bool = False,
        on_progress: SemanticSyncProgressCallback | None = None,
    ) -> SemanticSyncResult:
        """Incrementally embed changed sessions and update the sidecar index.

        Progress callback errors propagate and stop synchronization before later work.
        """
        try:
            before = source_fingerprint(self.source_database)
            with SemanticIndex.open(
                self.sidecar_path,
                replace_identity=rebuild,
            ) as index:
                if not force and not rebuild and index.stored_fingerprint() == before:
                    return SemanticSyncResult(
                        no_op=True,
                        indexed_session_ids=(),
                        deleted_session_ids=(),
                        source_changed=False,
                    )
                with _open_database(self.source_database) as database:
                    revisions = database.list_semantic_session_revisions()
                    source_revisions = {
                        revision.session_id: _serialize_revision(revision.time_updated)
                        for revision in revisions
                    }
                    if len(source_revisions) != len(revisions):
                        raise SemanticServiceError(
                            "OpenCode database returned duplicate session revisions"
                        )
                    indexed_revisions = {} if rebuild else index.indexed_revisions()
                    deleted = tuple(
                        sorted(indexed_revisions.keys() - source_revisions.keys())
                    )
                    changed = tuple(
                        session_id
                        for session_id in sorted(source_revisions)
                        if indexed_revisions.get(session_id)
                        != source_revisions[session_id]
                    )
                    batches = database.fetch_semantic_documents(changed)

                batches_by_session = _batches_by_session(
                    batches, changed, source_revisions
                )
                stored_ids_by_session = (
                    {session_id: frozenset[str]() for session_id in changed}
                    if rebuild
                    else index.document_ids_by_session(changed)
                )
                new_documents = tuple(
                    document
                    for session_id in changed
                    for document in batches_by_session[session_id].documents
                    if document.id not in stored_ids_by_session[session_id]
                )
                _emit_progress(
                    on_progress,
                    SemanticSyncPlanned(
                        changed_session_count=len(changed),
                        deleted_session_count=len(deleted),
                        document_count=len(new_documents),
                        batch_count=document_batch_count(len(new_documents)),
                    ),
                )
                if rebuild:
                    index.rebuild()
                else:
                    index.clear_fingerprint()

                def on_batch(batch: CohereDocumentBatchCompleted) -> None:
                    _emit_progress(
                        on_progress,
                        SemanticEmbeddingBatchCompleted(
                            batch_number=batch.batch_number,
                            batch_count=batch.batch_count,
                            document_count=batch.document_count,
                            completed_documents=batch.completed_documents,
                            total_documents=batch.total_documents,
                        ),
                    )

                vectors = (
                    self._embedder.embed_documents(
                        tuple(document.embedding_content for document in new_documents),
                        on_batch=on_batch,
                    )
                    if new_documents
                    else ()
                )
                if len(vectors) != len(new_documents):
                    raise SemanticServiceError(
                        "Cohere embedder returned an unexpected vector count"
                    )
                chunks_by_session = _chunks_by_session(new_documents, vectors)
                for session_id in changed:
                    current_document_ids = tuple(
                        document.id
                        for document in batches_by_session[session_id].documents
                    )
                    index.synchronize_session(
                        session_id,
                        source_revisions[session_id],
                        current_document_ids,
                        chunks_by_session.get(session_id, ()),
                    )
                    _emit_progress(
                        on_progress, SemanticSessionSynchronized(session_id=session_id)
                    )
                if deleted:
                    index.delete_sessions(deleted)
                    _emit_progress(
                        on_progress, SemanticSessionsDeleted(session_ids=deleted)
                    )

                source_changed = source_fingerprint(self.source_database) != before
                if not source_changed:
                    index.record_fingerprint(before)
                return SemanticSyncResult(
                    no_op=False,
                    indexed_session_ids=changed,
                    deleted_session_ids=deleted,
                    source_changed=source_changed,
                )
        except _ProgressCallbackError as error:
            raise error.error from error
        except SemanticServiceError:
            raise
        except OpencodePowerPackError:
            raise
        except (
            OSError,
            sqlite3.Error,
            SemanticIndexError,
        ) as error:
            raise SemanticServiceError(
                f"semantic index operation failed: {error}"
            ) from error

    def search(
        self,
        text: str,
        *,
        filters: SessionFilter = _DEFAULT_FILTERS,
        sources: Iterable[SearchSource] = _DEFAULT_SOURCES,
        roles: Iterable[str] = (),
        snippet_length: int = 200,
        limit: int | None = None,
        rerank: bool = True,
    ) -> tuple[SemanticSearchMatch, ...]:
        """Synchronize lazily, then rank eligible sessions against ``text``.

        Chunks are pre-ranked by cosine similarity. With ``rerank``, the best
        ``SEMANTIC_RERANK_CANDIDATES`` of them are rescored by Cohere Rerank and
        each session is ranked by its best reranked chunk; otherwise sessions
        keep their cosine order. ``limit`` defaults to ``SEMANTIC_RESULT_LIMIT``.
        """
        self.sync()
        limit = SEMANTIC_RESULT_LIMIT if limit is None else limit
        if limit == 0:
            return ()
        selected_sources = frozenset(sources)
        if not selected_sources or not selected_sources <= _SUPPORTED_SOURCES:
            return ()
        selected_roles = frozenset(roles)
        try:
            with _open_database(self.source_database) as database:
                sessions = database.list_sessions(ListSessionsQuery(filters=filters))
            sessions_by_id = {session.id: session for session in sessions}
            if not sessions_by_id:
                return ()
            query = self._embedder.embed_query(text)
            with SemanticIndex.open(self.sidecar_path) as index:
                results = index.search(
                    query,
                    session_ids=sessions_by_id,
                    sources=(source.value for source in selected_sources),
                    roles=(*selected_roles, None) if selected_roles else None,
                    chunks_per_session=SEMANTIC_CHUNKS_PER_SESSION,
                    limit_sessions=None if rerank else limit,
                )
            ranked = (
                self._reranked(text, results)
                if rerank
                else [
                    (result.score, [hit.chunk for hit in result.hits])
                    for result in results
                ]
            )
            return tuple(
                SemanticSearchMatch(
                    session=sessions_by_id[chunks[0].session_id],
                    source=SearchSource(chunks[0].source),
                    role=chunks[0].role,
                    snippet=_snippet(chunks[0].content, snippet_length),
                    score=score,
                    match_count=len(chunks),
                )
                for score, chunks in ranked[:limit]
                if chunks[0].session_id in sessions_by_id
            )
        except SemanticServiceError:
            raise
        except OpencodePowerPackError:
            raise
        except (
            OSError,
            sqlite3.Error,
            SemanticIndexError,
        ) as error:
            raise SemanticServiceError(f"semantic search failed: {error}") from error

    def _reranked(
        self, text: str, results: Sequence[SessionSearchResult]
    ) -> list[tuple[float, list[SemanticChunk]]]:
        """Rerank the best cosine candidates and group them by session.

        Each session is scored by its best reranked chunk, and its chunks are
        listed best first.
        """
        candidates = sorted(
            (hit for result in results for hit in result.hits),
            key=lambda hit: -hit.score,
        )[:SEMANTIC_RERANK_CANDIDATES]
        scores = self._embedder.rerank(text, [hit.chunk.content for hit in candidates])
        grouped: dict[str, tuple[float, list[SemanticChunk]]] = {}
        for item in scores:
            chunk = candidates[item.index].chunk
            grouped.setdefault(chunk.session_id, (item.score, []))[1].append(chunk)
        return list(grouped.values())


def _serialize_revision(value: int | None) -> str:
    return "null" if value is None else str(value)


def _emit_progress(
    callback: SemanticSyncProgressCallback | None, event: SemanticSyncEvent
) -> None:
    if callback is None:
        return
    try:
        callback(event)
    except Exception as error:
        raise _ProgressCallbackError(error) from error


def _batches_by_session(
    batches: tuple[SessionDocumentBatch, ...],
    changed: tuple[str, ...],
    revisions: dict[str, str],
) -> dict[str, SessionDocumentBatch]:
    batches_by_session = {batch.revision.session_id: batch for batch in batches}
    if set(batches_by_session) != set(changed) or len(batches_by_session) != len(
        batches
    ):
        raise SemanticServiceError("OpenCode database returned inconsistent documents")
    for session_id, batch in batches_by_session.items():
        if _serialize_revision(batch.revision.time_updated) != revisions[session_id]:
            raise SemanticServiceError(
                "OpenCode database changed during document fetch"
            )
    return batches_by_session


def _chunks_by_session(
    documents: tuple[SemanticDocument, ...], vectors: EmbeddingBatch
) -> dict[str, tuple[SemanticChunk, ...]]:
    chunks: dict[str, list[SemanticChunk]] = {}
    for document, vector in zip(documents, vectors, strict=True):
        chunks.setdefault(document.session_id, []).append(
            SemanticChunk(
                document_id=document.id,
                session_id=document.session_id,
                source=document.source.value,
                role=document.role,
                content=document.content,
                embedding=vector,
            )
        )
    return {
        session_id: tuple(session_chunks)
        for session_id, session_chunks in chunks.items()
    }


def _open_database(path: Path) -> OpenCodeDatabase:
    from ..database.client import OpenCodeDatabase

    return OpenCodeDatabase(path)


def _snippet(content: str, length: int) -> str:
    normalized = _SNIPPET_WHITESPACE.sub(" ", content).strip()
    if length <= 0 or len(normalized) <= length:
        return normalized if length > 0 else ""
    return normalized[: length - 1] + "…" if length > 1 else "…"
