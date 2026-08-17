"""Immutable records and deterministic chunking for safe semantic content.

Only titles and product-visible user or assistant text are represented here.
"""

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum

DOCUMENT_CHUNK_CHAR_LIMIT = 3_000
"""Maximum characters in one embedding chunk, below Cohere's input limits."""

DOCUMENT_CHUNK_OVERLAP_CHARS = 200
"""Characters retained from the preceding chunk to preserve nearby context."""


class SemanticDocumentSource(StrEnum):
    """Safe OpenCode record kinds eligible for remote embedding."""

    TITLE = "title"
    TEXT = "text"


@dataclass(frozen=True, slots=True)
class SessionRevision:
    """The current session revision, represented by OpenCode's update timestamp."""

    session_id: str
    time_updated: int | None


@dataclass(frozen=True, slots=True)
class SemanticDocument:
    """One stable, safe-to-embed chunk from an OpenCode session record.

    ``content`` is the product-visible text for display. ``embedding_content`` is
    kept separate so callers may later add embedding-only context without changing
    displayed content or this record's source metadata.
    """

    id: str
    session_id: str
    source: SemanticDocumentSource
    role: str | None
    source_record_id: str
    source_ordinal: int
    chunk_ordinal: int
    content: str
    embedding_content: str


@dataclass(frozen=True, slots=True)
class SessionDocumentBatch:
    """Current revision and all safe semantic chunks for one selected session."""

    revision: SessionRevision
    documents: tuple[SemanticDocument, ...]


def chunk_text(text: str) -> tuple[str, ...]:
    """Split text at whitespace near 3,000 characters with a 200-char overlap.

    Short text is returned unchanged. Long whitespace-free tokens are split at the
    limit, ensuring each iteration advances even for pathological input.
    """
    if not text:
        return ()
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        limit = min(start + DOCUMENT_CHUNK_CHAR_LIMIT, length)
        end = limit
        if limit < length:
            boundary = _whitespace_boundary(text, start, limit)
            if boundary is not None:
                end = boundary
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = max(end - DOCUMENT_CHUNK_OVERLAP_CHARS, start + 1)
    return tuple(chunks)


def make_semantic_documents(
    *,
    session_id: str,
    source: SemanticDocumentSource,
    role: str | None,
    source_record_id: str,
    source_ordinal: int,
    content: str,
) -> tuple[SemanticDocument, ...]:
    """Create stable chunks for one source record using content-derived IDs."""
    return tuple(
        SemanticDocument(
            id=_document_id(
                session_id=session_id,
                source=source,
                role=role,
                source_record_id=source_record_id,
                source_ordinal=source_ordinal,
                chunk_ordinal=chunk_ordinal,
                content=chunk,
            ),
            session_id=session_id,
            source=source,
            role=role,
            source_record_id=source_record_id,
            source_ordinal=source_ordinal,
            chunk_ordinal=chunk_ordinal,
            content=chunk,
            embedding_content=chunk,
        )
        for chunk_ordinal, chunk in enumerate(chunk_text(content))
    )


def _whitespace_boundary(text: str, start: int, limit: int) -> int | None:
    for index in range(limit - 1, start, -1):
        if text[index].isspace():
            return index
    return None


def _document_id(
    *,
    session_id: str,
    source: SemanticDocumentSource,
    role: str | None,
    source_record_id: str,
    source_ordinal: int,
    chunk_ordinal: int,
    content: str,
) -> str:
    identity = json.dumps(
        (
            session_id,
            source,
            role,
            source_record_id,
            source_ordinal,
            chunk_ordinal,
            content,
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"semantic:{hashlib.sha256(identity.encode()).hexdigest()}"
