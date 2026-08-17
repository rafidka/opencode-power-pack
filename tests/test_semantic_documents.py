import sqlite3
from pathlib import Path

from opencode_power_pack.database import OpenCodeDatabase
from opencode_power_pack.semantic.documents import (
    DOCUMENT_CHUNK_CHAR_LIMIT,
    DOCUMENT_CHUNK_OVERLAP_CHARS,
    SemanticDocumentSource,
    chunk_text,
)


def test_semantic_documents_extract_only_safe_title_and_visible_text(
    opencode_database: Path,
) -> None:
    with OpenCodeDatabase(opencode_database) as database:
        batches = database.fetch_semantic_documents({"session-root"})

    assert len(batches) == 1
    documents = batches[0].documents
    extracted = [
        (document.source, document.role, document.content) for document in documents
    ]
    assert extracted == [
        (
            SemanticDocumentSource.TITLE,
            None,
            "Fix naïve parser 100%_ready",
        ),
        (
            SemanticDocumentSource.TEXT,
            "assistant",
            "The assistant also mentions needle.",
        ),
        (
            SemanticDocumentSource.TEXT,
            "user",
            "Needle needle in user text; literal 100%_ready and 東京.",
        ),
    ]
    assert [document.source_record_id for document in documents] == [
        "title",
        "part-assistant",
        "part-user",
    ]
    content = "\n".join(document.content for document in documents)
    assert "private reasoning" not in content
    assert "tool input" not in content
    assert "tool output" not in content
    assert "Investigate needle" not in content
    assert all(document.content == document.embedding_content for document in documents)


def test_semantic_document_ids_are_stable_and_revisions_use_time_updated(
    opencode_database: Path,
) -> None:
    with OpenCodeDatabase(opencode_database) as database:
        first = database.fetch_semantic_documents({"session-root"})
        second = database.fetch_semantic_documents({"session-root"})
        revisions = database.list_semantic_session_revisions()

    assert first == second
    assert all(document.id.startswith("semantic:") for document in first[0].documents)
    assert first[0].revision.time_updated == 5_000
    revision_by_session = {
        revision.session_id: revision.time_updated for revision in revisions
    }
    assert revision_by_session["session-root"] == 5_000


def test_semantic_revisions_are_lazy_sized_and_consumed_inside_database_context(
    opencode_database: Path,
) -> None:
    with OpenCodeDatabase(opencode_database) as database:
        assert database._connection is not None
        statements: list[str] = []
        database._connection.set_trace_callback(statements.append)
        revisions = database.list_semantic_session_revisions()

        assert statements == []
        assert len(revisions) == 5
        assert [revision.session_id for revision in revisions] == [
            "session-archived",
            "session-child",
            "session-other",
            "session-root",
            "session-root-second",
        ]

    assert statements[:2] == [
        "SELECT COUNT(*) AS count FROM session",
        "SELECT id, time_updated FROM session ORDER BY id",
    ]


def test_selected_session_fetching_returns_only_requested_existing_sessions(
    opencode_database: Path,
) -> None:
    with OpenCodeDatabase(opencode_database) as database:
        assert database.fetch_semantic_documents(set()) == ()
        batches = database.fetch_semantic_documents(
            {"session-child", "missing", "session-root"}
        )

    assert [batch.revision.session_id for batch in batches] == [
        "session-child",
        "session-root",
    ]
    assert [document.content for document in batches[0].documents] == [
        "Child workspace needle",
        "child body needle",
    ]


def test_text_part_ids_preserve_unchanged_chunks_when_siblings_change_order(
    opencode_database: Path,
) -> None:
    connection = sqlite3.connect(opencode_database)
    connection.executemany(
        "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "part-alpha",
                "session-root",
                "message-user",
                "text",
                "alpha",
                '{"type":"text","text":"alpha"}',
                1,
                1,
            ),
            (
                "part-zeta",
                "session-root",
                "message-user",
                "text",
                "zeta",
                '{"type":"text","text":"zeta"}',
                1,
                1,
            ),
        ],
    )
    connection.commit()
    connection.close()

    with OpenCodeDatabase(opencode_database) as database:
        before = {
            document.content: document.id
            for document in database.fetch_semantic_documents({"session-root"})[
                0
            ].documents
        }

    connection = sqlite3.connect(opencode_database)
    connection.execute(
        "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "part-aardvark",
            "session-root",
            "message-user",
            "text",
            "aardvark",
            '{"type":"text","text":"aardvark"}',
            1,
            1,
        ),
    )
    connection.commit()
    connection.close()

    with OpenCodeDatabase(opencode_database) as database:
        inserted = {
            document.content: document.id
            for document in database.fetch_semantic_documents({"session-root"})[
                0
            ].documents
        }

    assert inserted["alpha"] == before["alpha"]
    assert inserted["zeta"] == before["zeta"]

    connection = sqlite3.connect(opencode_database)
    connection.execute(
        "UPDATE part SET text = ?, data = ? WHERE id = ?",
        ("zzzz", '{"type":"text","text":"zzzz"}', "part-alpha"),
    )
    connection.commit()
    connection.close()

    with OpenCodeDatabase(opencode_database) as database:
        after = {
            document.content: document.id
            for document in database.fetch_semantic_documents({"session-root"})[
                0
            ].documents
        }

    assert after["zeta"] == before["zeta"]


def test_chunking_is_deterministic_with_overlap_and_pathological_input() -> None:
    text = ("word " * 700) + "tail"
    first = chunk_text(text)

    assert first == chunk_text(text)
    assert len(first) > 1
    assert all(chunk and len(chunk) <= DOCUMENT_CHUNK_CHAR_LIMIT for chunk in first)
    assert (
        first[0][-DOCUMENT_CHUNK_OVERLAP_CHARS:]
        == first[1][:DOCUMENT_CHUNK_OVERLAP_CHARS]
    )
    assert chunk_text("short message") == ("short message",)
    assert chunk_text("") == ()

    token_chunks = chunk_text("x" * (DOCUMENT_CHUNK_CHAR_LIMIT + 1))
    assert token_chunks == (
        "x" * DOCUMENT_CHUNK_CHAR_LIMIT,
        "x" * (DOCUMENT_CHUNK_OVERLAP_CHARS + 1),
    )
