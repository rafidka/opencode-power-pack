import os
import sqlite3
from pathlib import Path

import pytest

from opencode_power_pack.semantic.cohere import COHERE_DIMENSION
from opencode_power_pack.semantic.index import (
    _SQLITE_BIND_BATCH_SIZE,
    SemanticChunk,
    SemanticIndex,
    SemanticIndexError,
    SourceFingerprint,
    resolve_sidecar_path,
    source_fingerprint,
)


def vector(first: float, second: float = 0.0) -> tuple[float, ...]:
    return (first, second, *((0.0,) * (COHERE_DIMENSION - 2)))


def chunk(
    document_id: str,
    session_id: str,
    vector: tuple[float, ...],
    *,
    role: str | None = "user",
) -> SemanticChunk:
    return SemanticChunk(
        document_id=document_id,
        session_id=session_id,
        source="message",
        role=role,
        content=document_id,
        embedding=vector,
    )


def test_sidecar_path_is_stable_and_source_fingerprint_tracks_wal(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.db"
    source.write_bytes(b"main")
    first = resolve_sidecar_path(source, cache_home=tmp_path / "cache")
    assert first == resolve_sidecar_path(
        source.absolute(), cache_home=tmp_path / "cache"
    )
    assert first.parent == tmp_path / "cache" / "opencode-power-pack"
    before = source_fingerprint(source)
    source.write_bytes(b"changed main")
    assert source_fingerprint(source).main != before.main
    wal = Path(f"{source}-wal")
    wal.write_bytes(b"wal")
    os.utime(wal, ns=(1, 2))
    after = source_fingerprint(source)
    assert not before.wal.exists
    assert after.wal.exists and after != before


def test_source_fingerprint_ignores_shm_reader_lock_activity(tmp_path: Path) -> None:
    source = tmp_path / "wal-source.db"
    writer = sqlite3.connect(source)
    writer.execute("PRAGMA journal_mode = WAL")
    writer.execute("CREATE TABLE records (value TEXT)")
    writer.execute("INSERT INTO records VALUES ('value')")
    writer.commit()
    before = source_fingerprint(source)

    reader = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    reader.execute("SELECT * FROM records").fetchall()
    reader.close()

    assert source_fingerprint(source) == before
    writer.close()


def test_identity_roundtrip_mismatch_and_fingerprint_recording(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    source = tmp_path / "source.db"
    source.write_bytes(b"data")
    with SemanticIndex.open(path) as index:
        fingerprint = source_fingerprint(source)
        assert index.stored_fingerprint() is None
        index.record_fingerprint(fingerprint)
        assert index.status().source_fingerprint == fingerprint
    connection = sqlite3.connect(path)
    connection.execute("UPDATE metadata SET value = '1' WHERE key = 'schema_version'")
    connection.commit()
    connection.close()
    with pytest.raises(SemanticIndexError, match="schema"):
        SemanticIndex.open(path)
    with SemanticIndex.open(path, replace_identity=True) as index:
        assert index.contract.dimension == COHERE_DIMENSION
        assert index.indexed_revisions() == {}


def test_stored_fingerprints_survive_their_recorded_form(tmp_path: Path) -> None:
    # Written by an earlier release: key order differs and the keys are the field
    # names, which is the private cache format this must keep reading.
    recorded = (
        '{"main":{"exists":true,"mtime_ns":7,"size":4},'
        '"wal":{"exists":false,"mtime_ns":null,"size":null}}'
    )
    restored = SourceFingerprint.from_json(recorded)
    assert restored is not None
    assert restored.main.mtime_ns == 7
    assert restored.wal.exists is False
    assert SourceFingerprint.from_json(restored.to_json()) == restored


def test_unreadable_stored_fingerprint_provokes_a_rescan(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    source = tmp_path / "source.db"
    source.write_bytes(b"data")
    with SemanticIndex.open(path) as index:
        index.record_fingerprint(source_fingerprint(source))
    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE metadata SET value = ? WHERE key = 'source_fingerprint'",
        ('{"main":{"exists":true}}',),
    )
    connection.commit()
    connection.close()

    with SemanticIndex.open(path) as index:
        # Absent rather than fatal: an unusable fingerprint only costs a rescan.
        assert index.stored_fingerprint() is None
        assert index.status().source_fingerprint is None


def test_vectors_roundtrip_rank_filter_and_deterministic_ties(tmp_path: Path) -> None:
    with SemanticIndex.open(tmp_path / "index.sqlite") as index:
        index.replace_session(
            "a",
            "1",
            [
                chunk("a-first", "a", vector(1)),
                chunk("a-second", "a", vector(1), role=None),
            ],
        )
        index.replace_session("b", "1", [chunk("b-first", "b", vector(1))])
        results = index.search(vector(1), chunks_per_session=2)
        assert [item.session_id for item in results] == ["a", "b"]
        assert [hit.chunk.document_id for hit in results[0].hits] == [
            "a-first",
            "a-second",
        ]
        assert index.search(vector(1), roles=[None])[0].hits[0].chunk.role is None
        assert [
            item.session_id for item in index.search(vector(1), sources=["missing"])
        ] == []
        assert index.search(vector(0))[0].score == 0.0
        with pytest.raises(SemanticIndexError, match="dimension"):
            index.search((1,))
        with pytest.raises(SemanticIndexError, match="non-finite"):
            index.replace_session(
                "bad", "1", [chunk("bad", "bad", vector(float("nan")))]
            )


def test_replace_is_atomic_delete_and_rebuild_clear_state(tmp_path: Path) -> None:
    with SemanticIndex.open(tmp_path / "index.sqlite") as index:
        index.replace_session("session", "old", [chunk("old", "session", vector(1))])
        index.replace_session("other", "1", [chunk("owned", "other", vector(0, 1))])
        with pytest.raises(SemanticIndexError, match="target session"):
            index.replace_session("session", "new", [chunk("new", "other", vector(1))])
        with pytest.raises(sqlite3.IntegrityError):
            index.replace_session(
                "session", "new", [chunk("owned", "session", vector(1))]
            )
        assert index.indexed_revisions() == {"other": "1", "session": "old"}
        assert index.search(vector(1))[0].hits[0].chunk.document_id == "old"
        index.delete_sessions(["session", "other"])
        assert index.status().chunk_count == 0
        index.replace_session("session", "new", [chunk("new", "session", vector(1))])
        index.rebuild()
        assert index.indexed_revisions() == {}
        assert index.status().chunk_count == 0


def test_document_lookup_splits_ids_at_sqlite_bind_batch_limit(tmp_path: Path) -> None:
    session_ids = tuple(
        f"session-{index}" for index in range(_SQLITE_BIND_BATCH_SIZE + 1)
    )

    with SemanticIndex.open(tmp_path / "index.sqlite") as index:
        assert index.document_ids_by_session(session_ids) == {
            session_id: frozenset() for session_id in session_ids
        }
