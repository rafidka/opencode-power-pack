import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from cohere.core.api_error import ApiError
from opencode_v2 import insert_message, insert_session, user

from opencode_power_pack.errors import EmbeddingError
from opencode_power_pack.models import SearchSource, SessionFilter
from opencode_power_pack.semantic.cohere import (
    COHERE_DIMENSION,
    COHERE_MAX_TEXTS_PER_REQUEST,
    CohereEmbedProvider,
)
from opencode_power_pack.semantic.index import SemanticIndex, source_fingerprint
from opencode_power_pack.semantic.service import (
    SemanticEmbeddingBatchCompleted,
    SemanticService,
    SemanticServiceError,
    SemanticSessionsDeleted,
    SemanticSessionSynchronized,
    SemanticSyncPlanned,
)


class FakeCohereClient:
    def __init__(self) -> None:
        self.document_calls: list[tuple[str, ...]] = []
        self.query_calls: list[str] = []
        self.rerank_calls: list[tuple[str, tuple[str, ...]]] = []
        self.fail = False
        self.error: Exception | None = None
        self.fail_after_document_calls: int | None = None
        self.invalid_vectors = False

    def embed(self, **kwargs: object) -> object:
        if self.error is not None:
            raise self.error
        if self.fail:
            raise ApiError(body="provider failed")
        texts = tuple(kwargs["texts"])
        if kwargs["input_type"] == "search_document":
            if (
                self.fail_after_document_calls is not None
                and len(self.document_calls) >= self.fail_after_document_calls
            ):
                raise ApiError(body="provider failed")
            self.document_calls.append(texts)
        else:
            self.query_calls.append(texts[0])
        vectors = (
            [[1.0] for _ in texts]
            if self.invalid_vectors
            else [list(self._vector(text)) for text in texts]
        )
        return SimpleNamespace(embeddings=SimpleNamespace(float_=vectors))

    def rerank(self, **kwargs: object) -> object:
        documents = tuple(kwargs["documents"])
        self.rerank_calls.append((str(kwargs["query"]), documents))
        scores = [self._relevance(document) for document in documents]
        order = sorted(range(len(documents)), key=lambda index: -scores[index])
        return SimpleNamespace(
            results=[
                SimpleNamespace(index=index, relevance_score=scores[index])
                for index in order[: int(kwargs["top_n"])]
            ]
        )

    @staticmethod
    def _relevance(document: str) -> float:
        """Prefer assistant text, so reranking visibly disagrees with cosine order."""
        if "assistant" in document:
            return 0.9
        return 0.5 if "needle" in document.casefold() else 0.1

    @staticmethod
    def _vector(text: str) -> tuple[float, ...]:
        return ((1.0, 0.0) if "needle" in text.casefold() else (0.0, 1.0)) + (
            (0.0,) * (COHERE_DIMENSION - 2)
        )


class FakeProvider(CohereEmbedProvider):
    def __init__(self) -> None:
        self.client = FakeCohereClient()
        super().__init__(client=self.client)

    @property
    def document_calls(self) -> list[tuple[str, ...]]:
        return self.client.document_calls

    @property
    def query_calls(self) -> list[str]:
        return self.client.query_calls


def test_sync_progress_orders_plan_batches_and_commits(
    opencode_database: Path, tmp_path: Path
) -> None:
    service = SemanticService(
        opencode_database, FakeProvider(), tmp_path / "index.sqlite"
    )
    events: list[object] = []

    result = service.sync(on_progress=events.append)

    assert isinstance(events[0], SemanticSyncPlanned)
    plan = events[0]
    assert plan.changed_session_count == result.indexed_session_count == 5
    assert plan.deleted_session_count == 0
    batch_events = [
        event for event in events if isinstance(event, SemanticEmbeddingBatchCompleted)
    ]
    assert plan.document_count == sum(event.document_count for event in batch_events)
    assert plan.batch_count == len(batch_events) > 0
    assert [event.batch_number for event in batch_events] == list(
        range(1, plan.batch_count + 1)
    )
    assert [event.completed_documents for event in batch_events] == [
        min((number + 1) * COHERE_MAX_TEXTS_PER_REQUEST, plan.document_count)
        for number in range(plan.batch_count)
    ]
    first_commit = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, SemanticSessionSynchronized)
    )
    assert all(
        isinstance(event, SemanticEmbeddingBatchCompleted)
        for event in events[1:first_commit]
    )
    assert [event.session_id for event in events[first_commit:]] == list(
        result.indexed_session_ids
    )


def test_sync_progress_is_silent_for_fingerprint_no_op(
    opencode_database: Path, tmp_path: Path
) -> None:
    service = SemanticService(
        opencode_database, FakeProvider(), tmp_path / "index.sqlite"
    )
    service.sync()
    events: list[object] = []

    assert service.sync(on_progress=events.append).no_op
    assert events == []


def test_sync_progress_reports_metadata_and_deletion_only_work(
    opencode_database: Path, tmp_path: Path
) -> None:
    service = SemanticService(
        opencode_database, FakeProvider(), tmp_path / "index.sqlite"
    )
    service.sync()
    connection = sqlite3.connect(opencode_database)
    connection.execute(
        "UPDATE session_v2 SET agent = ?, time_updated = ? WHERE id = ?",
        ("review", 7_000, "session-root"),
    )
    connection.commit()
    connection.close()
    metadata_events: list[object] = []

    service.sync(on_progress=metadata_events.append)

    assert metadata_events == [
        SemanticSyncPlanned(
            changed_session_count=1,
            deleted_session_count=0,
            document_count=0,
            batch_count=0,
        ),
        SemanticSessionSynchronized(session_id="session-root"),
    ]

    connection = sqlite3.connect(opencode_database)
    connection.execute("DELETE FROM session_v2 WHERE id = ?", ("session-other",))
    connection.commit()
    connection.close()
    deletion_events: list[object] = []

    service.sync(on_progress=deletion_events.append)

    assert deletion_events == [
        SemanticSyncPlanned(
            changed_session_count=0,
            deleted_session_count=1,
            document_count=0,
            batch_count=0,
        ),
        SemanticSessionsDeleted(session_ids=("session-other",)),
    ]


def test_sync_progress_stops_before_commits_when_provider_batch_fails(
    opencode_database: Path, tmp_path: Path
) -> None:
    provider = FakeProvider()
    service = SemanticService(opencode_database, provider, tmp_path / "index.sqlite")
    service.sync()
    provider.client.document_calls.clear()
    connection = sqlite3.connect(opencode_database)
    for number in range(3):
        insert_message(
            connection,
            "session-root",
            f"message-progress-{number}",
            "user",
            user(f"Progress needle {number}"),
            created=7_000,
        )
    connection.execute(
        "UPDATE session_v2 SET time_updated = ? WHERE id = ?", (7_000, "session-root")
    )
    connection.commit()
    connection.close()
    provider.client.fail_after_document_calls = 0
    events: list[object] = []

    with pytest.raises(EmbeddingError, match="unable to obtain Cohere embeddings"):
        service.sync(on_progress=events.append)

    assert isinstance(events[0], SemanticSyncPlanned)
    batch_events = [
        event for event in events if isinstance(event, SemanticEmbeddingBatchCompleted)
    ]
    assert batch_events == []
    assert not any(isinstance(event, SemanticSessionSynchronized) for event in events)


def test_sync_progress_callback_error_stops_before_index_mutation(
    opencode_database: Path, tmp_path: Path
) -> None:
    sidecar = tmp_path / "index.sqlite"
    service = SemanticService(opencode_database, FakeProvider(), sidecar)
    service.sync()
    with SemanticIndex.open(sidecar) as index:
        previous_fingerprint = index.stored_fingerprint()
    connection = sqlite3.connect(opencode_database)
    connection.execute(
        "UPDATE session_v2 SET agent = ?, time_updated = ? WHERE id = ?",
        ("review", 7_000, "session-root"),
    )
    connection.commit()
    connection.close()

    def fail(_: object) -> None:
        raise TypeError("stop progress")

    with pytest.raises(TypeError, match="stop progress"):
        service.sync(on_progress=fail)
    with SemanticIndex.open(sidecar) as index:
        assert index.stored_fingerprint() == previous_fingerprint
        assert index.indexed_revisions()["session-root"] != "7000"


def test_search_forwards_sync_progress(opencode_database: Path, tmp_path: Path) -> None:
    service = SemanticService(
        opencode_database, FakeProvider(), tmp_path / "index.sqlite"
    )
    original_sync = service.sync
    sync_calls: list[dict[str, object]] = []

    def sync(**kwargs: object) -> object:
        sync_calls.append(kwargs)
        return original_sync(**kwargs)

    service.sync = sync  # type: ignore[method-assign]

    service.search("needle", limit=0)
    service.search("needle", limit=0, on_progress=print)

    assert sync_calls == [{"on_progress": None}, {"on_progress": print}]


def test_sync_is_incremental_removes_sessions_and_rebuilds(
    opencode_database: Path, tmp_path: Path
) -> None:
    provider = FakeProvider()
    service = SemanticService(opencode_database, provider, tmp_path / "semantic.sqlite")

    initial = service.sync()
    assert initial.indexed_session_count == 5
    assert not initial.no_op
    calls = len(provider.document_calls)
    assert service.sync().no_op
    assert len(provider.document_calls) == calls

    connection = sqlite3.connect(opencode_database)
    connection.execute(
        "UPDATE session_v2 SET title = ?, time_updated = ? WHERE id = ?",
        ("updated needle", 7_000, "session-root"),
    )
    insert_session(
        connection,
        "session-new",
        project_id="project-root",
        title="new needle",
        updated=8_000,
    )
    connection.commit()
    connection.close()

    changed = service.sync()
    assert changed.indexed_session_ids == ("session-new", "session-root")
    assert changed.deleted_session_ids == ()
    assert "Other widget needle" not in provider.document_calls[-1]

    connection = sqlite3.connect(opencode_database)
    connection.execute("DELETE FROM session_v2 WHERE id = ?", ("session-other",))
    connection.commit()
    connection.close()

    changed = service.sync()
    assert changed.indexed_session_ids == ()
    assert changed.deleted_session_ids == ("session-other",)
    rebuilt = service.sync(rebuild=True)
    assert rebuilt.indexed_session_count == 5


def test_changed_session_embeds_only_new_chunks(
    opencode_database: Path, tmp_path: Path
) -> None:
    provider = FakeProvider()
    sidecar = tmp_path / "semantic.sqlite"
    service = SemanticService(opencode_database, provider, sidecar)
    service.sync()
    provider.document_calls.clear()

    connection = sqlite3.connect(opencode_database)
    insert_message(
        connection,
        "session-root",
        "message-new",
        "user",
        user("A newly appended needle."),
        created=7_000,
    )
    connection.execute(
        "UPDATE session_v2 SET time_updated = ? WHERE id = ?", (7_000, "session-root")
    )
    connection.commit()
    connection.close()

    service.sync()

    assert provider.document_calls == [("A newly appended needle.",)]


def test_changed_text_replaces_stale_chunk_without_reembedding_history(
    opencode_database: Path, tmp_path: Path
) -> None:
    provider = FakeProvider()
    sidecar = tmp_path / "semantic.sqlite"
    service = SemanticService(opencode_database, provider, sidecar)
    service.sync()
    with SemanticIndex.open(sidecar) as index:
        old_ids = index.document_ids_by_session(["session-root"])["session-root"]
    provider.document_calls.clear()

    replacement = "Replacement message text."
    connection = sqlite3.connect(opencode_database)
    connection.execute(
        "UPDATE session_message SET data = ? WHERE id = ?",
        (json.dumps(user(replacement)), "message-user"),
    )
    connection.execute(
        "UPDATE session_v2 SET time_updated = ? WHERE id = ?", (7_000, "session-root")
    )
    connection.commit()
    connection.close()

    service.sync()

    assert provider.document_calls == [(replacement,)]
    with SemanticIndex.open(sidecar) as index:
        new_ids = index.document_ids_by_session(["session-root"])["session-root"]
    assert old_ids - new_ids
    assert len(new_ids) == len(old_ids)


def test_text_deletion_and_metadata_update_do_not_embed_documents(
    opencode_database: Path, tmp_path: Path
) -> None:
    provider = FakeProvider()
    sidecar = tmp_path / "semantic.sqlite"
    service = SemanticService(opencode_database, provider, sidecar)
    service.sync()
    with SemanticIndex.open(sidecar) as index:
        before_ids = index.document_ids_by_session(["session-root"])["session-root"]
    provider.document_calls.clear()

    connection = sqlite3.connect(opencode_database)
    connection.execute("DELETE FROM session_message WHERE id = ?", ("message-user",))
    connection.execute(
        "UPDATE session_v2 SET time_updated = ? WHERE id = ?", (7_000, "session-root")
    )
    connection.commit()
    connection.close()
    service.sync()

    assert provider.document_calls == []
    with SemanticIndex.open(sidecar) as index:
        after_deletion = index.document_ids_by_session(["session-root"])["session-root"]
        assert index.indexed_revisions()["session-root"] == "7000"
    assert len(after_deletion) == len(before_ids) - 1

    connection = sqlite3.connect(opencode_database)
    connection.execute(
        "UPDATE session_v2 SET agent = ?, time_updated = ? WHERE id = ?",
        ("review", 8_000, "session-root"),
    )
    connection.commit()
    connection.close()
    service.sync()

    assert provider.document_calls == []
    with SemanticIndex.open(sidecar) as index:
        assert index.indexed_revisions()["session-root"] == "8000"


def test_failed_new_embedding_preserves_chunks_and_revision_for_retry(
    opencode_database: Path, tmp_path: Path
) -> None:
    provider = FakeProvider()
    sidecar = tmp_path / "semantic.sqlite"
    service = SemanticService(opencode_database, provider, sidecar)
    service.sync()
    with SemanticIndex.open(sidecar) as index:
        prior_ids = index.document_ids_by_session(["session-root"])["session-root"]
        prior_revision = index.indexed_revisions()["session-root"]
    provider.document_calls.clear()

    connection = sqlite3.connect(opencode_database)
    insert_message(
        connection,
        "session-root",
        "message-retry",
        "user",
        user("Retry needle."),
        created=7_000,
    )
    connection.execute(
        "UPDATE session_v2 SET time_updated = ? WHERE id = ?", (7_000, "session-root")
    )
    connection.commit()
    connection.close()

    provider.client.fail = True
    with pytest.raises(EmbeddingError, match="unable to obtain Cohere embeddings"):
        service.sync()
    with SemanticIndex.open(sidecar) as index:
        assert index.indexed_revisions()["session-root"] == prior_revision
        assert (
            index.document_ids_by_session(["session-root"])["session-root"] == prior_ids
        )
        assert index.stored_fingerprint() != source_fingerprint(opencode_database)

    provider.client.fail = False
    service.sync()
    assert provider.document_calls == [("Retry needle.",)]


def test_search_filters_groups_snippets_and_zero_limit(
    opencode_database: Path, tmp_path: Path
) -> None:
    provider = FakeProvider()
    service = SemanticService(opencode_database, provider, tmp_path / "semantic.sqlite")

    matches = service.search(
        "needle",
        filters=SessionFilter(),
        sources=(SearchSource.TEXT,),
        roles=("user",),
        snippet_length=12,
    )
    assert [match.session.id for match in matches] == ["session-child", "session-root"]
    assert all(match.source is SearchSource.TEXT for match in matches)
    assert all(match.role == "user" for match in matches)
    assert all(len(match.snippet) <= 12 for match in matches)
    assert all(match.match_count <= 3 for match in matches)
    assert service.search("needle", limit=0) == ()
    assert provider.query_calls == ["needle"]


def test_search_reranks_cosine_candidates_and_scores_sessions_by_best_chunk(
    opencode_database: Path, tmp_path: Path
) -> None:
    provider = FakeProvider()
    service = SemanticService(opencode_database, provider, tmp_path / "semantic.sqlite")

    cosine = service.search("needle", sources=(SearchSource.TEXT,), rerank=False)
    reranked = service.search("needle", sources=(SearchSource.TEXT,))

    assert [match.session.id for match in cosine] == ["session-child", "session-root"]
    assert [match.session.id for match in reranked] == [
        "session-root",
        "session-child",
    ]
    assert reranked[0].score == 0.9
    assert reranked[0].snippet == "The assistant also mentions needle."
    assert reranked[0].match_count == 2
    assert len(provider.client.rerank_calls) == 1
    query, documents = provider.client.rerank_calls[0]
    assert query == "needle"
    assert len(documents) == len(set(documents)) > 0


def test_search_limits_candidates_and_results_with_constants(
    opencode_database: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from opencode_power_pack.semantic import service as service_module

    monkeypatch.setattr(service_module, "SEMANTIC_RESULT_LIMIT", 1)
    provider = FakeProvider()
    service = SemanticService(opencode_database, provider, tmp_path / "semantic.sqlite")

    assert len(service.search("needle")) == 1
    assert len(service.search("needle", limit=3)) == 3
    assert all(len(documents) > 2 for _, documents in provider.client.rerank_calls)

    monkeypatch.setattr(service_module, "SEMANTIC_RERANK_CANDIDATES", 2)
    provider.client.rerank_calls.clear()
    service.search("needle", limit=3)

    assert [len(documents) for _, documents in provider.client.rerank_calls] == [2]


def test_search_without_rerank_makes_no_rerank_call(
    opencode_database: Path, tmp_path: Path
) -> None:
    provider = FakeProvider()
    service = SemanticService(opencode_database, provider, tmp_path / "semantic.sqlite")

    matches = service.search("needle", rerank=False)

    assert [match.session.id for match in matches] == [
        "session-child",
        "session-other",
        "session-root",
    ]
    assert provider.client.rerank_calls == []


def test_invalid_sdk_vector_becomes_expected_error(
    opencode_database: Path, tmp_path: Path
) -> None:
    provider = FakeProvider()
    provider.client.invalid_vectors = True
    service = SemanticService(opencode_database, provider, tmp_path / "semantic.sqlite")
    with pytest.raises(EmbeddingError, match="invalid vector dimension"):
        service.sync()


def test_programmer_errors_from_provider_propagate(
    opencode_database: Path, tmp_path: Path
) -> None:
    provider = FakeProvider()
    provider.client.error = ValueError("programmer bug")

    with pytest.raises(ValueError, match="programmer bug"):
        SemanticService(
            opencode_database, provider, tmp_path / "semantic.sqlite"
        ).sync()


def test_rebuild_and_sidecar_safety(opencode_database: Path, tmp_path: Path) -> None:
    first = SemanticService(
        opencode_database, FakeProvider(), tmp_path / "index.sqlite"
    )
    first.sync()

    replacement = SemanticService(
        opencode_database, FakeProvider(), tmp_path / "index.sqlite"
    )
    assert replacement.sync(rebuild=True).indexed_session_count == 5
    with pytest.raises(SemanticServiceError, match="must not be"):
        SemanticService(opencode_database, FakeProvider(), opencode_database)
    for suffix in ("-journal", "-shm", "-wal"):
        with pytest.raises(SemanticServiceError, match="must not be"):
            SemanticService(
                opencode_database,
                FakeProvider(),
                Path(f"{opencode_database}{suffix}"),
            )
