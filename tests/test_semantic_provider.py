import json
import re
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from cohere.core import ApiError, ParsingError
from pydantic import BaseModel, ValidationError

from opencode_power_pack.errors import EmbeddingError
from opencode_power_pack.semantic import (
    COHERE_DIMENSION,
    COHERE_MODEL,
    COHERE_RERANK_MODEL,
    CohereEmbeddingContract,
    CohereEmbedProvider,
)
from opencode_power_pack.semantic.cohere import (
    COHERE_MAX_TEXTS_PER_REQUEST,
    CohereDocumentBatchCompleted,
    document_batch_count,
)


class FakeCohereClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.requests: list[dict[str, object]] = []

    def embed(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def response(vectors: list[list[float]]) -> object:
    return SimpleNamespace(embeddings=SimpleNamespace(float_=vectors))


def vector(value: float) -> list[float]:
    return [value] * COHERE_DIMENSION


def test_documents_and_query_use_distinct_input_types() -> None:
    client = FakeCohereClient([response([vector(1)]), response([vector(3)])])
    provider = CohereEmbedProvider(client=client)

    documents = provider.embed_documents(["first"])
    query = provider.embed_query("find first")

    assert documents == (tuple(vector(1)),)
    assert query == tuple(vector(3))
    assert provider.contract == CohereEmbeddingContract()
    assert [request["input_type"] for request in client.requests] == [
        "search_document",
        "search_query",
    ]
    assert client.requests[0] == {
        "model": COHERE_MODEL,
        "input_type": "search_document",
        "texts": ("first",),
        "output_dimension": COHERE_DIMENSION,
        "embedding_types": ("float",),
        "truncate": "END",
    }


def test_documents_are_batched_at_cohere_limit() -> None:
    client = FakeCohereClient(
        [response([vector(1)] * COHERE_MAX_TEXTS_PER_REQUEST), response([vector(2)])]
    )
    provider = CohereEmbedProvider(client=client)

    vectors = provider.embed_documents([str(index) for index in range(97)])

    assert len(vectors) == 97
    request_sizes = [
        len(cast(tuple[str, ...], request["texts"])) for request in client.requests
    ]
    assert request_sizes == [96, 1]


def test_document_batch_progress_reports_validated_batches() -> None:
    documents = tuple(str(index) for index in range(193))
    client = FakeCohereClient(
        [
            response([vector(1)] * 96),
            response([vector(2)] * 96),
            response([vector(3)]),
        ]
    )
    events: list[CohereDocumentBatchCompleted] = []

    CohereEmbedProvider(client=client).embed_documents(
        documents, on_batch=events.append
    )

    assert COHERE_MAX_TEXTS_PER_REQUEST == 96
    assert document_batch_count(0) == 0
    assert document_batch_count(193) == 3
    assert [(event.batch_number, event.batch_count) for event in events] == [
        (1, 3),
        (2, 3),
        (3, 3),
    ]
    assert [event.document_count for event in events] == [96, 96, 1]
    assert all(not hasattr(event, "documents") for event in events)
    assert [event.completed_documents for event in events] == [96, 192, 193]
    assert all(event.total_documents == 193 for event in events)


def test_document_batch_progress_stops_before_failed_batch() -> None:
    client = FakeCohereClient([response([vector(1)] * 96), ApiError(status_code=429)])
    events: list[CohereDocumentBatchCompleted] = []

    with pytest.raises(EmbeddingError, match="rate limit"):
        CohereEmbedProvider(client=client).embed_documents(
            [str(index) for index in range(97)], on_batch=events.append
        )

    assert [event.batch_number for event in events] == [1]
    assert events[0].completed_documents == 96


def test_missing_api_key_is_an_expected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(EmbeddingError, match="COHERE_API_KEY"):
        CohereEmbedProvider()


def test_rejects_non_string_document_collection_and_query() -> None:
    provider = CohereEmbedProvider(client=FakeCohereClient([]))
    with pytest.raises(EmbeddingError, match="sequence of strings"):
        provider.embed_documents("one document")
    with pytest.raises(EmbeddingError, match="query must be a string"):
        provider.embed_query(1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("sdk_response", "message"),
    [
        (ApiError(body="secret SDK response"), "unable to obtain"),
        (SimpleNamespace(embeddings=SimpleNamespace(float_=[vector(1)])), "count"),
        (response([[1.0]]), "dimension"),
        (response([[float("nan")] * COHERE_DIMENSION]), "non-finite"),
    ],
)
def test_rejects_sdk_failures_and_malformed_responses(
    sdk_response: object, message: str
) -> None:
    provider = CohereEmbedProvider(client=FakeCohereClient([sdk_response]))

    with pytest.raises(EmbeddingError, match=message) as error:
        provider.embed_documents(["one", "two"] if message == "count" else ["one"])

    assert "secret SDK response" not in str(error.value)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, "authentication failed (HTTP 401): check COHERE_API_KEY"),
        (403, "permission denied (HTTP 403)"),
        (429, "rate limit exceeded (HTTP 429)"),
        (500, "service error (HTTP 500): retry later"),
        (503, "service error (HTTP 503): retry later"),
    ],
)
def test_api_error_statuses_are_actionable_and_secret_safe(
    status: int, expected: str
) -> None:
    secret = "cohere-secret-key-123456"
    provider = CohereEmbedProvider(
        client=FakeCohereClient(
            [
                ApiError(
                    status_code=status,
                    headers={"authorization": secret},
                    body={"message": secret},
                )
            ]
        )
    )

    with pytest.raises(EmbeddingError, match=re.escape(expected)) as error:
        provider.embed_documents(["private document text"])

    assert secret not in str(error.value)
    assert "private document text" not in str(error.value)


def test_invalid_and_unknown_api_errors_use_only_safe_allowlisted_messages() -> None:
    provider = CohereEmbedProvider(
        client=FakeCohereClient(
            [
                ApiError(
                    status_code=400,
                    body={"message": "Model is unavailable\nplease select another."},
                ),
                ApiError(
                    status_code=418,
                    body={
                        "error": {"message": "Token: abcdefghijklmnop"},
                        "details": "leak",
                    },
                ),
                ApiError(
                    status_code=422,
                    body={"message": "input: private document text"},
                ),
            ]
        )
    )

    with pytest.raises(
        EmbeddingError, match="Model is unavailable please select another"
    ):
        provider.embed_documents(["one"])
    with pytest.raises(EmbeddingError, match=r"HTTP 418.*\[REDACTED\]") as error:
        provider.embed_documents(["two"])
    assert "leak" not in str(error.value)
    with pytest.raises(EmbeddingError, match="HTTP 422") as error:
        provider.embed_documents(["private document text"])
    assert "private document text" not in str(error.value)


class _ValidationModel(BaseModel):
    value: int


def _validation_error() -> ValidationError:
    with pytest.raises(ValidationError) as caught:
        _ValidationModel.model_validate({"value": "not an integer"})
    return caught.value


@pytest.mark.parametrize(
    ("sdk_error", "message"),
    [
        (httpx.ConnectTimeout("timed out"), "timed out"),
        (httpx.ConnectError("offline"), "network request failed"),
        (json.JSONDecodeError("invalid JSON", "{", 1), "unreadable response"),
        (_validation_error(), "unreadable response"),
    ],
)
def test_sdk_transport_errors_are_expected_errors(
    sdk_error: Exception, message: str
) -> None:
    provider = CohereEmbedProvider(client=FakeCohereClient([sdk_error]))
    with pytest.raises(EmbeddingError, match=message):
        provider.embed_documents(["one"])


def test_sdk_parsing_error_is_an_expected_error() -> None:
    provider = CohereEmbedProvider(
        client=FakeCohereClient([ParsingError(body="private response")])
    )

    with pytest.raises(EmbeddingError, match="unreadable response") as error:
        provider.embed_documents(["private document text"])

    assert "private response" not in str(error.value)


def test_programmer_errors_are_not_recast_as_sdk_failures() -> None:
    provider = CohereEmbedProvider(client=FakeCohereClient([RuntimeError("bug")]))

    with pytest.raises(RuntimeError, match="bug"):
        provider.embed_documents(["one"])


class FakeRerankClient(FakeCohereClient):
    def rerank(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def rerank_response(*results: tuple[object, object]) -> object:
    return SimpleNamespace(
        results=[
            SimpleNamespace(index=index, relevance_score=score)
            for index, score in results
        ]
    )


def test_rerank_scores_every_document_with_the_rerank_model() -> None:
    client = FakeRerankClient([rerank_response((1, 0.9), (0, 0.2), (2, 0.2))])
    provider = CohereEmbedProvider(client=client)

    scores = provider.rerank("query", ["a", "b", "c"])

    assert [(score.index, score.score) for score in scores] == [
        (1, 0.9),
        (0, 0.2),
        (2, 0.2),
    ]
    assert client.requests == [
        {
            "model": COHERE_RERANK_MODEL,
            "query": "query",
            "documents": ("a", "b", "c"),
            "top_n": 3,
        }
    ]


def test_rerank_without_documents_makes_no_request() -> None:
    client = FakeRerankClient([])
    assert CohereEmbedProvider(client=client).rerank("query", []) == ()
    assert client.requests == []


@pytest.mark.parametrize(
    "malformed",
    [
        SimpleNamespace(results=None),
        rerank_response((0, 0.5)),
        rerank_response((0, 0.5), (0, 0.4)),
        rerank_response((0, 0.5), (5, 0.4)),
        rerank_response((0, 0.5), (1, float("nan"))),
        rerank_response((0, 0.5), (1, True)),
    ],
)
def test_rerank_rejects_malformed_responses(malformed: object) -> None:
    provider = CohereEmbedProvider(client=FakeRerankClient([malformed]))
    with pytest.raises(EmbeddingError, match="Cohere rerank API returned"):
        provider.rerank("query", ["a", "b"])


def test_rerank_api_errors_name_the_rerank_operation() -> None:
    client = FakeRerankClient(
        [ApiError(status_code=503, body={}), ApiError(status_code=418, body={})]
    )
    provider = CohereEmbedProvider(client=client)

    with pytest.raises(EmbeddingError, match="Cohere rerank service error"):
        provider.rerank("query", ["a"])
    with pytest.raises(EmbeddingError, match="unable to obtain Cohere rerank results"):
        provider.rerank("query", ["a"])
