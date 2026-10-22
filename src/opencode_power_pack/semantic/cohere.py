"""Cohere Embed and Rerank API v2 integration."""

import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

import cohere
import httpx
from cohere.core import ApiError, ParsingError
from pydantic import ValidationError

from ..errors import EmbeddingError
from ..models import ValueModel

COHERE_MODEL = "embed-v5.0-fast"
COHERE_DIMENSION = 1024
COHERE_MAX_TEXTS_PER_REQUEST = 96
COHERE_RERANK_MODEL = "rerank-v5.0-fast"

_MAX_PROVIDER_MESSAGE_LENGTH = 240
_SENSITIVE_VALUE = re.compile(
    r"(?i)\b(?:api[-_ ]?key|(?:access[-_ ]?)?token|auth(?:orization)?|bearer|"
    r"secret|password)\b\s*(?::|=|\s)\s*['\"]?[A-Za-z0-9._~+/=-]{6,}"
)
_UNSAFE_PROVIDER_MESSAGE = re.compile(
    r"(?i)\b(?:input|texts?|documents?|prompts?|content|payload|request body)\b|"
    r"https?://|\b[\w.+-]+@[\w.-]+\b"
)

type EmbeddingVector = tuple[float, ...]
type EmbeddingBatch = tuple[EmbeddingVector, ...]


class RerankScore(ValueModel):
    """The relevance of one reranked document, identified by its input position."""

    index: int
    score: float


class CohereDocumentBatchCompleted(ValueModel):
    """Progress reported after one document embedding batch succeeds."""

    batch_number: int
    batch_count: int
    document_count: int
    completed_documents: int
    total_documents: int


type CohereDocumentBatchCallback = Callable[[CohereDocumentBatchCompleted], None]


def document_batch_count(count: int) -> int:
    """Return the number of Cohere document requests needed for ``count`` texts."""
    if count < 0:
        raise ValueError("document count must not be negative")
    return (count + COHERE_MAX_TEXTS_PER_REQUEST - 1) // COHERE_MAX_TEXTS_PER_REQUEST


class CohereEmbeddingContract(ValueModel):
    """The fixed vector space stored in semantic sidecars."""

    model: str = COHERE_MODEL
    dimension: int = COHERE_DIMENSION


class _CohereClient(Protocol):
    def embed(
        self,
        *,
        model: str,
        input_type: str,
        texts: Sequence[str],
        output_dimension: int,
        embedding_types: Sequence[str],
        truncate: str,
    ) -> object: ...

    def rerank(
        self, *, model: str, query: str, documents: Sequence[str], top_n: int
    ) -> object: ...


class CohereEmbedProvider:
    """Embed and rerank with Cohere's v2 Embed and Rerank APIs."""

    contract = CohereEmbeddingContract()

    def __init__(
        self, *, api_key: str | None = None, client: _CohereClient | None = None
    ) -> None:
        if client is None:
            key = api_key if api_key is not None else os.environ.get("COHERE_API_KEY")
            if not key:
                raise EmbeddingError("COHERE_API_KEY is required for Cohere embeddings")
            client = cohere.ClientV2(api_key=key, client_name="opencode-power-pack")
        self._client = client

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        on_batch: CohereDocumentBatchCallback | None = None,
    ) -> EmbeddingBatch:
        """Embed documents in API-sized batches using ``search_document``."""
        documents = self._validate_texts(texts, "documents")
        vectors: list[EmbeddingVector] = []
        batch_count = document_batch_count(len(documents))
        for batch_number, start in enumerate(
            range(0, len(documents), COHERE_MAX_TEXTS_PER_REQUEST), start=1
        ):
            batch_documents = documents[start : start + COHERE_MAX_TEXTS_PER_REQUEST]
            batch_vectors = self._embed(batch_documents, input_type="search_document")
            vectors.extend(batch_vectors)
            if on_batch is not None:
                on_batch(
                    CohereDocumentBatchCompleted(
                        batch_number=batch_number,
                        batch_count=batch_count,
                        document_count=len(batch_documents),
                        completed_documents=len(vectors),
                        total_documents=len(documents),
                    )
                )
        return tuple(vectors)

    def embed_query(self, text: str) -> EmbeddingVector:
        """Embed one query using ``search_query``."""
        if type(text) is not str:
            raise EmbeddingError("embedding query must be a string")
        return self._embed((text,), input_type="search_query")[0]

    def rerank(self, query: str, documents: Sequence[str]) -> tuple[RerankScore, ...]:
        """Score every document against ``query``, most relevant first."""
        if type(query) is not str:
            raise EmbeddingError("rerank query must be a string")
        texts = self._validate_texts(documents, "rerank documents")
        if not texts:
            return ()
        response = self._call(
            "rerank",
            lambda: self._client.rerank(
                model=COHERE_RERANK_MODEL,
                query=query,
                documents=texts,
                top_n=len(texts),
            ),
        )
        return self._parse_rerank(response, len(texts))

    def _embed(self, texts: Sequence[str], *, input_type: str) -> EmbeddingBatch:
        response = self._call(
            "embedding",
            lambda: self._client.embed(
                model=COHERE_MODEL,
                input_type=input_type,
                texts=texts,
                output_dimension=COHERE_DIMENSION,
                embedding_types=("float",),
                truncate="END",
            ),
        )
        return self._parse_vectors(response, len(texts))

    @classmethod
    def _call(cls, operation: str, request: Callable[[], object]) -> object:
        """Run one Cohere request, turning expected failures into safe errors."""
        try:
            return request()
        except ApiError as error:
            raise EmbeddingError(cls._api_error_message(error, operation)) from error
        except httpx.TimeoutException as error:
            raise EmbeddingError(
                f"Cohere {operation} request timed out; retry later"
            ) from error
        except httpx.RequestError as error:
            raise EmbeddingError(
                f"Cohere {operation} network request failed; "
                "check your connection and retry"
            ) from error
        except (ParsingError, json.JSONDecodeError, ValidationError) as error:
            raise EmbeddingError(
                f"Cohere {operation} API returned an unreadable response"
            ) from error

    @staticmethod
    def _api_error_message(error: ApiError, operation: str = "embedding") -> str:
        status = error.status_code if type(error.status_code) is int else None
        status_text = str(status) if status is not None else "status unavailable"
        if status == 401:
            return (
                f"Cohere authentication failed (HTTP {status_text}): "
                "check COHERE_API_KEY"
            )
        if status == 403:
            return (
                f"Cohere permission denied (HTTP {status_text}): "
                "check API key permissions"
            )
        if status == 429:
            return f"Cohere rate limit exceeded (HTTP {status_text}): retry later"
        if status == 400:
            message = CohereEmbedProvider._provider_message(error.body)
            suffix = f": {message}" if message is not None else ""
            return (
                f"Cohere {operation} request was invalid (HTTP {status_text}){suffix}"
            )
        if status == 422:
            return f"Cohere {operation} request failed validation (HTTP {status_text})"
        if status is not None and 500 <= status <= 599:
            return f"Cohere {operation} service error (HTTP {status_text}): retry later"
        message = CohereEmbedProvider._provider_message(error.body)
        suffix = f": {message}" if message is not None else ""
        result = "embeddings" if operation == "embedding" else f"{operation} results"
        return f"unable to obtain Cohere {result} (HTTP {status_text}){suffix}"

    @staticmethod
    def _provider_message(body: object) -> str | None:
        if not isinstance(body, Mapping):
            return None
        body_mapping = cast(Mapping[str, object], body)
        message = body_mapping.get("message")
        if not isinstance(message, str):
            error = body_mapping.get("error")
            if isinstance(error, Mapping):
                message = cast(Mapping[str, object], error).get("message")
        if not isinstance(message, str):
            return None
        sanitized = "".join(
            " " if character in "\r\n\t" else character
            for character in message
            if ord(character) >= 32 or character in "\r\n\t"
        )
        sanitized = " ".join(sanitized.split())[:_MAX_PROVIDER_MESSAGE_LENGTH]
        if not sanitized or _UNSAFE_PROVIDER_MESSAGE.search(sanitized):
            return None
        return _SENSITIVE_VALUE.sub("[REDACTED]", sanitized)

    def _parse_vectors(self, response: object, expected_count: int) -> EmbeddingBatch:
        embeddings = getattr(response, "embeddings", None)
        raw_vectors = getattr(embeddings, "float_", None)
        if not isinstance(raw_vectors, list):
            raise EmbeddingError(
                "Cohere embedding API returned an unexpected vector count"
            )
        vector_objects = cast(list[object], raw_vectors)
        if len(vector_objects) != expected_count:
            raise EmbeddingError(
                "Cohere embedding API returned an unexpected vector count"
            )
        vectors: list[EmbeddingVector] = []
        for raw_vector in vector_objects:
            if not isinstance(raw_vector, list):
                raise EmbeddingError(
                    "Cohere embedding API returned an invalid vector dimension"
                )
            values = cast(list[object], raw_vector)
            if len(values) != COHERE_DIMENSION:
                raise EmbeddingError(
                    "Cohere embedding API returned an invalid vector dimension"
                )
            vector: list[float] = []
            for value in values:
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise EmbeddingError(
                        "Cohere embedding API returned a non-numeric vector"
                    )
                number = float(value)
                if not math.isfinite(number):
                    raise EmbeddingError(
                        "Cohere embedding API returned a non-finite vector"
                    )
                vector.append(number)
            vectors.append(tuple(vector))
        return tuple(vectors)

    @staticmethod
    def _parse_rerank(response: object, expected_count: int) -> tuple[RerankScore, ...]:
        results = getattr(response, "results", None)
        if not isinstance(results, list):
            raise EmbeddingError(
                "Cohere rerank API returned an unexpected result count"
            )
        scores: list[RerankScore] = []
        for result in cast(list[object], results):
            index = getattr(result, "index", None)
            score = getattr(result, "relevance_score", None)
            if (
                type(index) is not int
                or not 0 <= index < expected_count
                or isinstance(score, bool)
                or not isinstance(score, int | float)
                or not math.isfinite(score)
            ):
                raise EmbeddingError("Cohere rerank API returned an invalid result")
            scores.append(RerankScore(index=index, score=float(score)))
        if len(scores) != expected_count or len({s.index for s in scores}) != len(
            scores
        ):
            raise EmbeddingError(
                "Cohere rerank API returned an unexpected result count"
            )
        return tuple(sorted(scores, key=lambda item: (-item.score, item.index)))

    @staticmethod
    def _validate_texts(texts: Sequence[str], label: str) -> tuple[str, ...]:
        if type(texts) is str:
            raise EmbeddingError(f"embedding {label} must be a sequence of strings")
        validated = tuple(cast(Sequence[object], texts))
        if any(not isinstance(text, str) for text in validated):
            raise EmbeddingError(f"embedding {label} must be a sequence of strings")
        return cast(tuple[str, ...], validated)
