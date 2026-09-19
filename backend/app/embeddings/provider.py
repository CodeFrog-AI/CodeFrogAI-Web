"""Embedding provider abstraction and an OpenAI-compatible HTTP implementation."""

from typing import Protocol

import httpx

from app.core.config import get_settings
from app.embeddings import EMBEDDING_DIMENSIONS

REQUEST_TIMEOUT_SECONDS = 30.0


class EmbeddingError(RuntimeError):
    """An embedding request failed; messages never include keys, inputs, or provider bodies."""


class EmbeddingNotConfiguredError(EmbeddingError):
    """No embedding API key is configured."""


class EmbeddingProvider(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per input text, in order."""
        ...


class OpenAIEmbeddingProvider:
    """Call an OpenAI-compatible `/embeddings` endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
            transport=transport,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, object] = {"model": self.model, "input": texts}
        if self.model.startswith("text-embedding-3"):
            payload["dimensions"] = EMBEDDING_DIMENSIONS
        try:
            response = self._client.post("/embeddings", json=payload)
        except httpx.HTTPError:
            raise EmbeddingError("Embedding request failed") from None
        if response.status_code >= 400:
            raise EmbeddingError("Embedding provider rejected the request")
        try:
            items = sorted(response.json()["data"], key=lambda item: item["index"])
            vectors = [[float(value) for value in item["embedding"]] for item in items]
        except (ValueError, KeyError, TypeError):
            raise EmbeddingError("Embedding provider returned an invalid response") from None
        if len(vectors) != len(texts) or any(len(v) != EMBEDDING_DIMENSIONS for v in vectors):
            raise EmbeddingError("Embedding provider returned unexpected vectors")
        return vectors


def get_embedding_provider() -> EmbeddingProvider:
    """Build the configured provider, or raise if no API key is set."""

    settings = get_settings()
    if settings.embedding_api_key is None:
        raise EmbeddingNotConfiguredError("Embedding provider is not configured")
    return OpenAIEmbeddingProvider(
        settings.embedding_api_key.get_secret_value(),
        settings.embedding_model,
        settings.embedding_base_url,
    )
