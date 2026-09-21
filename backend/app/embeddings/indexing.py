"""Generate and store embeddings for a repository's code chunks."""

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from sqlalchemy.orm import Session, defer

from app.db.models import Repository, RepositoryChunk, RepositoryFile
from app.embeddings.provider import EmbeddingError, EmbeddingNotConfiguredError, EmbeddingProvider

logger = logging.getLogger(__name__)

BATCH_SIZE = 64
MAX_EMBEDDING_INPUT_CHARS = 16_000


@dataclass(frozen=True)
class EmbeddingIndexSummary:
    chunks_total: int
    chunks_embedded: int
    chunks_reused: int
    chunks_skipped: int


def build_embedding_text(path: str, content: str) -> str:
    """Text sent to the provider: the file path (useful semantic context) plus the code."""

    return f"{path}\n{content}"[:MAX_EMBEDDING_INPUT_CHARS]


def content_hash(model: str, text: str) -> str:
    """Identify the exact input and model an embedding was generated from."""

    return hashlib.sha256(f"{model}\0{text}".encode()).hexdigest()


def index_repository_embeddings(
    session: Session, repository: Repository, provider: EmbeddingProvider
) -> EmbeddingIndexSummary:
    """Embed chunks that are new or changed; reuse embeddings whose input is unchanged.

    Blank chunks are never embedded. Each batch is committed, so a provider failure
    keeps earlier progress and a re-run continues where it stopped.
    """

    # Vectors are large; only whether one exists is needed to decide on reuse.
    rows = (
        session.query(
            RepositoryFile.path, RepositoryChunk, RepositoryChunk.embedding.is_not(None)
        )
        .join(RepositoryChunk, RepositoryChunk.repository_file_id == RepositoryFile.id)
        .options(defer(RepositoryChunk.embedding))
        .filter(RepositoryFile.repository_id == repository.id)
        .order_by(RepositoryFile.path, RepositoryChunk.chunk_index)
        .all()
    )

    pending: list[tuple[RepositoryChunk, str, str]] = []
    reused = skipped = 0
    for path, chunk, has_embedding in rows:
        if not chunk.content.strip():
            skipped += 1
            if has_embedding:
                chunk.embedding = None
                chunk.embedding_content_hash = None
            continue
        text = build_embedding_text(path, chunk.content)
        digest = content_hash(provider.model, text)
        if has_embedding and chunk.embedding_content_hash == digest:
            reused += 1
        else:
            pending.append((chunk, text, digest))
    session.commit()

    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start : start + BATCH_SIZE]
        vectors = provider.embed([text for _, text, _ in batch])
        for (chunk, _, digest), vector in zip(batch, vectors, strict=True):
            chunk.embedding = vector
            chunk.embedding_content_hash = digest
        session.commit()

    return EmbeddingIndexSummary(
        chunks_total=len(rows),
        chunks_embedded=len(pending),
        chunks_reused=reused,
        chunks_skipped=skipped,
    )


@dataclass(frozen=True)
class EmbeddingOutcome:
    """Result of the embedding step that follows a scan; never raises for provider problems."""

    status: Literal["completed", "not_configured", "failed"]
    summary: EmbeddingIndexSummary | None = None
    message: str | None = None


def index_after_scan(
    session: Session, repository: Repository, provider_factory: Callable[[], EmbeddingProvider]
) -> EmbeddingOutcome:
    """Bring a freshly scanned repository's embeddings up to date.

    The scan has already been committed, so provider problems are reported in the
    outcome instead of failing the scan. Batches committed before a failure remain
    valid, and the next scan or manual run resumes with only the missing chunks.
    """

    try:
        summary = index_repository_embeddings(session, repository, provider_factory())
    except EmbeddingNotConfiguredError:
        return EmbeddingOutcome(
            status="not_configured",
            message="Embedding provider is not configured, so semantic search is unavailable.",
        )
    except EmbeddingError as error:
        session.rollback()
        logger.warning("Embedding generation failed after scan (exception type=%s)", type(error).__name__)
        return EmbeddingOutcome(
            status="failed",
            message="Embedding generation failed. Scan again or call the embeddings endpoint to retry.",
        )
    return EmbeddingOutcome(status="completed", summary=summary)
