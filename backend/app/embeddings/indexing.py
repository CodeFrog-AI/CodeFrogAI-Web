"""Generate and store embeddings for a repository's code chunks."""

import hashlib
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.db.models import Repository, RepositoryChunk, RepositoryFile
from app.embeddings.provider import EmbeddingProvider

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

    rows = (
        session.query(RepositoryFile.path, RepositoryChunk)
        .join(RepositoryChunk, RepositoryChunk.repository_file_id == RepositoryFile.id)
        .filter(RepositoryFile.repository_id == repository.id)
        .order_by(RepositoryFile.path, RepositoryChunk.chunk_index)
        .all()
    )

    pending: list[tuple[RepositoryChunk, str, str]] = []
    reused = skipped = 0
    for path, chunk in rows:
        if not chunk.content.strip():
            skipped += 1
            if chunk.embedding is not None:
                chunk.embedding = None
                chunk.embedding_content_hash = None
            continue
        text = build_embedding_text(path, chunk.content)
        digest = content_hash(provider.model, text)
        if chunk.embedding is not None and chunk.embedding_content_hash == digest:
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
