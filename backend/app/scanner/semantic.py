"""Semantic (vector similarity) search over a repository's embedded code chunks."""

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.exceptions import ConflictError
from app.db.models import Repository, RepositoryChunk, RepositoryFile
from app.embeddings.provider import EmbeddingProvider

DEFAULT_SEMANTIC_LIMIT = 10
MAX_SEMANTIC_LIMIT = 25
MAX_SEMANTIC_QUERY_LENGTH = 500


@dataclass(frozen=True)
class SemanticHit:
    file_path: str
    language: str | None
    start_line: int
    end_line: int
    snippet: str
    # Cosine similarity: 1.0 is identical direction, 0.0 unrelated, -1.0 opposite.
    score: float


def count_embedded_chunks(session: Session, repository: Repository) -> int:
    return session.scalar(
        select(func.count())
        .select_from(RepositoryChunk)
        .join(RepositoryFile, RepositoryChunk.repository_file_id == RepositoryFile.id)
        .where(RepositoryFile.repository_id == repository.id, RepositoryChunk.embedding.is_not(None))
    )


def find_similar_chunks(
    session: Session,
    repository: Repository,
    query_embedding: list[float],
    limit: int,
    min_score: float | None = None,
) -> list[SemanticHit]:
    """Rank one repository's embedded chunks by cosine distance (pgvector `<=>`)."""

    distance = RepositoryChunk.embedding.cosine_distance(query_embedding)
    query = (
        session.query(RepositoryFile, RepositoryChunk, distance.label("distance"))
        .join(RepositoryChunk, RepositoryChunk.repository_file_id == RepositoryFile.id)
        .filter(RepositoryFile.repository_id == repository.id, RepositoryChunk.embedding.is_not(None))
    )
    if min_score is not None:
        query = query.filter(distance <= 1 - min_score)
    rows = query.order_by(distance, RepositoryFile.path, RepositoryChunk.chunk_index).limit(limit)
    return [
        SemanticHit(
            file_path=file.path,
            language=file.language,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            snippet=chunk.content,
            score=round(1 - float(chunk_distance), 4),
        )
        for file, chunk, chunk_distance in rows
    ]


def semantic_search(
    session: Session,
    repository: Repository,
    provider: EmbeddingProvider,
    query: str,
    limit: int = DEFAULT_SEMANTIC_LIMIT,
    min_score: float | None = None,
) -> list[SemanticHit]:
    """Embed the query and return the most similar chunks of the repository."""

    if count_embedded_chunks(session, repository) == 0:
        raise ConflictError("Repository has no embeddings yet. Generate embeddings first.")
    (query_embedding,) = provider.embed([query])
    return find_similar_chunks(session, repository, query_embedding, limit, min_score)
