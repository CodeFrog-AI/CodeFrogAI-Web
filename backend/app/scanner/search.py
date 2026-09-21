"""Case-insensitive text search over a repository's indexed code chunks."""

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.db.models import Repository, RepositoryChunk, RepositoryFile

DEFAULT_RESULT_LIMIT = 20
MAX_RESULT_LIMIT = 50
MAX_QUERY_LENGTH = 200
CONTEXT_LINES = 2


@dataclass(frozen=True)
class CodeSearchHit:
    file_path: str
    language: str | None
    start_line: int
    end_line: int
    snippet: str


def _escape_like(value: str) -> str:
    """Escape LIKE wildcards so the query is matched literally."""

    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _build_hit(file: RepositoryFile, chunk: RepositoryChunk, query: str) -> CodeSearchHit:
    """Narrow a matching chunk to the first match plus a few lines of context."""

    lines = chunk.content.split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    position = chunk.content.lower().find(query.lower())
    if position < 0:
        first, last = 0, len(lines) - 1
    else:
        first = chunk.content.count("\n", 0, position)
        last = first + query.count("\n")
    first = max(first - CONTEXT_LINES, 0)
    last = min(last + CONTEXT_LINES, len(lines) - 1)

    return CodeSearchHit(
        file_path=file.path,
        language=file.language,
        start_line=chunk.start_line + first,
        end_line=chunk.start_line + last,
        snippet="\n".join(lines[first : last + 1]),
    )


def search_repository_code(
    session: Session, repository: Repository, query: str, limit: int = DEFAULT_RESULT_LIMIT
) -> list[CodeSearchHit]:
    """Return up to `limit` matching chunks of one repository, ordered by file and position."""

    rows = (
        session.query(RepositoryFile, RepositoryChunk)
        .join(RepositoryChunk, RepositoryChunk.repository_file_id == RepositoryFile.id)
        .filter(
            RepositoryFile.repository_id == repository.id,
            RepositoryChunk.content.ilike(f"%{_escape_like(query)}%", escape="\\"),
        )
        .order_by(RepositoryFile.path, RepositoryChunk.chunk_index)
        .limit(limit)
    )
    return [_build_hit(file, chunk, query) for file, chunk in rows]
