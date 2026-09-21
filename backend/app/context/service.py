"""Build structured repository context from project analysis, exact search, and semantic search.

Nothing here is persisted: the context is computed on demand from data the scanner,
analyzer, and embedding steps have already stored, and only relevant, sanitized
snippets leave this module.
"""

import logging
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.analyzer.service import get_repository_analysis
from app.context.query import extract_terms
from app.context.redaction import is_sensitive_path, redact_secrets
from app.core.exceptions import ConflictError
from app.db.models import Repository, RepositoryFile
from app.embeddings.provider import EmbeddingError, EmbeddingNotConfiguredError, EmbeddingProvider
from app.scanner.search import CodeSearchHit, search_repository_code
from app.scanner.semantic import SemanticHit, semantic_search

logger = logging.getLogger(__name__)

MAX_QUESTION_LENGTH = 1_000
DEFAULT_MAX_CHUNKS = 8
MAX_CHUNKS_LIMIT = 20
DEFAULT_MAX_CHARS = 24_000
MIN_MAX_CHARS = 2_000
MAX_MAX_CHARS = 60_000

MAX_CHUNKS_PER_FILE = 2
MAX_SNIPPET_CHARS = 3_000
MIN_USEFUL_CHARS = 200
EXACT_PER_TERM_LIMIT = 5
SEMANTIC_LIMIT = 10
RRF_K = 60

MAX_DEPENDENCIES = 40
MAX_IMPORTANT_FILES = 20
MAX_ENTRY_POINTS = 10
MAX_LAYOUT = 30
ROOT_LAYOUT_PATH = "(root)"


@dataclass(frozen=True)
class FusedChunk:
    file_path: str
    language: str | None
    start_line: int
    end_line: int
    snippet: str
    score: float
    sources: tuple[str, ...]
    matched_terms: tuple[str, ...]


@dataclass(frozen=True)
class Fusion:
    chunks: list[FusedChunk]
    duplicates_merged: int


@dataclass(frozen=True)
class ContextChunk:
    file_path: str
    language: str | None
    start_line: int
    end_line: int
    snippet: str
    score: float
    sources: tuple[str, ...]
    matched_terms: tuple[str, ...]
    truncated: bool


@dataclass(frozen=True)
class Assembly:
    chunks: list[ContextChunk]
    chars: int
    withheld: int
    redactions: int
    omitted: int
    truncated: bool


@dataclass(frozen=True)
class ContextFile:
    file_path: str
    language: str | None
    chunk_count: int
    sources: tuple[str, ...]


@dataclass(frozen=True)
class ProjectContext:
    status: str
    project_type: str
    languages: list[str]
    frameworks: list[str]
    package_managers: list[str]
    dependencies: list[dict]
    dependencies_total: int
    important_files: list[str]
    entry_points: list[dict]
    layout: list[dict]


@dataclass(frozen=True)
class ExactRetrieval:
    status: str
    terms: list[str]
    hits: int


@dataclass(frozen=True)
class SemanticRetrieval:
    status: str
    hits: int


@dataclass(frozen=True)
class RetrievalInfo:
    analysis: str
    exact: ExactRetrieval
    semantic: SemanticRetrieval
    candidates: int
    duplicates_merged: int
    returned: int
    omitted_chunks: int
    chars: int
    max_chars: int
    truncated: bool
    withheld_chunks: int
    redactions: int


@dataclass(frozen=True)
class RepositoryContext:
    repository_id: uuid.UUID
    question: str
    project: ProjectContext | None
    relevant_chunks: list[ContextChunk]
    relevant_files: list[ContextFile]
    retrieval: RetrievalInfo


@dataclass(frozen=True)
class _Item:
    file_path: str
    language: str | None
    start_line: int
    end_line: int
    snippet: str
    source: str
    list_id: str
    term_index: int | None
    contribution: float


def fuse_hits(
    exact_results: list[tuple[str, list[CodeSearchHit]]], semantic_hits: list[SemanticHit]
) -> Fusion:
    """Combine ranked result lists with Reciprocal Rank Fusion and merge overlapping ranges.

    Every exact term and the semantic search each contribute one ranked list; a chunk
    scores the sum of 1 / (RRF_K + rank) over the lists it appears in. Overlapping line
    ranges in one file collapse into one chunk that keeps the widest range (a semantic
    chunk over the narrower exact hit inside it) and the union of sources and terms.
    """

    items: list[_Item] = []
    for term_index, (_term, hits) in enumerate(exact_results):
        for rank, hit in enumerate(hits, start=1):
            items.append(
                _Item(hit.file_path, hit.language, hit.start_line, hit.end_line, hit.snippet,
                      "exact", f"exact:{term_index}", term_index, 1 / (RRF_K + rank))
            )
    for rank, hit in enumerate(semantic_hits, start=1):
        items.append(
            _Item(hit.file_path, hit.language, hit.start_line, hit.end_line, hit.snippet,
                  "semantic", "semantic", None, 1 / (RRF_K + rank))
        )

    by_file: dict[str, list[_Item]] = defaultdict(list)
    for item in items:
        by_file[item.file_path].append(item)

    clusters: list[list[_Item]] = []
    for file_path in sorted(by_file):
        current: list[_Item] = []
        current_end = 0
        for item in sorted(by_file[file_path], key=lambda entry: (entry.start_line, entry.end_line)):
            if current and item.start_line <= current_end:
                current.append(item)
                current_end = max(current_end, item.end_line)
            else:
                if current:
                    clusters.append(current)
                current, current_end = [item], item.end_line
        if current:
            clusters.append(current)

    terms = [term for term, _hits in exact_results]
    chunks: list[FusedChunk] = []
    for cluster in clusters:
        widest = max(
            cluster,
            key=lambda entry: (entry.end_line - entry.start_line, entry.source == "semantic", -entry.start_line),
        )
        best_per_list: dict[str, float] = {}
        for entry in cluster:
            best_per_list[entry.list_id] = max(best_per_list.get(entry.list_id, 0.0), entry.contribution)
        chunks.append(
            FusedChunk(
                file_path=widest.file_path,
                language=widest.language,
                start_line=widest.start_line,
                end_line=widest.end_line,
                snippet=widest.snippet,
                score=sum(best_per_list[list_id] for list_id in sorted(best_per_list)),
                sources=tuple(sorted({entry.source for entry in cluster})),
                matched_terms=tuple(
                    terms[index] for index in sorted({e.term_index for e in cluster if e.term_index is not None})
                ),
            )
        )
    chunks.sort(key=lambda chunk: (-chunk.score, chunk.file_path, chunk.start_line))
    return Fusion(chunks=chunks, duplicates_merged=len(items) - len(clusters))


def truncate_at_line(text: str, limit: int) -> tuple[str, bool]:
    """Cut text to at most `limit` characters on a line boundary; returns (text, was_cut).

    A single line longer than the limit is hard-cut, so something is always returned.
    """

    if len(text) <= limit:
        return text, False
    cut = text[:limit]
    if text[limit] == "\n":
        return cut, True
    boundary = cut.rfind("\n")
    return (cut[:boundary] if boundary > 0 else cut), True


def assemble_chunks(fused: list[FusedChunk], *, max_chunks: int, max_chars: int) -> Assembly:
    """Apply safety rules and budgets to ranked chunks, in rank order.

    Sensitive files are withheld, secrets are redacted, each snippet is capped, at most
    MAX_CHUNKS_PER_FILE chunks come from one file, and the total stays within
    `max_chunks` and `max_chars`. Truncated snippets end on whole lines and their
    `end_line` is adjusted to match.
    """

    chunks: list[ContextChunk] = []
    per_file: Counter[str] = Counter()
    chars = redactions = omitted = withheld = 0
    truncated = False

    for chunk in fused:
        if is_sensitive_path(chunk.file_path):
            withheld += 1
            continue
        if len(chunks) >= max_chunks:
            omitted += 1
            truncated = True
            continue
        if per_file[chunk.file_path] >= MAX_CHUNKS_PER_FILE:
            omitted += 1
            continue

        text, cut = truncate_at_line(chunk.snippet.rstrip("\n"), MAX_SNIPPET_CHARS)
        text, replaced = redact_secrets(text)
        text, recut = truncate_at_line(text, MAX_SNIPPET_CHARS)
        cut = cut or recut

        remaining = max_chars - chars
        if len(text) > remaining:
            if remaining < MIN_USEFUL_CHARS:
                omitted += 1
                truncated = True
                continue
            text, _ = truncate_at_line(text, remaining)
            cut = True

        chunks.append(
            ContextChunk(
                file_path=chunk.file_path,
                language=chunk.language,
                start_line=chunk.start_line,
                end_line=chunk.start_line + text.count("\n") if cut else chunk.end_line,
                snippet=text,
                score=round(chunk.score, 6),
                sources=chunk.sources,
                matched_terms=chunk.matched_terms,
                truncated=cut,
            )
        )
        per_file[chunk.file_path] += 1
        chars += len(text)
        redactions += replaced
        truncated = truncated or cut

    return Assembly(chunks, chars, withheld, redactions, omitted, truncated)


def build_layout(paths: list[str]) -> list[dict]:
    """Top-level directory overview (file counts), skipping sensitive paths."""

    counts: Counter[str] = Counter()
    for path in paths:
        if is_sensitive_path(path):
            continue
        head, separator, _rest = path.partition("/")
        counts[f"{head}/" if separator else ROOT_LAYOUT_PATH] += 1
    ordered = sorted(counts.items(), key=lambda entry: (-entry[1], entry[0]))
    return [{"path": path, "files": files} for path, files in ordered[:MAX_LAYOUT]]


def _project_context(session: Session, repository: Repository, paths: list[str]) -> ProjectContext | None:
    analysis = get_repository_analysis(session, repository)
    if analysis is None:
        return None
    dependencies = analysis.dependencies or []
    return ProjectContext(
        status=analysis.status,
        project_type=analysis.project_type,
        languages=list(analysis.languages or []),
        frameworks=list(analysis.frameworks or []),
        package_managers=list(analysis.package_managers or []),
        dependencies=dependencies[:MAX_DEPENDENCIES],
        dependencies_total=len(dependencies),
        important_files=list((analysis.important_files or [])[:MAX_IMPORTANT_FILES]),
        entry_points=list((analysis.entry_points or [])[:MAX_ENTRY_POINTS]),
        layout=build_layout(paths),
    )


def _semantic_stage(
    session: Session,
    repository: Repository,
    provider_factory: Callable[[], EmbeddingProvider],
    question: str,
) -> tuple[str, list[SemanticHit]]:
    """Run semantic search; every failure degrades to a status instead of an error."""

    try:
        return "used", semantic_search(session, repository, provider_factory(), question, SEMANTIC_LIMIT)
    except ConflictError:
        return "no_embeddings", []
    except EmbeddingNotConfiguredError:
        return "not_configured", []
    except EmbeddingError as error:
        logger.warning("Semantic retrieval failed for context (exception type=%s)", type(error).__name__)
    except Exception as error:
        session.rollback()
        logger.warning("Semantic retrieval failed for context (exception type=%s)", type(error).__name__)
    return "failed", []


def build_repository_context(
    session: Session,
    repository: Repository,
    provider_factory: Callable[[], EmbeddingProvider],
    *,
    question: str,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    max_chars: int = DEFAULT_MAX_CHARS,
    include_analysis: bool = True,
    include_exact: bool = True,
    include_semantic: bool = True,
) -> RepositoryContext:
    """Assemble context for one question about a repository the caller already owns."""

    paths = [path for (path,) in session.query(RepositoryFile.path).filter(RepositoryFile.repository_id == repository.id)]
    if not paths:
        raise ConflictError("Repository has not been scanned yet. Scan the repository first.")
    question = question.strip()

    project = _project_context(session, repository, paths) if include_analysis else None
    analysis_status = "skipped" if not include_analysis else "included" if project else "not_available"

    exact_results: list[tuple[str, list[CodeSearchHit]]] = []
    terms: list[str] = []
    if not include_exact:
        exact_status = "skipped"
    else:
        terms = extract_terms(question)
        exact_status = "used" if terms else "no_terms"
        for term in terms:
            exact_results.append((term, search_repository_code(session, repository, term, EXACT_PER_TERM_LIMIT)))

    semantic_status, semantic_hits = (
        _semantic_stage(session, repository, provider_factory, question) if include_semantic else ("skipped", [])
    )

    fusion = fuse_hits(exact_results, semantic_hits)
    assembly = assemble_chunks(fusion.chunks, max_chunks=max_chunks, max_chars=max_chars)

    files: dict[str, list[ContextChunk]] = {}
    for chunk in assembly.chunks:
        files.setdefault(chunk.file_path, []).append(chunk)
    relevant_files = [
        ContextFile(
            file_path=path,
            language=chunks[0].language,
            chunk_count=len(chunks),
            sources=tuple(sorted({source for chunk in chunks for source in chunk.sources})),
        )
        for path, chunks in files.items()
    ]

    return RepositoryContext(
        repository_id=repository.id,
        question=question,
        project=project,
        relevant_chunks=assembly.chunks,
        relevant_files=relevant_files,
        retrieval=RetrievalInfo(
            analysis=analysis_status,
            exact=ExactRetrieval(exact_status, terms, sum(len(hits) for _term, hits in exact_results)),
            semantic=SemanticRetrieval(semantic_status, len(semantic_hits)),
            candidates=len(fusion.chunks),
            duplicates_merged=fusion.duplicates_merged,
            returned=len(assembly.chunks),
            omitted_chunks=assembly.omitted,
            chars=assembly.chars,
            max_chars=max_chars,
            truncated=assembly.truncated,
            withheld_chunks=assembly.withheld,
            redactions=assembly.redactions,
        ),
    )
