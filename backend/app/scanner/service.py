"""Scan a connected repository and persist its files and code chunks."""

import uuid
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.core.exceptions import BadRequestError, NotFoundError
from app.db.models import GitHubAccount, Repository, RepositoryChunk, RepositoryFile, User
from app.integrations.github.contents import GitHubNotFoundError, GitHubTree
from app.scanner.chunking import chunk_source
from app.scanner.filters import MAX_FILE_BYTES, detect_language

MAX_INDEXED_FILES = 1000


class ContentClient(Protocol):
    def get_tree(self, owner: str, name: str, ref: str) -> GitHubTree: ...

    def get_file_content(self, owner: str, name: str, sha: str) -> bytes: ...


@dataclass(frozen=True)
class ScanSummary:
    repository_id: uuid.UUID
    status: str
    files_discovered: int
    files_indexed: int
    files_skipped: int
    files_removed: int
    chunks_created: int


def get_owned_repository(session: Session, repository_id: uuid.UUID, user: User) -> Repository:
    """Load a repository only if it belongs to the user; otherwise behave as not found."""

    repository = (
        session.query(Repository)
        .join(GitHubAccount, Repository.github_account_id == GitHubAccount.id)
        .filter(Repository.id == repository_id, GitHubAccount.user_id == user.id)
        .first()
    )
    if repository is None:
        raise NotFoundError("Repository not found")
    return repository


def _decode_text(content: bytes) -> str | None:
    """Return UTF-8 text, or None for binary/undecodable content."""

    if b"\x00" in content:
        return None
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return None


def scan_repository(session: Session, repository: Repository, client: ContentClient) -> ScanSummary:
    """Synchronize the stored index for a repository with its default branch.

    Files are matched by path. Unchanged files (same blob SHA) are left alone,
    changed files get fresh chunks, and files that disappeared are removed, so
    repeated scans never accumulate duplicates. Everything commits atomically.
    """

    tree = client.get_tree(repository.owner, repository.name, repository.default_branch)
    if tree.truncated:
        raise BadRequestError("Repository is too large to scan")

    candidates = sorted(
        (
            (entry, language)
            for entry in tree.entries
            if (language := detect_language(entry.path)) is not None
            and (entry.size is None or entry.size <= MAX_FILE_BYTES)
        ),
        key=lambda item: item[0].path,
    )[:MAX_INDEXED_FILES]

    existing = {
        file.path: file
        for file in session.query(RepositoryFile).filter(RepositoryFile.repository_id == repository.id)
    }
    kept_paths: set[str] = set()
    chunks_created = 0

    try:
        for entry, language in candidates:
            stored = existing.get(entry.path)
            if stored is not None and stored.sha == entry.sha:
                kept_paths.add(entry.path)
                continue

            try:
                raw_content = client.get_file_content(repository.owner, repository.name, entry.sha)
            except GitHubNotFoundError:
                continue
            text = _decode_text(raw_content) if len(raw_content) <= MAX_FILE_BYTES else None
            if text is None:
                continue

            if stored is None:
                stored = RepositoryFile(id=uuid.uuid4(), repository_id=repository.id, path=entry.path)
                session.add(stored)
            else:
                session.execute(
                    delete(RepositoryChunk).where(RepositoryChunk.repository_file_id == stored.id)
                )
            stored.language = language
            stored.sha = entry.sha
            stored.size_bytes = len(raw_content)
            session.flush()

            chunks = chunk_source(text)
            session.add_all(
                RepositoryChunk(
                    repository_file_id=stored.id,
                    chunk_index=chunk.chunk_index,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    content=chunk.content,
                )
                for chunk in chunks
            )
            chunks_created += len(chunks)
            kept_paths.add(entry.path)

        removed = [file for path, file in existing.items() if path not in kept_paths]
        for file in removed:
            session.delete(file)
        session.commit()
    except Exception:
        session.rollback()
        raise

    return ScanSummary(
        repository_id=repository.id,
        status="completed",
        files_discovered=len(tree.entries),
        files_indexed=len(kept_paths),
        files_skipped=len(tree.entries) - len(kept_paths),
        files_removed=len(removed),
        chunks_created=chunks_created,
    )
