"""Analyze a scanned repository and persist the result (one row per repository)."""

import logging
from collections import defaultdict
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.analyzer.manifests import ManifestInfo, ManifestParseError, is_manifest, parse_manifest
from app.analyzer.project import analyze_project, is_ignored_path
from app.db.models import Repository, RepositoryAnalysis, RepositoryChunk, RepositoryFile
from app.integrations.github.contents import GitHubNotFoundError
from app.scanner.filters import MAX_FILE_BYTES
from app.scanner.service import ContentClient

logger = logging.getLogger(__name__)

ANALYZER_VERSION = 1
MAX_MANIFESTS = 25
FAILURE_MESSAGE = "Project analysis failed. The next scan will retry."


def _indexed_texts(session: Session, repository: Repository, paths: list[str]) -> dict[str, str]:
    """Rebuild file text from chunks the scanner already stored, avoiding a re-download."""

    if not paths:
        return {}
    rows = (
        session.query(RepositoryFile.path, RepositoryChunk.content)
        .join(RepositoryChunk, RepositoryChunk.repository_file_id == RepositoryFile.id)
        .filter(RepositoryFile.repository_id == repository.id, RepositoryFile.path.in_(paths))
        .order_by(RepositoryFile.path, RepositoryChunk.chunk_index)
    )
    parts: dict[str, list[str]] = defaultdict(list)
    for path, content in rows:
        parts[path].append(content)
    return {path: "".join(chunks) for path, chunks in parts.items()}


def _language_counts(session: Session, repository: Repository) -> dict[str, int]:
    rows = (
        session.query(RepositoryFile.language, func.count())
        .filter(RepositoryFile.repository_id == repository.id, RepositoryFile.language.is_not(None))
        .group_by(RepositoryFile.language)
    )
    return {language: count for language, count in rows}


def _load_manifests(
    session: Session, repository: Repository, client: ContentClient, entries: list
) -> tuple[list[ManifestInfo], list[dict[str, str]]]:
    """Parse the repository's manifests; unreadable ones are skipped with a safe reason."""

    selected = sorted(
        (entry for entry in entries if is_manifest(entry.path) and not is_ignored_path(entry.path)),
        key=lambda entry: (entry.path.count("/"), entry.path),
    )[:MAX_MANIFESTS]
    indexed = _indexed_texts(session, repository, [entry.path for entry in selected])

    manifests: list[ManifestInfo] = []
    skipped: list[dict[str, str]] = []
    for entry in selected:
        text = indexed.get(entry.path)
        if text is None:
            if entry.size is not None and entry.size > MAX_FILE_BYTES:
                skipped.append({"path": entry.path, "reason": "too_large"})
                continue
            try:
                raw = client.get_file_content(repository.owner, repository.name, entry.sha)
            except GitHubNotFoundError:
                skipped.append({"path": entry.path, "reason": "unavailable"})
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                skipped.append({"path": entry.path, "reason": "unreadable"})
                continue
        try:
            manifests.append(parse_manifest(entry.path, text))
        except ManifestParseError:
            skipped.append({"path": entry.path, "reason": "malformed"})
    return manifests, skipped


def _get_or_create(session: Session, repository: Repository) -> RepositoryAnalysis:
    analysis = (
        session.query(RepositoryAnalysis)
        .filter(RepositoryAnalysis.repository_id == repository.id)
        .first()
    )
    if analysis is None:
        analysis = RepositoryAnalysis(repository_id=repository.id)
        session.add(analysis)
    return analysis


def analyze_repository(
    session: Session, repository: Repository, client: ContentClient
) -> RepositoryAnalysis:
    """Compute and store the analysis from the stored index plus a few manifest files."""

    tree = client.get_tree(repository.owner, repository.name, repository.default_branch)
    manifests, skipped = _load_manifests(session, repository, client, tree.entries)
    result = analyze_project(
        [entry.path for entry in tree.entries], manifests, _language_counts(session, repository)
    )

    analysis = _get_or_create(session, repository)
    analysis.status = "partial" if skipped else "completed"
    analysis.project_type = result.project_type
    analysis.languages = result.languages
    analysis.frameworks = result.frameworks
    analysis.package_managers = result.package_managers
    analysis.dependencies = result.dependencies
    analysis.important_files = result.important_files
    analysis.entry_points = result.entry_points
    analysis.analysis_metadata = {
        "analyzer_version": ANALYZER_VERSION,
        "manifests_analyzed": sorted(manifest.path for manifest in manifests),
        "manifests_skipped": skipped,
        "dependencies_truncated": result.dependencies_truncated,
    }
    session.commit()
    return analysis


def _record_failure(session: Session, repository: Repository) -> None:
    analysis = _get_or_create(session, repository)
    analysis.status = "failed"
    analysis.analysis_metadata = {**(analysis.analysis_metadata or {}), "analyzer_version": ANALYZER_VERSION, "error": FAILURE_MESSAGE}
    session.commit()


def analyze_after_scan(session: Session, repository: Repository, client: ContentClient) -> str:
    """Run the analysis without ever failing the scan that triggered it; returns its status.

    Analysis is optional enrichment: any problem is logged (exception type only, never
    file content) and recorded as a `failed` status, and a later scan retries it.
    """

    try:
        return analyze_repository(session, repository, client).status
    except Exception as error:
        session.rollback()
        logger.warning("Project analysis failed after scan (exception type=%s)", type(error).__name__)
    try:
        _record_failure(session, repository)
    except Exception:
        session.rollback()
    return "failed"


def get_repository_analysis(session: Session, repository: Repository) -> RepositoryAnalysis | None:
    return (
        session.query(RepositoryAnalysis)
        .filter(RepositoryAnalysis.repository_id == repository.id)
        .first()
    )
