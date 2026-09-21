"""Request and response schemas for AI repository context."""

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.context.service import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_CHUNKS,
    MAX_CHUNKS_LIMIT,
    MAX_MAX_CHARS,
    MAX_QUESTION_LENGTH,
    MIN_MAX_CHARS,
)
from app.schemas.repositories import DependencyResponse, EntryPointResponse

Source = Literal["exact", "semantic"]


class ContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH, pattern=r"\S")
    max_chunks: int = Field(default=DEFAULT_MAX_CHUNKS, ge=1, le=MAX_CHUNKS_LIMIT)
    max_chars: int = Field(default=DEFAULT_MAX_CHARS, ge=MIN_MAX_CHARS, le=MAX_MAX_CHARS)
    include_analysis: bool = True
    include_exact: bool = True
    include_semantic: bool = True

    @field_validator("question")
    @classmethod
    def reject_nul_characters(cls, value: str) -> str:
        """PostgreSQL text parameters cannot contain NUL, so reject it before any query runs."""

        if "\x00" in value:
            raise ValueError("question must not contain NUL characters")
        return value


class ContextLayoutEntry(BaseModel):
    path: str
    files: int


class ContextProject(BaseModel):
    """Compact project analysis. Names and paths only: no versions, contents, or secrets."""

    status: Literal["completed", "partial", "failed"]
    project_type: str
    languages: list[str]
    frameworks: list[str]
    package_managers: list[str]
    dependencies: list[DependencyResponse]
    dependencies_total: int
    important_files: list[str]
    entry_points: list[EntryPointResponse]
    layout: list[ContextLayoutEntry]


class ContextChunk(BaseModel):
    """A sanitized snippet. `score` is a Reciprocal Rank Fusion score (higher is better)."""

    file_path: str
    language: str | None
    start_line: int
    end_line: int
    snippet: str
    score: float
    sources: list[Source]
    matched_terms: list[str]
    truncated: bool


class ContextFile(BaseModel):
    file_path: str
    language: str | None
    chunk_count: int
    sources: list[Source]


class ExactRetrieval(BaseModel):
    status: Literal["used", "skipped", "no_terms"]
    terms: list[str]
    hits: int


class SemanticRetrieval(BaseModel):
    status: Literal["used", "skipped", "no_embeddings", "not_configured", "failed"]
    hits: int


class ContextRetrieval(BaseModel):
    analysis: Literal["included", "not_available", "skipped"]
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


class RepositoryContextResponse(BaseModel):
    repository_id: uuid.UUID
    question: str
    project: ContextProject | None
    relevant_chunks: list[ContextChunk]
    relevant_files: list[ContextFile]
    retrieval: ContextRetrieval
