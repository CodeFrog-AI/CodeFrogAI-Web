"""search_code: find relevant code in an indexed repository (exact + semantic, sanitized)."""

import uuid
from dataclasses import asdict
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.context.service import (
    DEFAULT_MAX_CHUNKS,
    MAX_CHUNKS_LIMIT,
    MAX_QUESTION_LENGTH,
    build_repository_context,
)
from app.schemas.context import ContextChunk, ContextRetrieval, RepositoryContextResponse
from app.tools.base import Tool, ToolContext, ToolInput, owned_repository

# Total snippet characters returned by one call; a little under the context endpoint's default.
SEARCH_MAX_CHARS = 20_000


class SearchCodeInput(ToolInput):
    repository_id: uuid.UUID = Field(description="ID of the repository to search.")
    query: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
        pattern=r"\S",
        description="A question, symbol, or phrase to look for, e.g. 'where is GitHub OAuth handled?'.",
    )
    limit: int = Field(
        default=DEFAULT_MAX_CHUNKS,
        ge=1,
        le=MAX_CHUNKS_LIMIT,
        description="Maximum number of code snippets to return (fewer may be returned).",
    )

    @field_validator("query")
    @classmethod
    def reject_nul_characters(cls, value: str) -> str:
        if chr(0) in value:
            raise ValueError("query must not contain NUL characters")
        return value


class SearchCodeOutput(BaseModel):
    """Sanitized snippets (secrets redacted, sensitive files withheld) and how they were found."""

    repository_id: uuid.UUID
    query: str
    results: list[ContextChunk]
    retrieval: ContextRetrieval
    source: Literal["last_scan"] = Field(
        default="last_scan",
        description="Snippets come from the last repository scan and may not include local edits; use read_file for current content.",
    )


def _search_code(context: ToolContext, arguments: SearchCodeInput) -> SearchCodeOutput:
    repository = owned_repository(context, arguments.repository_id)
    found = build_repository_context(
        context.session,
        repository,
        context.embedding_provider_factory,
        question=arguments.query,
        max_chunks=arguments.limit,
        max_chars=SEARCH_MAX_CHARS,
        include_analysis=False,
    )
    response = RepositoryContextResponse.model_validate(asdict(found))
    return SearchCodeOutput(
        repository_id=response.repository_id,
        query=response.question,
        results=response.relevant_chunks,
        retrieval=response.retrieval,
    )


SEARCH_CODE = Tool(
    name="search_code",
    description=(
        "Search the repository's indexed code. Combines exact text matching (terms extracted from "
        "the query) with semantic search when embeddings are available. Returns snippets with file "
        "path, language, line range, relevance score, and which sources matched. Secrets are "
        "redacted and sensitive files are never returned. Results reflect the last repository scan and may "
        "not include local edits made since; use read_file to see a file's current content."
    ),
    input_model=SearchCodeInput,
    handler=_search_code,
)
