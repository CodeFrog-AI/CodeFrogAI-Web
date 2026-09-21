"""Request and response schemas for the repository agent endpoint."""

import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_MESSAGE_CHARS = 4_000
MAX_HISTORY_MESSAGES = 20


def _reject_nul(value: str) -> str:
    if chr(0) in value:
        raise ValueError("text must not contain NUL characters")
    return value


class AgentHistoryMessage(BaseModel):
    """An earlier turn. Only user and assistant turns are accepted, never system or tool ones."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS, pattern=r"\S")

    _no_nul = field_validator("content")(_reject_nul)


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS, pattern=r"\S")
    history: list[AgentHistoryMessage] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)

    _no_nul = field_validator("message")(_reject_nul)


class AgentToolCall(BaseModel):
    """A tool the model called. The repository is always the selected one and is not listed."""

    iteration: int
    name: str
    arguments: dict[str, Any]
    ok: bool
    error_code: str | None
    duration_ms: int


class AgentMetadata(BaseModel):
    iterations: int
    tool_calls: int
    stop_reason: Literal["final_answer", "max_iterations", "empty_response"]
    model: str
    duration_ms: int


class AgentResponse(BaseModel):
    repository_id: uuid.UUID
    answer: str
    tool_calls: list[AgentToolCall]
    metadata: AgentMetadata
