"""A small, typed abstraction for read-only agent tools.

A tool has a name, a description, a Pydantic input model (which doubles as its JSON
schema), and a handler. `Tool.execute` validates the arguments and always returns a
structured `ToolResult` for expected failures, so an agent loop can feed errors back to
the model instead of handling exceptions.
"""

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy.orm import Session

from app.core.exceptions import ApplicationError, UnauthorizedError
from app.db.models import Repository, User
from app.embeddings.provider import EmbeddingProvider, get_embedding_provider
from app.scanner.service import get_owned_repository

if TYPE_CHECKING:
    from app.workspace import Workspace

logger = logging.getLogger(__name__)

InputT = TypeVar("InputT", bound=BaseModel)


class ToolInput(BaseModel):
    """Base class for tool inputs. Unknown fields are rejected so mistakes are reported."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


@dataclass(frozen=True)
class ToolContext:
    """Who is calling and with what resources. `user` is None for an unauthenticated caller.

    `workspace` is set only by the server, after the user has approved a plan. Without it,
    write tools refuse to run: approval is a property of the context, never of model input.
    `checkout` is a read-only view of the repository's local checkout (if one exists): it lets
    read_file show current local content but never enables writes.
    """

    session: Session
    user: User | None
    embedding_provider_factory: Callable[[], EmbeddingProvider] = get_embedding_provider
    workspace: "Workspace | None" = None
    checkout: "Workspace | None" = None


@dataclass(frozen=True)
class ToolError:
    """A client-safe failure. Codes match the API's error codes (e.g. RESOURCE_NOT_FOUND)."""

    code: str
    message: str
    details: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: dict[str, Any] | None = None
    error: ToolError | None = None

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable form suitable for returning to a model."""

        if self.ok:
            return {"ok": True, "output": self.output}
        assert self.error is not None
        error: dict[str, Any] = {"code": self.error.code, "message": self.error.message}
        if self.error.details:
            error["details"] = self.error.details
        return {"ok": False, "error": error}


def failure(code: str, message: str, details: list[dict[str, Any]] | None = None) -> ToolResult:
    return ToolResult(ok=False, error=ToolError(code, message, details))


@dataclass(frozen=True)
class Tool(Generic[InputT]):
    name: str
    description: str
    input_model: type[InputT]
    handler: Callable[[ToolContext, InputT], BaseModel]
    writes: bool = False

    @property
    def input_schema(self) -> dict[str, Any]:
        return self.input_model.model_json_schema()

    def definition(self) -> dict[str, Any]:
        """The discoverable description of this tool: name, description, and input schema."""

        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def execute(self, context: ToolContext, arguments: Any) -> ToolResult:
        """Validate `arguments`, run the handler, and return a structured result.

        Expected failures (bad input, missing auth, not found, conflicts) become error
        results. Unexpected exceptions are bugs and propagate.
        """

        try:
            if context.user is None or context.user.status != "active":
                raise UnauthorizedError()
            if self.writes and context.workspace is None:
                return failure("APPROVAL_REQUIRED", "Changes can only be made after the user approves a plan.")
            try:
                parsed = self.input_model.model_validate(arguments)
            except ValidationError as error:
                return failure("INVALID_INPUT", "Invalid tool input", _safe_details(error))
            output = self.handler(context, parsed)
        except ApplicationError as error:
            logger.info("Tool failed name=%s error_code=%s", self.name, error.code)
            return failure(error.code, error.message)
        logger.info("Tool succeeded name=%s", self.name)
        return ToolResult(ok=True, output=output.model_dump(mode="json"))


def _safe_details(error: ValidationError) -> list[dict[str, Any]]:
    """Field locations and messages only, never the submitted values."""

    return [
        {"location": [str(part) for part in item["loc"]], "message": item["msg"], "type": item["type"]}
        for item in error.errors()
    ]


def owned_repository(context: ToolContext, repository_id: uuid.UUID) -> Repository:
    """Load a repository only if the calling user owns it (otherwise a not-found error)."""

    if context.user is None:
        raise UnauthorizedError()
    return get_owned_repository(context.session, repository_id, context.user)
