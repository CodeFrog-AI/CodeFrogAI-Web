"""analyze_project: return the stored static project analysis of a repository."""

import uuid

from pydantic import Field

from app.analyzer.service import get_repository_analysis
from app.core.exceptions import ConflictError
from app.schemas.repositories import ProjectAnalysisResponse
from app.tools.base import Tool, ToolContext, ToolInput, owned_repository


class AnalyzeProjectInput(ToolInput):
    repository_id: uuid.UUID = Field(description="ID of the repository to describe.")


def _analyze_project(context: ToolContext, arguments: AnalyzeProjectInput) -> ProjectAnalysisResponse:
    repository = owned_repository(context, arguments.repository_id)
    analysis = get_repository_analysis(context.session, repository)
    if analysis is None:
        raise ConflictError("Repository has not been analyzed yet. Scan the repository first.")
    return ProjectAnalysisResponse.from_analysis(repository.id, analysis)


ANALYZE_PROJECT = Tool(
    name="analyze_project",
    description=(
        "Describe the repository's project structure from its stored analysis: project type, "
        "languages, frameworks, package managers, dependencies, important files, and entry points. "
        "Names and paths only; no file contents."
    ),
    input_model=AnalyzeProjectInput,
    handler=_analyze_project,
)
