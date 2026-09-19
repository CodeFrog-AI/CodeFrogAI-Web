"""Response schemas for repository operations."""

import uuid
from typing import Literal

from pydantic import BaseModel


class RepositoryScanResponse(BaseModel):
    """Outcome of a repository scan. Contains counts only, never GitHub credentials."""

    repository_id: uuid.UUID
    status: Literal["completed"]
    files_discovered: int
    files_indexed: int
    files_skipped: int
    files_removed: int
    chunks_created: int
