"""Fix state kept on disk beside the workspace (no database).

One small JSON file per repository records where a fix stands: which pull request and
branch it is for, the head commit it started from, the test result, and a fingerprint of the
exact uncommitted changes the tests ran against. Committing and pushing check it, so only
tested, unchanged, expected changes can be committed and only the fix commit can be pushed.
The file lives outside the checkout, so it never shows up in Git status.
"""

import json
import logging
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

FIXING = "fixing"
CHANGES_READY = "changes_ready"
TEST_FAILED = "test_failed"
READY_TO_COMMIT = "ready_to_commit"
COMMITTED = "committed"
PUSHED = "pushed"


@dataclass
class FixState:
    pr_number: int
    branch: str
    head_sha: str  # the pull request's head commit when the fix started
    status: str
    allowed_paths: list[str] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    fingerprint: str | None = None
    tests: dict | None = None
    commit: str | None = None


def _path(base_directory: Path, repository_id: uuid.UUID) -> Path:
    return Path(base_directory) / "state" / f"{repository_id}.json"


def read_state(base_directory: Path, repository_id: uuid.UUID) -> FixState | None:
    try:
        return FixState(**json.loads(_path(base_directory, repository_id).read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


def write_state(base_directory: Path, repository_id: uuid.UUID, state: FixState) -> None:
    target = _path(base_directory, repository_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(asdict(state), handle)
        os.replace(temporary, target)
    except OSError:
        Path(temporary).unlink(missing_ok=True)
        raise


def clear_state(base_directory: Path, repository_id: uuid.UUID) -> None:
    _path(base_directory, repository_id).unlink(missing_ok=True)
