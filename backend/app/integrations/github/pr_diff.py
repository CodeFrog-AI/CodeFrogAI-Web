"""Turn a pull request's raw file list into a bounded, redacted, safe-to-share diff.

Protected files (env files, keys, credentials) and unsafe paths are counted but never named
or shown; secrets in patches are redacted; sizes are capped per file and in total.
"""

import re
from dataclasses import dataclass

from app.context.redaction import is_sensitive_path, redact_secrets
from app.context.service import truncate_at_line
from app.integrations.github.pull_requests import PullRequestDiff
from app.workspace.workspace import WorkspaceError, validate_path

MAX_FILES = 100
MAX_PATCH_CHARS_PER_FILE = 20_000
MAX_TOTAL_PATCH_CHARS = 100_000

_STATUS = {"added": "added", "removed": "deleted", "modified": "modified", "changed": "modified", "renamed": "renamed", "copied": "added"}
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
NEWLINE = chr(10)


@dataclass(frozen=True)
class DiffFileView:
    path: str
    status: str  # added | modified | deleted | renamed
    additions: int
    deletions: int
    patch: str
    patch_truncated: bool
    patch_available: bool
    previous_path: str | None = None


@dataclass(frozen=True)
class DiffView:
    files: list[DiffFileView]
    total_files: int
    withheld: int
    truncated: bool  # files or patch text were left out because of the limits


def sanitize_diff(diff: PullRequestDiff) -> DiffView:
    files: list[DiffFileView] = []
    withheld = 0
    truncated = diff.truncated
    budget = MAX_TOTAL_PATCH_CHARS
    for raw in diff.files:
        path = _safe_path(raw.path)
        previous = _safe_path(raw.previous_path) if raw.previous_path else None
        if path is None or (raw.previous_path and previous is None):
            withheld += 1
            continue
        if len(files) >= MAX_FILES:
            truncated = True
            continue
        text, cut = truncate_at_line(redact_secrets(raw.patch or "")[0], max(min(MAX_PATCH_CHARS_PER_FILE, budget), 0))
        budget -= len(text)
        truncated = truncated or cut
        files.append(
            DiffFileView(
                path=path,
                status=_STATUS.get(raw.status, "modified"),
                additions=max(raw.additions, 0),
                deletions=max(raw.deletions, 0),
                patch=text,
                patch_truncated=cut,
                patch_available=raw.patch is not None,
                previous_path=previous,
            )
        )
    return DiffView(files=files, total_files=len(diff.files), withheld=withheld, truncated=truncated)


def _safe_path(path: str) -> str | None:
    try:
        relative = validate_path(path)
    except WorkspaceError:
        return None
    return None if is_sensitive_path(relative) else relative


def new_side_lines(patch: str) -> set[int]:
    """Line numbers (in the new version of the file) that a unified patch shows."""

    lines: set[int] = set()
    current = 0
    in_hunk = False
    for line in patch.split(NEWLINE):
        header = _HUNK.match(line)
        if header:
            current, in_hunk = int(header.group(1)), True
        elif in_hunk and line[:1] in ("+", " "):
            lines.add(current)
            current += 1
    return lines


def annotate_patch(patch: str) -> str:
    """The patch with the new-file line number in front of each shown line, so a reviewer can cite lines."""

    out: list[str] = []
    current = 0
    for line in patch.split(NEWLINE):
        header = _HUNK.match(line)
        if header:
            current = int(header.group(1))
            out.append(line)
        elif line[:1] in ("+", " "):
            out.append(f"{current:>6}| {line}")
            current += 1
        else:
            out.append(f"      | {line}")
    return NEWLINE.join(out)
