"""Creating and inspecting persistent workspaces: real Git checkouts of connected repositories.

A workspace is cloned once, only when it is first needed, using the repository owner's stored
GitHub token (passed to Git through the environment; it is never written to the checkout's
config or remote URL). After that it is the local working tree: this module never resets,
cleans, or re-clones an existing checkout, so uncommitted work is never destroyed.

Callers hold the repository's `exclusive_workspace` lock around anything that changes the
checkout (initialize, sync, edit, branch, commit, push).
"""

import logging
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.db.models import Repository, User
from app.git import GitError, GitRepository, validate_base_branch
from app.git.runner import run_git
from app.integrations.github.contents import GitHubAuthError
from app.integrations.github.tokens import decrypt_access_token
from app.workspace.layout import checkout_path
from app.workspace.workspace import Workspace

logger = logging.getLogger(__name__)

CLONE_DEPTH = 50
_GITHUB_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class GitRemote:
    """Where to fetch from and push to. The token is used through the environment only."""

    url: str
    token: str | None


@dataclass(frozen=True)
class WorkspaceState:
    exists: bool
    branch: str | None = None
    default_branch: str | None = None
    uncommitted_changes: bool = False
    commit: str | None = None


@dataclass(frozen=True)
class SyncOutcome:
    action: str  # up_to_date | fast_forwarded | skipped
    reason: str | None = None


def remote_url(repository: Repository) -> str:
    """The clean, credential-free GitHub URL of the repository."""

    if not _GITHUB_NAME.match(repository.owner) or not _GITHUB_NAME.match(repository.name):
        raise GitError("INVALID_REPOSITORY", "The repository name is not supported.", 400)
    return f"https://github.com/{repository.owner}/{repository.name}.git"


def resolve_remote(repository: Repository) -> GitRemote:
    """The remote and the owner's stored GitHub token (the existing OAuth connection; nothing new)."""

    url = remote_url(repository)
    try:
        token = decrypt_access_token(repository.github_account.access_token_encrypted)
    except GitHubAuthError:
        token = None
    if token is None:
        raise GitError("GITHUB_AUTH_REQUIRED", "GitHub authorization is required. Reconnect your GitHub account.", 403)
    return GitRemote(url, token)


def _git(root: Path, repository: Repository, user: User | None = None) -> GitRepository:
    name = (user.name if user and user.name else None) or (user.email.split("@")[0] if user else "CodeFrog")
    return GitRepository(
        root,
        default_branch=validate_base_branch(repository.default_branch),
        remote_url=remote_url(repository),
        author_name=name,
        author_email=user.email if user else "codefrog@users.noreply.github.com",
    )


def _is_checkout(path: Path) -> bool:
    return path.is_dir() and not path.is_symlink() and (path / ".git").exists()


def find_checkout(base_directory: Path, repository: Repository) -> Path | None:
    """The existing checkout directory, or None. Never creates anything."""

    path = checkout_path(base_directory, repository.id)
    return path if _is_checkout(path) else None


def open_checkout(base_directory: Path, repository: Repository) -> Workspace | None:
    """A read-only view of the existing checkout (for read_file), or None if there is none yet."""

    path = find_checkout(base_directory, repository)
    return Workspace.attach(path, repository.id) if path else None


def require_git(base_directory: Path, repository: Repository, user: User | None = None) -> GitRepository:
    """A Git handle for an existing, verified checkout; a controlled error if it is missing or foreign."""

    path = find_checkout(base_directory, repository)
    if path is None:
        raise GitError("WORKSPACE_NOT_INITIALIZED", "The workspace has not been initialized. Initialize it first.", 409)
    git = _git(path, repository, user)
    git.verify_origin()
    return git


def get_state(base_directory: Path, repository: Repository) -> WorkspaceState:
    """What exists on disk, without creating or changing anything."""

    if find_checkout(base_directory, repository) is None:
        return WorkspaceState(exists=False, default_branch=repository.default_branch)
    git = require_git(base_directory, repository)
    return WorkspaceState(
        exists=True,
        branch=git.current_branch(),
        default_branch=repository.default_branch,
        uncommitted_changes=not git.status().clean,
        commit=git.head_commit(),
    )


def ensure_workspace(base_directory: Path, repository: Repository) -> Path:
    """Return the checkout, cloning the repository first only if it does not exist yet."""

    existing = find_checkout(base_directory, repository)
    if existing is not None:
        require_git(base_directory, repository)
        return existing
    return _clone(base_directory, repository)


def _clone(base_directory: Path, repository: Repository) -> Path:
    remote = resolve_remote(repository)
    branch = validate_base_branch(repository.default_branch)
    final = checkout_path(base_directory, repository.id)
    parent = final.parent
    staging = parent / f".cloning-{uuid.uuid4().hex}"
    try:
        parent.mkdir(parents=True, exist_ok=True)
        if final.exists() or final.is_symlink():  # something that is not a checkout: never overwrite it
            raise GitError("WORKSPACE_CONFLICT", "The workspace location is in use by something that is not a checkout.", 409)
        result = run_git(
            parent,
            ["clone", "--depth", str(CLONE_DEPTH), "--single-branch", "--no-tags", "--branch", branch, "--", remote.url, staging.name],
            token=remote.token,
            network=True,
            config=("core.symlinks=false", "core.autocrlf=false"),
        )
        if not result.ok:
            reason = result.stderr.lower()
            if "authentication" in reason or "could not read username" in reason or "403" in reason or "401" in reason:
                raise GitError("GITHUB_AUTH_FAILED", "GitHub rejected the stored credentials. Reconnect your GitHub account and try again.", 403)
            if "not found" in reason or "does not exist" in reason:
                raise GitError("REMOTE_NOT_FOUND", "The repository could not be found on GitHub.", 404)
            raise GitError("CLONE_FAILED", "The repository could not be cloned. Try again later.", 502)
        git = _git(staging, repository)
        if git.head_commit() is None or git.current_branch() != branch:
            raise GitError("CLONE_FAILED", "The cloned repository could not be verified.", 502)
        git.verify_origin()
        staging.rename(final)
    except OSError as error:
        logger.warning("Workspace clone failed (exception type=%s)", type(error).__name__)
        raise GitError("WORKSPACE_ERROR", "The workspace could not be prepared.") from None
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    logger.info("Workspace created repository_id=%s", repository.id)
    return final


def sync_workspace(base_directory: Path, repository: Repository) -> SyncOutcome:
    """Fetch, and fast-forward only a clean checkout of the default branch. Never discards local work."""

    git = require_git(base_directory, repository)
    remote = resolve_remote(repository)
    git.fetch(remote.token)
    if git.current_branch() != validate_base_branch(repository.default_branch):
        return SyncOutcome("skipped", "The workspace is not on the default branch.")
    if not git.status().clean:
        return SyncOutcome("skipped", "The workspace has uncommitted changes; they were left untouched.")
    return SyncOutcome("fast_forwarded" if git.fast_forward_default_branch() else "up_to_date")

