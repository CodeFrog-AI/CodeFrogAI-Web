"""Creating and reading pull requests for a connected repository.

The server decides everything that matters: the repository (the caller's connected one, and
confirmed against GitHub by its numeric id), the head branch (only the workspace's current
`codefrog/` branch, which must already be pushed), the base branch (the repository's actual
default branch on GitHub), and the credentials (the stored OAuth token). The caller supplies
only a title and a body, which are redacted before they leave the server.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from app.context.redaction import redact_secrets
from app.core.exceptions import (
    BadGatewayError,
    BadRequestError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    RateLimitedError,
)
from app.db.models import Repository
from app.git import GitError, GitRepository, validate_base_branch
from app.integrations.github.contents import GitHubAuthError, GitHubContentError, GitHubNotFoundError
from app.integrations.github.pull_requests import (
    GitHubPullRequestClient,
    GitHubRateLimitError,
    GitHubValidationError,
    PullRequest,
)
from app.integrations.github.tokens import decrypt_access_token

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CreatedPullRequest:
    pull_request: PullRequest
    created: bool
    redactions: int


def pull_request_client(repository: Repository) -> GitHubPullRequestClient:
    """A GitHub client authenticated with the repository owner's stored token (the existing OAuth connection)."""

    token = decrypt_access_token(repository.github_account.access_token_encrypted)
    if token is None:
        raise GitHubAuthError("No stored GitHub token")
    return GitHubPullRequestClient(token)


@contextmanager
def pull_request_errors(not_found: str = "The pull request was not found on GitHub.") -> Iterator[None]:
    """Translate GitHub failures into controlled API errors (never raw bodies, tokens, or headers)."""

    try:
        yield
    except GitHubAuthError:
        raise ForbiddenError("GitHub access could not be verified. Reconnect GitHub and try again.") from None
    except GitHubNotFoundError:
        raise NotFoundError(not_found) from None
    except GitHubRateLimitError:
        raise RateLimitedError("GitHub rate limit reached. Try again later.") from None
    except GitHubValidationError:
        raise BadRequestError(
            "GitHub rejected the pull request. Make sure the branch has commits that differ from the base branch."
        ) from None
    except GitHubContentError:
        raise BadGatewayError("GitHub request failed. Try again later.") from None


def confirmed_repository(client: GitHubPullRequestClient, repository: Repository) -> tuple[str, str, str]:
    """(owner, name, default branch), verified against GitHub by numeric id so a renamed or transferred repository is refused."""

    info = client.get_repository_by_id(repository.github_repository_id)
    if (info.owner.lower(), info.name.lower()) != (repository.owner.lower(), repository.name.lower()):
        raise ConflictError("The connected repository no longer matches the one on GitHub. Reconnect it.")
    return repository.owner, repository.name, validate_base_branch(info.default_branch)


def create_pull_request(
    client: GitHubPullRequestClient, repository: Repository, git: GitRepository, *, title: str, body: str
) -> CreatedPullRequest:
    """Open a pull request from the workspace's pushed `codefrog/` branch to the default branch, once."""

    head = git.current_pull_request_branch()
    owner, name, base = confirmed_repository(client, repository)
    if head == base:
        raise GitError("PROTECTED_BRANCH", "The head branch cannot be the base branch.", 403)
    local_commit = git.head_commit()
    try:
        remote_commit = client.get_branch_sha(owner, name, head)
    except GitHubNotFoundError:
        raise GitError("BRANCH_NOT_PUSHED", "The branch has not been pushed to GitHub yet. Push it first.", 409) from None
    if remote_commit != local_commit:
        raise GitError("BRANCH_OUT_OF_DATE", "The branch on GitHub differs from the workspace. Push it again first.", 409)

    existing = client.find_open_pull_request(owner, name, head, base)
    if existing is not None:
        return CreatedPullRequest(existing, created=False, redactions=0)

    title, title_redactions = redact_secrets(title)
    body, body_redactions = redact_secrets(body)
    try:
        pull_request = client.create_pull_request(owner, name, title=title, body=body, head_branch=head, base_branch=base)
    except GitHubValidationError as error:
        if error.duplicate:  # a retry raced with an earlier request: return that pull request
            existing = client.find_open_pull_request(owner, name, head, base)
            if existing is not None:
                return CreatedPullRequest(existing, created=False, redactions=0)
        raise
    logger.info("Pull request created repository_id=%s number=%d", repository.id, pull_request.number)
    return CreatedPullRequest(pull_request, created=True, redactions=title_redactions + body_redactions)
