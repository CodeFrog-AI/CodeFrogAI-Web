"""GitHub pull request access: create, find, read, and list the changed files of pull requests.

Built on the existing GitHub client (same token handling, headers, and timeout). Errors are
classified into a few safe categories; response bodies, tokens, and headers are never
included in error messages.
"""

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from app.integrations.github.contents import (
    GitHubAuthError,
    GitHubContentClient,
    GitHubContentError,
    GitHubNotFoundError,
    GitHubRepository,
    _parse_repository,
)

FILES_PER_PAGE = 100
MAX_FILE_PAGES = 3


class GitHubRateLimitError(GitHubContentError):
    """GitHub throttled the request."""


class GitHubUnavailableError(GitHubContentError):
    """GitHub could not be reached or failed (network error, timeout, 5xx)."""


class GitHubValidationError(GitHubContentError):
    """GitHub rejected the request as invalid (HTTP 422)."""

    def __init__(self, message: str, *, duplicate: bool = False) -> None:
        super().__init__(message)
        self.duplicate = duplicate


@dataclass(frozen=True)
class PullRequest:
    number: int
    url: str
    title: str
    body: str
    state: str  # open | closed | merged
    head_branch: str
    base_branch: str
    draft: bool
    created_at: str | None
    updated_at: str | None


@dataclass(frozen=True)
class PullRequestFile:
    path: str
    status: str
    additions: int
    deletions: int
    patch: str | None
    previous_path: str | None = None


@dataclass(frozen=True)
class PullRequestDiff:
    files: list[PullRequestFile]
    truncated: bool  # GitHub has more files than were fetched


class GitHubPullRequestClient(GitHubContentClient):
    """Pull request operations for one repository, authenticated as the connected user."""

    def get_repository_by_id(self, github_repository_id: int) -> GitHubRepository:
        """The repository by numeric id, with the same failure classification as the other pull request calls."""

        return _parse_repository(self._call("GET", f"/repositories/{int(github_repository_id)}"))

    def get_branch_sha(self, owner: str, name: str, branch: str) -> str:
        """The commit a branch points to on GitHub (NotFound if it was never pushed)."""

        payload = self._call("GET", f"{_repo(owner, name)}/branches/{quote(branch, safe='/')}")
        try:
            return str(payload["commit"]["sha"])
        except (KeyError, TypeError):
            raise GitHubContentError("GitHub returned an invalid response") from None

    def find_open_pull_request(self, owner: str, name: str, head_branch: str, base_branch: str) -> PullRequest | None:
        """The open pull request for exactly this head and base, if one exists."""

        items = self._call(
            "GET",
            f"{_repo(owner, name)}/pulls",
            params={"state": "open", "head": f"{owner}:{head_branch}", "base": base_branch, "per_page": "5"},
        )
        if not isinstance(items, list):
            raise GitHubContentError("GitHub returned an invalid response")
        for item in items:
            pull_request = _parse_pull_request(item)
            if pull_request.head_branch == head_branch and pull_request.base_branch == base_branch:
                return pull_request
        return None

    def create_pull_request(self, owner: str, name: str, *, title: str, body: str, head_branch: str, base_branch: str) -> PullRequest:
        payload = self._call(
            "POST",
            f"{_repo(owner, name)}/pulls",
            json={"title": title, "body": body, "head": f"{owner}:{head_branch}", "base": base_branch, "maintainer_can_modify": False},
        )
        return _parse_pull_request(payload)

    def get_pull_request(self, owner: str, name: str, number: int) -> PullRequest:
        return _parse_pull_request(self._call("GET", f"{_repo(owner, name)}/pulls/{int(number)}"))

    def get_pull_request_diff(self, owner: str, name: str, number: int) -> PullRequestDiff:
        """The changed files with their patches (at most a few hundred; `truncated` says if there are more)."""

        files: list[PullRequestFile] = []
        truncated = False
        for page in range(1, MAX_FILE_PAGES + 1):
            items = self._call(
                "GET",
                f"{_repo(owner, name)}/pulls/{int(number)}/files",
                params={"per_page": str(FILES_PER_PAGE), "page": str(page)},
            )
            if not isinstance(items, list):
                raise GitHubContentError("GitHub returned an invalid response")
            files.extend(_parse_file(item) for item in items)
            if len(items) < FILES_PER_PAGE:
                break
            truncated = page == MAX_FILE_PAGES
        return PullRequestDiff(files, truncated)

    def _call(self, method: str, path: str, *, params: dict[str, str] | None = None, json: dict[str, Any] | None = None) -> Any:
        try:
            response = self._client.request(method, path, params=params, json=json)
        except httpx.HTTPError:
            raise GitHubUnavailableError("GitHub request failed") from None
        status = response.status_code
        if status == 401:
            raise GitHubAuthError("GitHub authentication failed")
        if status == 429 or (status == 403 and _rate_limited(response)):
            raise GitHubRateLimitError("GitHub rate limit exceeded")
        if status == 403:
            raise GitHubAuthError("GitHub denied access")
        if status == 404:
            raise GitHubNotFoundError("GitHub resource not found")
        if status == 422:
            raise GitHubValidationError("GitHub rejected the request", duplicate=_mentions_existing(response))
        if status >= 500:
            raise GitHubUnavailableError("GitHub is unavailable")
        if status >= 400:
            raise GitHubContentError("GitHub request failed")
        try:
            return response.json()
        except ValueError:
            raise GitHubContentError("GitHub returned an invalid response") from None


def _repo(owner: str, name: str) -> str:
    return f"/repos/{quote(owner, safe='')}/{quote(name, safe='')}"


def _rate_limited(response: httpx.Response) -> bool:
    if response.headers.get("X-RateLimit-Remaining") == "0" or "Retry-After" in response.headers:
        return True
    return "rate limit" in response.text[:2_000].lower()


def _mentions_existing(response: httpx.Response) -> bool:
    return "already exists" in response.text[:4_000].lower()


def _parse_pull_request(item: Any) -> PullRequest:
    try:
        state = "merged" if item.get("merged") or item.get("merged_at") else str(item["state"])
        return PullRequest(
            number=int(item["number"]),
            url=str(item["html_url"]),
            title=str(item.get("title") or ""),
            body=str(item.get("body") or ""),
            state=state,
            head_branch=str(item["head"]["ref"]),
            base_branch=str(item["base"]["ref"]),
            draft=bool(item.get("draft", False)),
            created_at=item.get("created_at"),
            updated_at=item.get("updated_at"),
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        raise GitHubContentError("GitHub returned an invalid response") from None


def _parse_file(item: Any) -> PullRequestFile:
    try:
        return PullRequestFile(
            path=str(item["filename"]),
            status=str(item.get("status", "modified")),
            additions=int(item.get("additions", 0)),
            deletions=int(item.get("deletions", 0)),
            patch=item.get("patch") if isinstance(item.get("patch"), str) else None,
            previous_path=item.get("previous_filename") if isinstance(item.get("previous_filename"), str) else None,
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        raise GitHubContentError("GitHub returned an invalid response") from None
