"""Read-only GitHub repository content access (git trees and blobs)."""

import base64
import binascii
from dataclasses import dataclass
from urllib.parse import quote

import httpx

GITHUB_API_URL = "https://api.github.com"
REQUEST_TIMEOUT_SECONDS = 15.0
SYMLINK_MODE = "120000"


class GitHubContentError(RuntimeError):
    """A GitHub request failed; messages never include tokens or response bodies."""


class GitHubAuthError(GitHubContentError):
    """GitHub rejected the credentials or denied access."""


class GitHubNotFoundError(GitHubContentError):
    """The repository, ref, or blob does not exist or is not visible."""


class GitHubEmptyRepositoryError(GitHubContentError):
    """The repository has no commits."""


@dataclass(frozen=True)
class GitHubTreeEntry:
    path: str
    sha: str
    size: int | None


@dataclass(frozen=True)
class GitHubTree:
    entries: list[GitHubTreeEntry]
    truncated: bool


class GitHubContentClient:
    """Fetch repository files with an optional user token (public repos need none)."""

    def __init__(
        self, access_token: str | None = None, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        self._client = httpx.Client(
            base_url=GITHUB_API_URL,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
            transport=transport,
        )

    def __enter__(self) -> "GitHubContentClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self._client.close()

    def get_tree(self, owner: str, name: str, ref: str) -> GitHubTree:
        """List every file in a branch; an empty repository yields no entries."""

        path = f"/repos/{quote(owner, safe='')}/{quote(name, safe='')}/git/trees/{quote(ref, safe='')}"
        try:
            payload = self._get_json(path, params={"recursive": "1"})
        except GitHubEmptyRepositoryError:
            return GitHubTree(entries=[], truncated=False)
        entries = [
            GitHubTreeEntry(path=item["path"], sha=item["sha"], size=item.get("size"))
            for item in payload.get("tree", [])
            if item.get("type") == "blob" and item.get("mode") != SYMLINK_MODE
        ]
        return GitHubTree(entries=entries, truncated=bool(payload.get("truncated")))

    def get_file_content(self, owner: str, name: str, sha: str) -> bytes:
        """Download a blob's raw bytes."""

        path = f"/repos/{quote(owner, safe='')}/{quote(name, safe='')}/git/blobs/{quote(sha, safe='')}"
        payload = self._get_json(path)
        if payload.get("encoding") != "base64":
            raise GitHubContentError("GitHub returned an unsupported file encoding")
        try:
            return base64.b64decode(payload.get("content", ""), validate=False)
        except (binascii.Error, ValueError):
            raise GitHubContentError("GitHub returned unreadable file content") from None

    def _get_json(self, path: str, params: dict[str, str] | None = None) -> dict:
        try:
            response = self._client.get(path, params=params)
        except httpx.HTTPError:
            raise GitHubContentError("GitHub request failed") from None

        status = response.status_code
        if status == 401:
            raise GitHubAuthError("GitHub authentication failed")
        if status == 429 or (status == 403 and response.headers.get("X-RateLimit-Remaining") == "0"):
            raise GitHubContentError("GitHub rate limit exceeded")
        if status == 403:
            raise GitHubAuthError("GitHub denied access")
        if status == 404:
            raise GitHubNotFoundError("GitHub resource not found")
        if status == 409:
            raise GitHubEmptyRepositoryError("GitHub repository is empty")
        if status >= 400:
            raise GitHubContentError("GitHub request failed")
        try:
            payload = response.json()
        except ValueError:
            raise GitHubContentError("GitHub returned an invalid response") from None
        if not isinstance(payload, dict):
            raise GitHubContentError("GitHub returned an invalid response")
        return payload
