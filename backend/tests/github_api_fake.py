"""A fake GitHub REST API (httpx MockTransport) for pull request tests. Nothing touches the network."""

import json
import re

import httpx

from app.integrations.github.pull_requests import GitHubPullRequestClient
from tests.git_helpers import FAKE_TOKEN

RAW_BODY_MARKER = "RAW-GITHUB-BODY-MUST-NOT-LEAK"
_REPO = r"/repos/(?P<owner>[^/]+)/(?P<name>[^/]+)"


def pr_json(number, *, title="A change", body="", head="codefrog/task-1", base="main", state="open", merged=False, owner="me", name="project", draft=False):
    return {
        "number": number,
        "html_url": f"https://github.com/{owner}/{name}/pull/{number}",
        "title": title,
        "body": body,
        "state": state,
        "merged": merged,
        "draft": draft,
        "head": {"ref": head},
        "base": {"ref": base},
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-02T00:00:00Z",
    }


def file_json(path, patch="@@ -1,2 +1,3 @@" + chr(10) + " context" + chr(10) + "+added" + chr(10) + " more", status="modified", additions=1, deletions=0, previous=None):
    item = {"filename": path, "status": status, "additions": additions, "deletions": deletions}
    if patch is not None:
        item["patch"] = patch
    if previous:
        item["previous_filename"] = previous
    return item


class FakeGitHubAPI:
    """In-memory GitHub. Set `fail[op]` to a status code, (status, headers) or "timeout" to inject failures."""

    def __init__(self, github_id: int, owner: str, name: str) -> None:
        self.github_id, self.owner, self.name = github_id, owner, name
        self.default_branch = "main"
        self.branches: dict[str, str] = {}
        self.pulls: dict[int, dict] = {}
        self.files: dict[int, list[dict]] = {}
        self.fail: dict[str, object] = {}
        self.requests: list[tuple[str, str]] = []
        self.created: list[dict] = []
        self.hide_next_find = False
        self.authorizations: list[str | None] = []
        self.repo_owner_override: str | None = None

    def client(self) -> GitHubPullRequestClient:
        return GitHubPullRequestClient(FAKE_TOKEN, transport=httpx.MockTransport(self.handle))

    def add_pull(self, number, **kwargs):
        self.pulls[number] = pr_json(number, owner=self.owner, name=self.name, **kwargs)
        return self.pulls[number]

    def posts(self):
        return [entry for entry in self.requests if entry[0] == "POST"]

    def methods(self):
        return {method for method, _ in self.requests}

    # ------------------------------------------------------------------ the handler

    def _failure(self, op):
        spec = self.fail.get(op)
        if spec is None:
            return None
        if spec == "timeout":
            raise httpx.ConnectTimeout("timed out")
        status, headers = (spec, {}) if isinstance(spec, int) else spec
        return httpx.Response(status, headers=headers, json={"message": RAW_BODY_MARKER + " " + FAKE_TOKEN, "errors": [{"message": RAW_BODY_MARKER}]})

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.requests.append((request.method, path))
        self.authorizations.append(request.headers.get("Authorization"))
        if request.headers.get("Authorization") != f"Bearer {FAKE_TOKEN}":
            return httpx.Response(401, json={"message": "Bad credentials"})

        if match := re.fullmatch(r"/repositories/(\d+)", path):
            if failure := self._failure("repo"):
                return failure
            if int(match.group(1)) != self.github_id:
                return httpx.Response(404, json={})
            return httpx.Response(200, json={"id": self.github_id, "owner": {"login": self.repo_owner_override or self.owner}, "name": self.name, "default_branch": self.default_branch, "private": False})

        if match := re.fullmatch(_REPO + r"/branches/(?P<branch>.+)", path):
            if failure := self._failure("branch"):
                return failure
            branch = match.group("branch")
            if branch not in self.branches:
                return httpx.Response(404, json={"message": "Branch not found"})
            return httpx.Response(200, json={"name": branch, "commit": {"sha": self.branches[branch]}})

        if re.fullmatch(_REPO + r"/pulls", path):
            if request.method == "GET":
                if failure := self._failure("find"):
                    return failure
                if self.hide_next_find:
                    self.hide_next_find = False
                    return httpx.Response(200, json=[])
                head = request.url.params.get("head", "")
                base = request.url.params.get("base")
                matches = [
                    item for item in self.pulls.values()
                    if item["state"] == "open" and f"{self.owner}:{item['head']['ref']}" == head and item["base"]["ref"] == base
                ]
                return httpx.Response(200, json=matches)
            if failure := self._failure("create"):
                return failure
            payload = json.loads(request.content)
            self.created.append(payload)
            head_branch = payload["head"].split(":", 1)[1]
            if any(item["state"] == "open" and item["head"]["ref"] == head_branch and item["base"]["ref"] == payload["base"] for item in self.pulls.values()):
                return httpx.Response(422, json={"message": "Validation Failed", "errors": [{"message": "A pull request already exists for " + payload["head"]}]})
            number = max(self.pulls, default=100) + 1
            self.add_pull(number, title=payload["title"], body=payload["body"], head=head_branch, base=payload["base"])
            return httpx.Response(201, json=self.pulls[number])

        if match := re.fullmatch(_REPO + r"/pulls/(?P<number>\d+)/files", path):
            if failure := self._failure("files"):
                return failure
            number = int(match.group("number"))
            if number not in self.pulls:
                return httpx.Response(404, json={})
            page = int(request.url.params.get("page", "1"))
            per_page = int(request.url.params.get("per_page", "30"))
            files = self.files.get(number, [])
            return httpx.Response(200, json=files[(page - 1) * per_page : page * per_page])

        if match := re.fullmatch(_REPO + r"/pulls/(?P<number>\d+)", path):
            if failure := self._failure("get"):
                return failure
            number = int(match.group("number"))
            if number not in self.pulls:
                return httpx.Response(404, json={"message": "Not Found"})
            return httpx.Response(200, json=self.pulls[number])

        return httpx.Response(404, json={"message": "unrouted"})
