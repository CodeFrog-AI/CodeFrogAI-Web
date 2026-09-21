"""Tests for GitHub pull request creation, inspection, and diffs.

GitHub is a fake httpx transport (tests/github_api_fake.py) and Git remotes are local bare
repositories: no network, no real credentials.
"""

import json
import re
import uuid

import pytest

from app.api.routes import repository_git as git_routes
from app.api.routes import repository_pr as pr_routes
from app.api.routes import repositories as repository_routes
from app.agent.llm import LLMNotConfiguredError
from app.db.models import Repository
from app.embeddings.provider import EmbeddingNotConfiguredError
from app.integrations.github import pr_diff
from app.integrations.github.contents import GitHubAuthError
from tests.git_helpers import FAKE_TOKEN, fake_github  # noqa: F401
from tests.github_api_fake import RAW_BODY_MARKER, FakeGitHubAPI, file_json
from tests.test_git_workspace import UI, Repo, other_user_headers, ready, repo  # noqa: F401
from tests.test_repository_context import SECRET_LINES, SECRET_VALUES
from tests.test_semantic_search import client, create_repository, database  # noqa: F401

NL = chr(10)
BRANCH = "codefrog/task-1"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path, fake_github):
    def llm_unconfigured():
        raise LLMNotConfiguredError("not configured")

    def embeddings_unconfigured():
        raise EmbeddingNotConfiguredError("not configured")

    monkeypatch.setattr(repository_routes, "get_llm_provider", llm_unconfigured)
    monkeypatch.setattr(repository_routes, "get_embedding_provider", embeddings_unconfigured)
    monkeypatch.setattr(pr_routes, "get_llm_provider", llm_unconfigured)
    monkeypatch.setattr(pr_routes, "get_embedding_provider", embeddings_unconfigured)
    for module in (repository_routes, git_routes, pr_routes):
        monkeypatch.setattr(module, "get_workspace_root", lambda: tmp_path / "workspaces")


def make_api(repo, monkeypatch) -> FakeGitHubAPI:
    with repo.database() as session:
        row = session.get(Repository, repo.id)
        api = FakeGitHubAPI(row.github_repository_id, row.owner, row.name)
    monkeypatch.setattr(pr_routes, "pull_request_client", lambda repository: api.client())
    return api


@pytest.fixture
def api(repo, monkeypatch):
    return make_api(repo, monkeypatch)


def pr_url(repo, suffix="agent/pr"):
    return repo.url(suffix)


def create_pr(repo, title="feat: add the thing", body="Explains the change.", approved=True, headers=None, **extra):
    return repo.post("agent/pr", {"title": title, "body": body, "approved": approved, **extra}, headers=headers)


def pushed_branch(repo, api, name=BRANCH):
    """A committed and pushed codefrog branch, known to the fake GitHub."""

    repo.init()
    repo.write(UI, "render button with css layout" + NL + "label = 'Save'" + NL)
    assert repo.branch(name).status_code == 200
    assert repo.commit().status_code == 200
    assert repo.push().status_code == 200
    api.branches[name] = repo.git("rev-parse", "HEAD").strip()
    return name


# ------------------------------------------------------------------ PR creation


def test_a_pull_request_is_created_from_the_pushed_branch_to_the_default_branch(repo, api):
    pushed_branch(repo, api)

    response = create_pr(repo, title="feat: add JWT authentication", body="Implemented JWT authentication.")

    assert response.status_code == 200
    assert response.json() == {
        "repository_id": str(repo.id), "number": 101, "url": f"https://github.com/{api.owner}/{api.name}/pull/101",
        "title": "feat: add JWT authentication", "head_branch": BRANCH, "base_branch": "main", "state": "open",
        "created": True, "redactions": 0,
    }
    assert api.created == [{
        "title": "feat: add JWT authentication", "body": "Implemented JWT authentication.",
        "head": f"{api.owner}:{BRANCH}", "base": "main", "maintainer_can_modify": False,
    }]
    assert set(api.authorizations) == {f"Bearer {FAKE_TOKEN}"} and FAKE_TOKEN not in response.text


def test_the_base_branch_is_the_repositorys_actual_default_branch_on_github(repo, api):
    api.default_branch = "develop"
    pushed_branch(repo, api)

    body = create_pr(repo).json()

    assert body["base_branch"] == "develop" and api.created[0]["base"] == "develop"


@pytest.mark.parametrize("approval", [{"approved": False}, {}, {"approved": None}, {"approved": "true"}, {"approved": 1}], ids=["false", "missing", "null", "string", "number"])
def test_a_pull_request_requires_explicit_approval(repo, api, approval):
    pushed_branch(repo, api)
    before = list(api.requests)

    response = repo.post("agent/pr", {"title": "feat: x", **approval})

    assert response.status_code in (403, 422) and api.requests == before and api.posts() == []


@pytest.mark.parametrize("extra", [{"head": "codefrog/x"}, {"base": "main"}, {"repository_id": str(uuid.uuid4())}, {"owner": "someone"}, {"token": "abc"}, {"repository": "a/b"}], ids=lambda e: next(iter(e)))
def test_the_caller_cannot_choose_head_base_repository_owner_or_credentials(repo, api, extra):
    pushed_branch(repo, api)

    response = create_pr(repo, **extra)

    assert response.status_code == 422 and api.created == []


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"title": ""}, id="empty-title"),
        pytest.param({"title": "   "}, id="blank-title"),
        pytest.param({"title": "line one" + NL + "line two"}, id="newline-title"),
        pytest.param({"title": "bell" + chr(7)}, id="control-title"),
        pytest.param({"title": "x" * 201}, id="long-title"),
        pytest.param({"title": "ok", "body": "nul" + chr(0)}, id="nul-body"),
        pytest.param({"title": "ok", "body": "esc" + chr(27)}, id="control-body"),
        pytest.param({"title": "ok", "body": "x" * 10_001}, id="long-body"),
        pytest.param({"title": 5}, id="non-string-title"),
    ],
)
def test_titles_and_bodies_are_validated(repo, api, fields):
    pushed_branch(repo, api)

    response = repo.post("agent/pr", {"approved": True, **fields})

    assert response.status_code == 422 and api.created == []


def test_multiline_bodies_are_allowed_and_the_body_is_optional(repo, api):
    pushed_branch(repo, api)

    assert repo.post("agent/pr", {"title": "feat: x", "approved": True}).status_code == 200
    assert api.created[0]["body"] == ""


def test_authentication_and_ownership_are_required(repo, api, database):
    pushed_branch(repo, api)
    before = list(api.requests)

    assert create_pr(repo, headers={}).status_code == 401
    assert create_pr(repo, headers=other_user_headers(database)).status_code == 404
    unknown = repo.client.post(f"/api/v1/repositories/{uuid.uuid4()}/agent/pr", json={"title": "x", "approved": True}, headers=repo.headers)
    assert unknown.status_code == 404 and api.requests == before


def test_approval_is_checked_after_ownership(repo, api, database):
    pushed_branch(repo, api)

    assert create_pr(repo, approved=False, headers=other_user_headers(database)).status_code == 404


def test_a_workspace_is_required(repo, api):
    response = create_pr(repo)

    assert response.status_code == 409 and response.json()["error"]["code"] == "WORKSPACE_NOT_INITIALIZED" and api.requests == []


def test_a_pull_request_needs_a_codefrog_branch_and_never_uses_main_or_master(repo, api):
    repo.init()
    repo.write(UI, "changed" + NL)

    on_main = create_pr(repo)
    repo.git("switch", "-c", "master")
    on_master = create_pr(repo)
    repo.git("switch", "-c", "feature/mine")
    on_other = create_pr(repo)

    assert [r.status_code for r in (on_main, on_master, on_other)] == [403, 403, 403]
    assert {r.json()["error"]["code"] for r in (on_main, on_master, on_other)} == {"PROTECTED_BRANCH"}
    assert api.requests == []


def test_the_branch_must_have_been_pushed(repo, api):
    repo.init()
    repo.write(UI, "changed" + NL)
    repo.branch()
    repo.commit()

    response = create_pr(repo)

    assert response.status_code == 409 and response.json()["error"]["code"] == "BRANCH_NOT_PUSHED" and api.created == []


def test_a_branch_that_is_behind_the_local_commit_must_be_pushed_again(repo, api):
    pushed_branch(repo, api)
    repo.write(UI, "another change" + NL)
    repo.commit("feat: second commit")

    response = create_pr(repo)

    assert response.status_code == 409 and response.json()["error"]["code"] == "BRANCH_OUT_OF_DATE" and api.created == []


def test_a_repository_that_no_longer_matches_github_is_refused(repo, api):
    pushed_branch(repo, api)
    api.repo_owner_override = "somebody-else"

    response = create_pr(repo)

    assert response.status_code == 409 and api.created == []


def test_an_existing_open_pull_request_is_returned_instead_of_a_duplicate(repo, api):
    pushed_branch(repo, api)
    first = create_pr(repo).json()

    second = create_pr(repo, title="a different title").json()

    assert first["created"] is True and second["created"] is False and second["number"] == first["number"]
    assert len(api.posts()) == 1 and second["title"] == "feat: add the thing"
    assert create_pr(repo).json()["number"] == first["number"] and len(api.posts()) == 1


def test_a_pull_request_that_already_exists_on_github_is_not_recreated(repo, api):
    pushed_branch(repo, api)
    api.add_pull(7, title="made by hand", head=BRANCH, base="main")

    body = create_pr(repo).json()

    assert (body["number"], body["created"], body["title"]) == (7, False, "made by hand") and api.posts() == []


def test_a_pull_request_to_a_different_base_or_a_closed_one_does_not_count(repo, api):
    pushed_branch(repo, api)
    api.add_pull(7, head=BRANCH, base="release")
    api.add_pull(8, head=BRANCH, base="main", state="closed")

    body = create_pr(repo).json()

    assert body["created"] is True and body["number"] == 9


def test_a_retry_that_races_with_an_earlier_request_returns_the_existing_pull_request(repo, api):
    pushed_branch(repo, api)
    api.add_pull(7, head=BRANCH, base="main")
    api.hide_next_find = True  # the first lookup misses it; GitHub then answers "already exists"

    body = create_pr(repo).json()

    assert (body["number"], body["created"]) == (7, False) and len(api.posts()) == 1


@pytest.mark.parametrize(
    ("failure", "status", "code"),
    [
        pytest.param(401, 403, "FORBIDDEN", id="unauthorized"),
        pytest.param(403, 403, "FORBIDDEN", id="forbidden"),
        pytest.param(404, 404, "RESOURCE_NOT_FOUND", id="not-found"),
        pytest.param(422, 400, "BAD_REQUEST", id="validation"),
        pytest.param(429, 429, "RATE_LIMITED", id="too-many"),
        pytest.param((403, {"X-RateLimit-Remaining": "0"}), 429, "RATE_LIMITED", id="rate-limit-403"),
        pytest.param(500, 502, "BAD_GATEWAY", id="server-error"),
        pytest.param(503, 502, "BAD_GATEWAY", id="unavailable"),
        pytest.param("timeout", 502, "BAD_GATEWAY", id="timeout"),
    ],
)
@pytest.mark.parametrize("operation", ["repo", "branch", "find", "create"])
def test_github_failures_become_controlled_errors_that_leak_nothing(repo, api, failure, status, code, operation):
    pushed_branch(repo, api)
    api.fail[operation] = failure
    expected_status = 409 if (failure == 404 and operation == "branch") else status

    response = create_pr(repo)

    assert response.status_code == expected_status
    assert response.json()["error"]["code"] in (code, "BRANCH_NOT_PUSHED", "RESOURCE_NOT_FOUND")
    text = response.text
    assert RAW_BODY_MARKER not in text and FAKE_TOKEN not in text and "Bearer" not in text and "Authorization" not in text
    assert str(repo.tmp_path) not in text and "api.github.com" not in text
    assert api.created == [] or operation != "create"


def test_a_missing_github_connection_is_a_controlled_error(repo, api, monkeypatch):
    pushed_branch(repo, api)

    def unauthorized(repository):
        raise GitHubAuthError("No stored GitHub token")

    monkeypatch.setattr(pr_routes, "pull_request_client", unauthorized)

    response = create_pr(repo)

    assert response.status_code == 403 and "token" not in response.text.lower()


def test_secrets_in_the_title_and_body_are_redacted_before_they_reach_github(repo, api):
    pushed_branch(repo, api)
    title = "fix: use " + SECRET_LINES[1].split(chr(34))[1]
    body = NL.join(SECRET_LINES)

    response = create_pr(repo, title=title, body=body)

    sent = json.dumps(api.created)
    assert response.status_code == 200 and response.json()["redactions"] >= 2
    assert not [value for value in SECRET_VALUES if value in sent + response.text]


def test_nothing_creates_a_pull_request_automatically(repo, api, monkeypatch):
    pushed_branch(repo, api)  # branch, commit, and push all happened

    assert api.posts() == [] and not any(path.endswith("/pulls") for _, path in api.requests)


def test_the_head_is_always_a_codefrog_branch_in_the_connected_repository(repo, api):
    pushed_branch(repo, api, "codefrog/fix-auth")

    create_pr(repo)

    [payload] = api.created
    assert re.fullmatch(rf"{api.owner}:codefrog/[A-Za-z0-9._/-]+", payload["head"])


# ------------------------------------------------------------------ reading a PR


def test_a_pull_request_can_be_read(repo, api):
    api.add_pull(5, title="Add auth", body="Body text", head="codefrog/a", base="main", draft=True)

    response = repo.get("agent/pr/5")

    assert response.status_code == 200
    assert response.json() == {
        "repository_id": str(repo.id), "number": 5, "url": f"https://github.com/{api.owner}/{api.name}/pull/5", "title": "Add auth",
        "body": "Body text", "body_redactions": 0, "state": "open", "draft": True, "head_branch": "codefrog/a",
        "base_branch": "main", "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-02T00:00:00Z",
    }


def test_merged_and_closed_states_are_reported(repo, api):
    api.add_pull(5, state="closed", merged=True)
    api.add_pull(6, state="closed")

    assert repo.get("agent/pr/5").json()["state"] == "merged" and repo.get("agent/pr/6").json()["state"] == "closed"


def test_a_pull_request_body_is_redacted_and_bounded(repo, api):
    api.add_pull(5, body=NL.join(SECRET_LINES) + NL + "x" * 20_000)

    response = repo.get("agent/pr/5")

    assert not [value for value in SECRET_VALUES if value in response.text] and response.json()["body_redactions"] >= 1
    assert len(response.json()["body"]) <= 10_000


def test_reading_a_pull_request_needs_ownership_and_a_real_pull_request(repo, api, database):
    api.add_pull(5)
    before = list(api.requests)

    assert repo.get("agent/pr/5", headers={}).status_code == 401
    assert repo.get("agent/pr/5", headers=other_user_headers(database)).status_code == 404
    assert api.requests == before
    assert repo.get("agent/pr/999").status_code == 404
    for bad in ("0", "-1", "abc", "1.5"):
        assert repo.get(f"agent/pr/{bad}").status_code == 422


def test_reading_only_reaches_the_connected_repository(repo, api):
    api.add_pull(5)

    repo.get("agent/pr/5")

    assert {path for _, path in api.requests} == {f"/repos/{api.owner}/{api.name}/pulls/5"}
    assert api.methods() == {"GET"}


def test_get_errors_are_controlled(repo, api):
    api.add_pull(5)
    for failure, status in ((401, 403), (429, 429), (500, 502), ("timeout", 502)):
        api.fail["get"] = failure
        response = repo.get("agent/pr/5")
        assert response.status_code == status and RAW_BODY_MARKER not in response.text and FAKE_TOKEN not in response.text


# ------------------------------------------------------------------ PR diff


def diff_of(repo, number=5):
    response = repo.get(f"agent/pr/{number}/diff")
    assert response.status_code == 200, response.text
    return response.json()


def test_changed_files_are_returned_with_counts_and_patches(repo, api):
    api.add_pull(5)
    api.files[5] = [
        file_json("backend/app/auth.py", additions=20, deletions=5),
        file_json("new.py", status="added", additions=3, deletions=0),
        file_json("gone.py", status="removed", additions=0, deletions=4),
        file_json("moved.py", status="renamed", previous="old_name.py"),
    ]

    body = diff_of(repo)

    files = {f["path"]: f for f in body["files"]}
    assert (files["backend/app/auth.py"]["status"], files["backend/app/auth.py"]["additions"], files["backend/app/auth.py"]["deletions"]) == ("modified", 20, 5)
    assert files["new.py"]["status"] == "added" and files["gone.py"]["status"] == "deleted"
    assert files["moved.py"]["status"] == "renamed" and files["moved.py"]["previous_path"] == "old_name.py"
    assert "+added" in files["new.py"]["patch"] and body["total_files"] == 4 and body["withheld"] == 0 and body["truncated"] is False


def test_a_large_patch_is_cut_at_the_size_limit(repo, api):
    api.add_pull(5)
    api.files[5] = [file_json("big.py", patch="@@ -1 +1,9000 @@" + NL + NL.join("+line %d" % i for i in range(9000)))]

    [entry] = diff_of(repo)["files"]

    assert entry["patch_truncated"] is True and len(entry["patch"]) <= pr_diff.MAX_PATCH_CHARS_PER_FILE


def test_the_total_patch_size_is_limited(repo, api):
    api.add_pull(5)
    api.files[5] = [file_json(f"f{i}.py", patch="@@ -1 +1,3000 @@" + NL + NL.join("+x%d" % j for j in range(3000))) for i in range(12)]

    body = diff_of(repo)

    assert sum(len(f["patch"]) for f in body["files"]) <= pr_diff.MAX_TOTAL_PATCH_CHARS and body["truncated"] is True


def test_the_number_of_files_is_limited_and_all_github_pages_are_read(repo, api):
    api.add_pull(5)
    api.files[5] = [file_json(f"src/file{i:03d}.py") for i in range(230)]

    body = diff_of(repo)

    assert len(body["files"]) == 100 and body["total_files"] == 230 and body["truncated"] is True
    assert len([1 for _, path in api.requests if path.endswith("/files")]) == 3


def test_protected_and_unsafe_files_are_counted_but_never_shown(repo, api):
    api.add_pull(5)
    api.files[5] = [
        file_json("app/ok.py"),
        file_json(".env", patch="@@ -1 +1 @@" + NL + "+TOPSECRET=hunter22secret"),
        file_json("keys/server.pem"),
        file_json("deploy/credentials.json"),
        file_json("../escape.py"),
        file_json("a" + chr(92) + "b.py"),
    ]

    response = repo.get("agent/pr/5/diff")

    body = response.json()
    assert [f["path"] for f in body["files"]] == ["app/ok.py"] and body["withheld"] == 5
    assert "TOPSECRET" not in response.text and "server.pem" not in response.text and ".env" not in response.text


def test_secrets_in_patches_are_redacted(repo, api):
    api.add_pull(5)
    api.files[5] = [file_json("app/config.py", patch="@@ -1 +1,9 @@" + NL + NL.join("+" + line for line in SECRET_LINES))]

    response = repo.get("agent/pr/5/diff")

    assert not [value for value in SECRET_VALUES if value in response.text]


def test_files_without_a_patch_are_reported_as_such(repo, api):
    api.add_pull(5)
    api.files[5] = [file_json("logo.png", patch=None, additions=0, deletions=0)]

    [entry] = diff_of(repo)["files"]

    assert entry["patch_available"] is False and entry["patch"] == ""


def test_diff_errors_and_ownership(repo, api, database):
    api.add_pull(5)

    assert repo.get("agent/pr/999/diff").status_code == 404
    assert repo.get("agent/pr/5/diff", headers={}).status_code == 401
    assert repo.get("agent/pr/5/diff", headers=other_user_headers(database)).status_code == 404
    api.fail["files"] = 500
    response = repo.get("agent/pr/5/diff")
    assert response.status_code == 502 and RAW_BODY_MARKER not in response.text and FAKE_TOKEN not in response.text


def test_patch_line_helpers_follow_the_new_side_of_the_hunks():
    patch = "@@ -10,4 +20,5 @@" + NL + " keep" + NL + "-old" + NL + "+new one" + NL + "+new two" + NL + " keep2" + NL + "@@ -50 +60 @@" + NL + "+later"

    assert pr_diff.new_side_lines(patch) == {20, 21, 22, 23, 60}
    assert "    21| +new one" in pr_diff.annotate_patch(patch) and "      | -old" in pr_diff.annotate_patch(patch)


def test_the_pull_request_client_only_talks_to_github_with_the_stored_token_and_never_logs_it(repo, api, caplog):
    api.add_pull(5)
    with caplog.at_level("DEBUG"):
        repo.get("agent/pr/5")
        repo.get("agent/pr/999")

    assert FAKE_TOKEN not in caplog.text
