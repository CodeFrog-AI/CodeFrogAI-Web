"""Tests for the persistent workspace and the Git layer.

GitHub is a local bare repository (see tests/git_helpers.py) and the LLM is a scripted fake:
no network and no real provider is ever used. Real Git runs against temporary directories.
"""

import json
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

from app.agent.llm import LLMNotConfiguredError
from app.api.routes import repositories as repository_routes
from app.api.routes import repository_git as git_routes
from app.core.exceptions import ConflictError
from app.embeddings.provider import EmbeddingNotConfiguredError
from app.git import GitError, validate_branch_name, validate_commit_message
from app.git import repository as git_repository
from app.git import runner as git_runner
from app.git.runner import GitResult
from app.tools import tool_definitions
from app.workspace import Workspace, WriteScope, exclusive_workspace
from app.workspace import service as workspace_service
from tests.git_helpers import FAKE_TOKEN, add_remote_commit, fake_github, git, make_remote  # noqa: F401
from tests.test_agent_code_editing import FILES, REQUEST, edit, execute, plan
from tests.test_agent_loop import FakeLLM, call, calls, say, tool_result, use_llm
from tests.test_repository_context import SECRET_LINES, SECRET_VALUES
from tests.test_semantic_search import client, create_repository, database  # noqa: F401

UI = "app/ui.py"
BACKEND = Path(__file__).resolve().parents[1]
APP = BACKEND / "app"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path, fake_github):
    """No real providers; workspaces live in this test's temporary directory."""

    def llm_unconfigured():
        raise LLMNotConfiguredError("not configured")

    def embeddings_unconfigured():
        raise EmbeddingNotConfiguredError("not configured")

    monkeypatch.setattr(repository_routes, "get_llm_provider", llm_unconfigured)
    monkeypatch.setattr(repository_routes, "get_embedding_provider", embeddings_unconfigured)
    for module in (repository_routes, git_routes):
        monkeypatch.setattr(module, "get_workspace_root", lambda: tmp_path / "workspaces")


class Repo:
    """A repository the caller owns, with helpers for its API and its checkout."""

    def __init__(self, client, database, tmp_path, fake_github, files=None, email="me@example.com"):
        self.client, self.database, self.tmp_path, self.fake = client, database, tmp_path, fake_github
        self.id, self.headers = create_repository(database, email=email, files=FILES if files is None else files)

    @property
    def base(self) -> Path:
        return self.tmp_path / "workspaces"

    @property
    def root(self) -> Path:
        return self.base / "repositories" / str(self.id)

    @property
    def remote(self) -> Path:
        with self.database() as session:
            from app.db.models import Repository

            return self.fake.remote_path(session.get(Repository, self.id))

    def url(self, suffix):
        return f"/api/v1/repositories/{self.id}/{suffix}"

    def get(self, suffix, headers=None, **params):
        return self.client.get(self.url(suffix), params=params, headers=self.headers if headers is None else headers)

    def post(self, suffix, body=None, headers=None):
        return self.client.post(self.url(suffix), json=body, headers=self.headers if headers is None else headers)

    def init(self):
        response = self.post("workspace")
        assert response.status_code == 200, response.text
        return response

    def git(self, *arguments):
        return git(self.root, *arguments)

    def read(self, path):
        return (self.root / path).read_text()

    def write(self, path, text):
        (self.root / path).parent.mkdir(parents=True, exist_ok=True)
        (self.root / path).write_bytes(text.encode())

    def branch(self, name="codefrog/task-1", approved=True):
        return self.post("agent/branch", {"name": name, "approved": approved})

    def commit(self, message="feat: change the label", approved=True):
        return self.post("agent/commit", {"message": message, "approved": approved})

    def push(self, approved=True):
        return self.post("agent/push", {"approved": approved})

    def commit_count(self):
        return int(self.git("rev-list", "--count", "HEAD").strip())

    def remote_branches(self):
        return git(self.remote, "for-each-ref", "--format=%(refname:short)", "refs/heads").split()


@pytest.fixture
def repo(client, database, tmp_path, fake_github):
    return Repo(client, database, tmp_path, fake_github)


@pytest.fixture
def ready(repo):
    """An initialized workspace."""

    repo.init()
    return repo


def other_user_headers(database):
    return create_repository(database, email="other@example.com", files={})[1]


def paths_leak(text, tmp_path) -> bool:
    forms = {str(tmp_path), str(tmp_path).replace(chr(92), "/"), str(tmp_path.resolve()), tmp_path.as_uri()}
    return any(form in text for form in forms)


# ------------------------------------------------------------------ WORKSPACE


def test_the_workspace_does_not_exist_until_it_is_initialized(repo):
    response = repo.get("workspace")

    assert response.status_code == 200
    assert response.json() == {
        "repository_id": str(repo.id), "exists": False, "branch": None, "default_branch": "main",
        "uncommitted_changes": False, "commit": None,
    }
    assert not repo.base.exists()


def test_initializing_clones_the_repository_into_a_deterministic_directory(repo):
    body = repo.init().json()

    assert body["exists"] is True and body["branch"] == "main" and body["uncommitted_changes"] is False
    assert re.fullmatch(r"[0-9a-f]{40}", body["commit"])
    assert repo.root == repo.base / "repositories" / str(repo.id) and (repo.root / ".git").is_dir()
    assert repo.read(UI) == "render button with css layout" + chr(10) + "label = 'OK'" + chr(10)
    assert repo.get("workspace").json() == body


def test_the_checkout_config_never_contains_the_token_or_credentials(ready):
    config = (ready.root / ".git" / "config").read_text()

    assert FAKE_TOKEN not in config and "x-access-token" not in config and "Authorization" not in config
    assert ready.git("remote", "get-url", "origin").strip() == ready.remote.as_uri()
    assert ready.git("config", "core.symlinks").strip() == "false"


def test_an_existing_workspace_is_reused_and_local_work_is_never_touched(ready):
    ready.write(UI, "locally edited" + chr(10))
    ready.write("scratch/notes.txt", "untracked" + chr(10))
    head = ready.git("rev-parse", "HEAD")

    body = ready.init().json()

    assert ready.read(UI) == "locally edited" + chr(10) and ready.read("scratch/notes.txt") == "untracked" + chr(10)
    assert ready.git("rev-parse", "HEAD") == head and body["uncommitted_changes"] is True


def test_a_failed_clone_leaves_nothing_behind(repo, monkeypatch):
    monkeypatch.setattr(workspace_service, "remote_url", lambda repository: (repo.tmp_path / "missing.git").as_uri())
    monkeypatch.setattr(workspace_service, "resolve_remote", lambda repository: workspace_service.GitRemote((repo.tmp_path / "missing.git").as_uri(), FAKE_TOKEN))

    response = repo.post("workspace")

    assert response.status_code in (404, 502) and "missing.git" not in response.text
    assert not repo.root.exists() and not [p for p in (repo.base / "repositories").iterdir()]


def test_a_missing_github_connection_is_a_controlled_error(repo, monkeypatch):
    def unauthorized(repository):
        raise GitError("GITHUB_AUTH_REQUIRED", "GitHub authorization is required. Reconnect your GitHub account.", 403)

    monkeypatch.setattr(workspace_service, "resolve_remote", unauthorized)

    response = repo.post("workspace")

    assert response.status_code == 403 and response.json()["error"]["code"] == "GITHUB_AUTH_REQUIRED"
    assert not repo.root.exists()


def test_each_repository_gets_its_own_isolated_workspace(client, database, tmp_path, fake_github):
    first = Repo(client, database, tmp_path, fake_github)
    second = Repo(client, database, tmp_path, fake_github, files={"other.py": [(1, "x = 1" + chr(10))]}, email="second@example.com")
    first.init(), second.init()

    first.write(UI, "changed" + chr(10))

    assert first.root != second.root and not (second.root / "app").exists()
    assert second.get("git/status").json()["clean"] is True and first.get("git/status").json()["modified"] == [UI]


def test_a_workspace_that_belongs_to_another_repository_is_refused(ready):
    ready.git("remote", "set-url", "origin", "https://example.com/somebody/else.git")

    for response in (ready.get("git/status"), ready.get("workspace"), ready.commit()):
        assert response.status_code == 409 and response.json()["error"]["code"] == "WORKSPACE_MISMATCH"


def test_a_symlink_from_the_remote_is_materialized_as_a_plain_file_and_cannot_escape(client, database, tmp_path, fake_github):
    secret = tmp_path / "outside-secret.txt"
    secret.write_text("outside-data")
    repo = Repo(client, database, tmp_path, fake_github, files={UI: [(1, "x = 1" + chr(10))]})
    with database() as session:
        from app.db.models import Repository

        fake_github.remotes[repo.id] = make_remote(
            tmp_path / "custom", "linked", {UI: "x = 1" + chr(10)}, symlinks={"link.txt": str(secret)}
        )
    repo.init()

    link = repo.root / "link.txt"
    workspace = Workspace.attach(repo.root, repo.id, WriteScope(frozenset(), frozenset({"link.txt"}), frozenset()))

    assert link.is_file() and not link.is_symlink()
    assert workspace.read_text("link.txt") == str(secret)  # the link's text, not the target's content
    assert secret.read_text() == "outside-data"


def test_a_symlink_inside_the_checkout_is_refused_by_the_workspace(ready, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "victim.py").write_text("keep = 1")
    try:
        (ready.root / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available here")
    workspace = Workspace.attach(ready.root, ready.id, WriteScope(frozenset(), frozenset({"link/victim.py"}), frozenset({"link/victim.py"})))

    for action in (lambda: workspace.read_text("link/victim.py"), lambda: workspace.edit("link/victim.py", "keep", "gone"), lambda: workspace.delete("link/victim.py")):
        with pytest.raises(Exception):
            action()
    assert (outside / "victim.py").read_text() == "keep = 1"


def test_attaching_to_a_missing_directory_is_a_controlled_error(tmp_path):
    from app.workspace import WorkspaceError

    with pytest.raises(WorkspaceError) as caught:
        Workspace.attach(tmp_path / "nope", uuid.uuid4())

    assert caught.value.code == "WORKSPACE_NOT_FOUND" and str(tmp_path) not in caught.value.message


def test_no_endpoint_ever_returns_an_absolute_path(ready, tmp_path):
    ready.write(UI, "changed" + chr(10))
    ready.write("new.py", "n = 1" + chr(10))
    ready.branch()
    responses = [ready.get("workspace"), ready.get("git/status"), ready.get("git/diff"), ready.get("git/log"), ready.commit(), ready.push(), ready.commit(), ready.branch("main")]

    assert not [r.text for r in responses if paths_leak(r.text, tmp_path) or "remotes" in r.text or ".git/" in r.text]


def test_workspace_operations_are_refused_while_another_operation_holds_the_lock(ready, monkeypatch):
    use_llm(monkeypatch, FakeLLM(say("unused")))
    with exclusive_workspace(ready.base, ready.id):
        responses = [
            ready.post("workspace"), ready.post("workspace/sync"), ready.branch(), ready.commit(), ready.push(),
            execute(ready.client, ready.id, ready.headers),
        ]

    assert [r.status_code for r in responses] == [409] * 6
    assert ready.get("git/status").status_code == 200  # reading never waits for the lock


def test_the_lock_admits_one_holder_at_a_time_across_threads(tmp_path):
    repository_id = uuid.uuid4()
    inside, peak, conflicts, lock = [0], [0], [0], threading.Lock()
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        try:
            with exclusive_workspace(tmp_path, repository_id):
                with lock:
                    inside[0] += 1
                    peak[0] = max(peak[0], inside[0])
                time.sleep(0.05)
                with lock:
                    inside[0] -= 1
        except ConflictError:
            with lock:
                conflicts[0] += 1

    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert peak[0] == 1 and conflicts[0] >= 1
    with exclusive_workspace(tmp_path, repository_id):  # released afterwards
        pass


def test_the_lock_works_across_processes_and_is_released_when_the_holder_dies(tmp_path):
    repository_id = uuid.uuid4()
    code = (
        "import sys, time, uuid" + chr(10)
        + "from pathlib import Path" + chr(10)
        + "from app.workspace.lock import exclusive_workspace" + chr(10)
        + "with exclusive_workspace(Path(sys.argv[1]), uuid.UUID(sys.argv[2])):" + chr(10)
        + "    print('locked', flush=True)" + chr(10)
        + "    time.sleep(60)" + chr(10)
    )
    holder = subprocess.Popen([sys.executable, "-c", code, str(tmp_path), str(repository_id)], cwd=BACKEND, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(ConflictError):
            with exclusive_workspace(tmp_path, repository_id):
                pass
    finally:
        holder.kill()
        holder.wait()
    with exclusive_workspace(tmp_path, repository_id):
        pass


# ------------------------------------------------------------------ GIT STATUS


def status_of(repo):
    response = repo.get("git/status")
    assert response.status_code == 200, response.text
    return response.json()


def test_status_of_a_clean_repository(ready):
    assert status_of(ready) == {
        "repository_id": str(ready.id), "branch": "main", "clean": True, "modified": [], "added": [], "deleted": [],
        "untracked": [], "conflicted": [], "withheld": 0, "truncated": False,
    }


def test_status_reports_modified_added_deleted_and_untracked_files(ready):
    ready.write(UI, "changed" + chr(10))
    (ready.root / "app" / "old.py").unlink()
    ready.write("staged.py", "s = 1" + chr(10))
    ready.git("add", "staged.py")
    ready.write("brand/new.py", "n = 1" + chr(10))

    body = status_of(ready)

    assert (body["modified"], body["deleted"], body["added"], body["untracked"]) == ([UI], ["app/old.py"], ["staged.py"], ["brand/new.py"])
    assert body["clean"] is False and body["branch"] == "main"


def test_status_counts_but_never_names_protected_files(ready):
    ready.write(".env", "TOKEN=abc" + chr(10))
    ready.write("keys/server.pem", "key" + chr(10))

    body = status_of(ready)

    assert body["withheld"] == 2 and body["untracked"] == [] and ".env" not in json.dumps(body) and body["clean"] is False


def test_status_needs_an_initialized_workspace_and_ownership(repo, database):
    assert repo.get("git/status").status_code == 409
    repo.init()

    assert repo.get("git/status", headers={}).status_code == 401
    assert repo.get("git/status", headers=other_user_headers(database)).status_code == 404
    assert repo.get("git/status").status_code == 200


# ------------------------------------------------------------------ GIT DIFF


def diff_of(repo):
    response = repo.get("git/diff")
    assert response.status_code == 200, response.text
    return response.json()


def test_diff_of_a_modified_file(ready):
    ready.write(UI, "render button with css layout" + chr(10) + "label = 'Save'" + chr(10) + "extra = 1" + chr(10))

    body = diff_of(ready)

    [entry] = body["files"]
    assert (entry["path"], entry["status"], entry["additions"], entry["deletions"]) == (UI, "modified", 2, 1)
    assert "-label = 'OK'" in entry["diff"] and "+label = 'Save'" in entry["diff"] and entry["diff_truncated"] is False


def test_diff_of_added_deleted_and_untracked_files(ready):
    (ready.root / "app" / "old.py").unlink()
    ready.write("staged.py", "s = 1" + chr(10))
    ready.git("add", "staged.py")
    ready.write("fresh.py", "f = 1" + chr(10) + "g = 2" + chr(10))

    files = {entry["path"]: entry for entry in diff_of(ready)["files"]}

    assert files["app/old.py"]["status"] == "deleted" and files["app/old.py"]["deletions"] == 1 and "-obsolete = True" in files["app/old.py"]["diff"]
    assert files["staged.py"]["status"] == "added" and "+s = 1" in files["staged.py"]["diff"]
    assert files["fresh.py"]["status"] == "untracked" and files["fresh.py"]["additions"] == 2 and "+g = 2" in files["fresh.py"]["diff"]


def test_a_large_diff_is_cut_at_the_size_limit(ready):
    ready.write("big.txt", "".join(f"line number {i}" + chr(10) for i in range(6000)))
    ready.write(UI, "".join(f"changed {i}" + chr(10) for i in range(6000)))

    files = {entry["path"]: entry for entry in diff_of(ready)["files"]}

    assert all(entry["diff_truncated"] and len(entry["diff"]) <= 20_000 for entry in files.values())


def test_too_many_changed_files_are_limited(ready):
    for number in range(60):
        ready.write(f"many/file{number:02d}.txt", "x" + chr(10))

    body = diff_of(ready)

    assert len(body["files"]) == 50 and body["truncated"] is True


def test_diffs_redact_secrets_and_leave_out_protected_files(ready):
    ready.write("app/config.py", chr(10).join(SECRET_LINES) + chr(10))
    ready.write(UI, "password = 'hunter22secret'" + chr(10))
    ready.write(".env", "TOPSECRET=hunter22secret" + chr(10))

    response = ready.get("git/diff")
    body = response.json()

    assert not [value for value in SECRET_VALUES if value in response.text] and "TOPSECRET" not in response.text
    assert body["withheld"] == 1 and {entry["path"] for entry in body["files"]} == {"app/config.py", UI}


def test_binary_and_unreadable_untracked_files_have_no_diff_text(ready):
    (ready.root / "blob.bin").write_bytes(bytes([0, 159, 146, 150]))

    [entry] = diff_of(ready)["files"]

    assert entry["binary"] is True and entry["diff"] == ""


def test_diff_needs_ownership(ready, database):
    assert ready.get("git/diff", headers={}).status_code == 401
    assert ready.get("git/diff", headers=other_user_headers(database)).status_code == 404


def test_log_lists_commits_with_redacted_subjects(ready):
    body = ready.get("git/log", limit=5).json()

    assert [c["subject"] for c in body["commits"]] == ["initial commit"] and re.fullmatch(r"[0-9a-f]{40}", body["commits"][0]["commit"])
    assert ready.get("git/log", limit=0).status_code == 422 and ready.get("git/log", limit=51).status_code == 422


# ------------------------------------------------------------------ BRANCH

INVALID_BRANCHES = [
    pytest.param("main", id="main"),
    pytest.param("master", id="master"),
    pytest.param("origin/main", id="remote-main"),
    pytest.param("codefrog/", id="empty-suffix"),
    pytest.param("codefrog", id="no-slash"),
    pytest.param("feature/x", id="wrong-prefix"),
    pytest.param("codefrog/has space", id="space"),
    pytest.param("codefrog/../main", id="dotdot"),
    pytest.param("codefrog//x", id="double-slash"),
    pytest.param("codefrog/x.lock", id="lock-suffix"),
    pytest.param("codefrog/x.", id="trailing-dot"),
    pytest.param("codefrog/x~1", id="tilde"),
    pytest.param("codefrog/x^", id="caret"),
    pytest.param("codefrog/x:y", id="colon"),
    pytest.param("codefrog/x" + chr(10) + "y", id="newline"),
    pytest.param("codefrog/x" + chr(0), id="nul"),
    pytest.param("--upload-pack=touch pwned", id="option"),
    pytest.param("-c", id="dash-c"),
    pytest.param("codefrog/x;touch pwned", id="semicolon"),
    pytest.param("codefrog/$(touch pwned)", id="substitution"),
    pytest.param("codefrog/x|touch pwned", id="pipe"),
    pytest.param("codefrog/`touch pwned`", id="backtick"),
    pytest.param("codefrog/x&&touch pwned", id="and"),
    pytest.param("codefrog/naïve", id="unicode"),
    pytest.param("codefrog/" + "a" * 100, id="too-long"),
]


def test_a_valid_branch_is_created_and_checked_out(ready):
    ready.write(UI, "changed" + chr(10))

    response = ready.branch("codefrog/fix-auth")

    assert response.status_code == 200 and response.json() == {"repository_id": str(ready.id), "branch": "codefrog/fix-auth", "created": True}
    assert status_of(ready)["branch"] == "codefrog/fix-auth" and ready.read(UI) == "changed" + chr(10)  # local work carries over
    assert ready.get("workspace").json()["branch"] == "codefrog/fix-auth"


@pytest.mark.parametrize("name", INVALID_BRANCHES)
def test_invalid_and_malicious_branch_names_are_rejected(ready, name):
    response = ready.branch(name)

    assert response.status_code in (400, 422)
    assert status_of(ready)["branch"] == "main" and ready.git("branch", "--list").split() == ["*", "main"]
    assert not list(ready.tmp_path.rglob("pwned"))


def test_an_existing_branch_is_never_overwritten(ready):
    ready.git("branch", "codefrog/taken")
    tip = ready.git("rev-parse", "codefrog/taken")
    ready.write(UI, "changed" + chr(10))

    response = ready.branch("codefrog/taken")

    assert response.status_code == 409 and response.json()["error"]["code"] == "BRANCH_EXISTS"
    assert ready.git("rev-parse", "codefrog/taken") == tip and status_of(ready)["branch"] == "main"


@pytest.mark.parametrize("approval", [{"approved": False}, {}, {"approved": None}, {"approved": "true"}, {"approved": 1}], ids=["false", "missing", "null", "string", "number"])
def test_creating_a_branch_requires_explicit_approval(ready, approval):
    response = ready.post("agent/branch", {"name": "codefrog/ok", **approval})

    assert response.status_code in (403, 422)
    assert status_of(ready)["branch"] == "main" and "codefrog/ok" not in ready.git("branch", "--list")


def test_branch_names_are_validated_by_the_service_itself():
    assert validate_branch_name("codefrog/task-2024.01_a/b") == "codefrog/task-2024.01_a/b"
    for name in ("main", "codefrog/..", "codefrog/a b", "-x"):
        with pytest.raises(GitError):
            validate_branch_name(name)


def test_branch_endpoints_need_a_workspace_and_ownership(repo, database):
    assert repo.branch().status_code == 409
    repo.init()

    assert repo.post("agent/branch", {"name": "codefrog/a", "approved": True}, headers={}).status_code == 401
    assert repo.post("agent/branch", {"name": "codefrog/a", "approved": True}, headers=other_user_headers(database)).status_code == 404


# ------------------------------------------------------------------ COMMIT


def changed_on_branch(repo, name="codefrog/task-1"):
    repo.write(UI, "render button with css layout" + chr(10) + "label = 'Save'" + chr(10))
    assert repo.branch(name).status_code == 200


def test_a_commit_is_created_on_the_codefrog_branch(ready):
    changed_on_branch(ready)
    before = ready.commit_count()

    response = ready.commit("feat: change the label" + chr(10) + chr(10) + "Body line.")

    body = response.json()
    assert response.status_code == 200 and body["branch"] == "codefrog/task-1" and body["files_changed"] == 1
    assert body["commit"] == ready.git("rev-parse", "HEAD").strip() and ready.commit_count() == before + 1
    assert ready.git("log", "-1", "--format=%B").strip() == "feat: change the label" + chr(10) + chr(10) + "Body line."
    assert ready.git("log", "-1", "--format=%an <%ae>").strip() == "Owner <me@example.com>"
    assert status_of(ready)["clean"] is True and ready.get("git/log").json()["commits"][0]["subject"] == "feat: change the label"


def test_a_commit_includes_new_and_deleted_files(ready):
    changed_on_branch(ready)
    ready.write("new/module.py", "n = 1" + chr(10))
    (ready.root / "app" / "old.py").unlink()

    body = ready.commit().json()

    assert body["files_changed"] == 3
    assert sorted(ready.git("show", "--name-status", "--format=", "HEAD").split()) == sorted(["M", UI, "A", "new/module.py", "D", "app/old.py"])


@pytest.mark.parametrize("approval", [{"approved": False}, {}, {"approved": None}, {"approved": "true"}], ids=["false", "missing", "null", "string"])
def test_a_commit_requires_explicit_approval(ready, approval):
    changed_on_branch(ready)
    before = ready.commit_count()

    response = ready.post("agent/commit", {"message": "feat: x", **approval})

    assert response.status_code in (403, 422) and ready.commit_count() == before and status_of(ready)["clean"] is False


def test_commits_are_refused_on_the_default_branch(ready):
    ready.write(UI, "changed" + chr(10))
    before = ready.commit_count()

    response = ready.commit()

    assert response.status_code == 403 and response.json()["error"]["code"] == "PROTECTED_BRANCH"
    assert ready.commit_count() == before and status_of(ready)["modified"] == [UI]


def test_commits_are_refused_on_a_branch_that_is_not_a_codefrog_branch(ready):
    ready.git("switch", "-c", "master")
    ready.write(UI, "changed" + chr(10))

    assert ready.commit().json()["error"]["code"] == "PROTECTED_BRANCH"


@pytest.mark.parametrize(
    "message",
    [
        pytest.param("", id="empty"),
        pytest.param("   ", id="blank"),
        pytest.param("x" * 201, id="long-subject"),
        pytest.param("subject" + chr(10) + "y" * 2_000, id="long-body"),
        pytest.param("bad" + chr(0) + "message", id="nul"),
        pytest.param("bell" + chr(7), id="control"),
        pytest.param("fix: use " + SECRET_LINES[1].split('"')[1], id="secret"),
    ],
)
def test_invalid_commit_messages_are_rejected(ready, message):
    changed_on_branch(ready)
    before = ready.commit_count()

    response = ready.commit(message)

    assert response.status_code in (400, 422) and ready.commit_count() == before


def test_option_like_and_shell_like_messages_are_stored_verbatim(ready):
    changed_on_branch(ready)
    message = "--amend $(touch pwned); rm -rf / `id` && echo hi | cat > x"

    assert ready.commit(message).status_code == 200

    assert ready.git("log", "-1", "--format=%s").strip() == message and ready.commit_count() == 2
    assert not list(ready.tmp_path.rglob("pwned"))


def test_committing_with_no_changes_is_a_conflict(ready):
    assert ready.branch().status_code == 200
    before = ready.commit_count()

    response = ready.commit()

    assert response.status_code == 409 and response.json()["error"]["code"] == "NO_CHANGES" and ready.commit_count() == before


def test_a_protected_file_blocks_the_whole_commit(ready):
    changed_on_branch(ready)
    ready.write(".env", "TOKEN=abc" + chr(10))
    before = ready.commit_count()

    response = ready.commit()

    assert response.status_code == 403 and response.json()["error"]["code"] == "PROTECTED_PATH" and ".env" not in response.text
    assert ready.commit_count() == before and ready.git("diff", "--cached", "--name-only").strip() == ""


def test_commit_message_validation_is_available_to_the_service():
    assert validate_commit_message("  feat: ok  ") == "feat: ok"
    with pytest.raises(GitError):
        validate_commit_message("")


def test_commit_endpoints_need_ownership_authentication_and_a_workspace(repo, database):
    assert repo.commit().status_code == 409
    repo.init()
    changed_on_branch(repo)
    before = repo.commit_count()

    assert repo.post("agent/commit", {"message": "feat: x", "approved": True}, headers={}).status_code == 401
    assert repo.post("agent/commit", {"message": "feat: x", "approved": True}, headers=other_user_headers(database)).status_code == 404
    assert repo.post("agent/commit", {"message": "feat: x", "approved": True, "extra": 1}).status_code == 422
    assert repo.commit_count() == before


# ------------------------------------------------------------------ PUSH


def committed_branch(repo, name="codefrog/task-1"):
    changed_on_branch(repo, name)
    assert repo.commit().status_code == 200


def test_a_push_publishes_only_the_current_codefrog_branch(ready, caplog):
    committed_branch(ready)
    main_tip = git(ready.remote, "rev-parse", "refs/heads/main").strip()

    with caplog.at_level("DEBUG"):
        response = ready.push()

    head = ready.git("rev-parse", "HEAD").strip()
    assert response.status_code == 200 and response.json() == {"repository_id": str(ready.id), "branch": "codefrog/task-1", "commit": head, "pushed": True}
    assert git(ready.remote, "rev-parse", "refs/heads/codefrog/task-1").strip() == head
    assert git(ready.remote, "rev-parse", "refs/heads/main").strip() == main_tip and sorted(ready.remote_branches()) == ["codefrog/task-1", "main"]
    assert FAKE_TOKEN not in response.text + caplog.text and FAKE_TOKEN not in (ready.root / ".git" / "config").read_text()
    assert ready.git("config", "--get", "branch.codefrog/task-1.remote").strip() == "origin"


@pytest.mark.parametrize("approval", [{"approved": False}, {}, {"approved": None}, {"approved": "yes"}], ids=["false", "missing", "null", "string"])
def test_a_push_requires_explicit_approval(ready, approval):
    committed_branch(ready)

    response = ready.post("agent/push", approval)

    assert response.status_code in (403, 422) and ready.remote_branches() == ["main"]


def test_the_default_branch_is_never_pushed(ready):
    ready.write(UI, "changed" + chr(10))
    ready.git("add", "--all")
    ready.git("commit", "-q", "-m", "local commit on main")
    main_tip = git(ready.remote, "rev-parse", "refs/heads/main").strip()

    response = ready.push()

    assert response.status_code == 403 and response.json()["error"]["code"] == "PROTECTED_BRANCH"
    assert git(ready.remote, "rev-parse", "refs/heads/main").strip() == main_tip


def test_a_rejected_push_is_reported_without_git_output(ready):
    committed_branch(ready)
    (ready.tmp_path / "scratch-a").mkdir()
    add_remote_commit(ready.remote, ready.tmp_path / "scratch-a", {"other.txt": "x"}, branch="codefrog/task-1")

    response = ready.push()

    assert response.status_code == 409 and response.json()["error"]["code"] == "PUSH_REJECTED"
    assert "non-fast-forward" not in response.text and "refs/heads" not in response.text and not paths_leak(response.text, ready.tmp_path)


def test_a_push_never_forces(ready):
    committed_branch(ready)
    (ready.tmp_path / "scratch-b").mkdir()
    add_remote_commit(ready.remote, ready.tmp_path / "scratch-b", {"other.txt": "x"}, branch="codefrog/task-1")
    remote_tip = git(ready.remote, "rev-parse", "refs/heads/codefrog/task-1").strip()

    ready.push()

    assert git(ready.remote, "rev-parse", "refs/heads/codefrog/task-1").strip() == remote_tip


def test_an_authentication_failure_is_safe_and_leaks_nothing(ready, monkeypatch):
    committed_branch(ready)
    real = git_repository.run_git

    def failing(directory, arguments, **kwargs):
        if arguments[0] == "push":
            return GitResult(128, "", f"fatal: Authentication failed for 'https://x-access-token:{FAKE_TOKEN}@github.com/o/r.git/' at {directory}")
        return real(directory, arguments, **kwargs)

    monkeypatch.setattr(git_repository, "run_git", failing)

    response = ready.push()

    assert response.status_code == 403 and response.json()["error"]["code"] == "GITHUB_AUTH_FAILED"
    assert FAKE_TOKEN not in response.text and "github.com" not in response.text and not paths_leak(response.text, ready.tmp_path)


def test_an_unexpected_push_failure_is_a_generic_bad_gateway(ready, monkeypatch):
    committed_branch(ready)
    real = git_repository.run_git
    monkeypatch.setattr(git_repository, "run_git", lambda directory, arguments, **kw: GitResult(1, "", "fatal: secret internal detail " + FAKE_TOKEN) if arguments[0] == "push" else real(directory, arguments, **kw))

    response = ready.push()

    assert response.status_code == 502 and "secret internal detail" not in response.text and FAKE_TOKEN not in response.text


def test_a_push_without_a_github_connection_is_refused(ready, monkeypatch):
    committed_branch(ready)

    def unauthorized(repository):
        raise GitError("GITHUB_AUTH_REQUIRED", "GitHub authorization is required. Reconnect your GitHub account.", 403)

    monkeypatch.setattr(workspace_service, "resolve_remote", unauthorized)

    assert ready.push().status_code == 403 and ready.remote_branches() == ["main"]


def test_push_endpoints_need_ownership_authentication_and_a_workspace(repo, database):
    assert repo.push().status_code == 409
    repo.init()
    committed_branch(repo)

    assert repo.post("agent/push", {"approved": True}, headers={}).status_code == 401
    assert repo.post("agent/push", {"approved": True}, headers=other_user_headers(database)).status_code == 404
    assert repo.remote_branches() == ["main"]


# ------------------------------------------------------------------ SYNC


def test_sync_fast_forwards_a_clean_default_branch(ready):
    (ready.tmp_path / "scratch-c").mkdir()
    add_remote_commit(ready.remote, ready.tmp_path / "scratch-c", {"remote_file.py": "r = 1" + chr(10)})

    body = ready.post("workspace/sync").json()

    assert body["action"] == "fast_forwarded" and ready.read("remote_file.py") == "r = 1" + chr(10)
    assert ready.post("workspace/sync").json()["action"] == "up_to_date"


def test_sync_never_discards_local_changes(ready):
    (ready.tmp_path / "scratch-d").mkdir()
    add_remote_commit(ready.remote, ready.tmp_path / "scratch-d", {"remote_file.py": "r = 1" + chr(10)})
    ready.write(UI, "local work" + chr(10))

    body = ready.post("workspace/sync").json()

    assert body["action"] == "skipped" and "uncommitted" in body["reason"]
    assert ready.read(UI) == "local work" + chr(10) and not (ready.root / "remote_file.py").exists()


def test_sync_skips_a_workspace_that_is_on_a_task_branch(ready):
    ready.branch()

    body = ready.post("workspace/sync").json()

    assert body["action"] == "skipped" and status_of(ready)["branch"] == "codefrog/task-1"


# ------------------------------------------------------------------ EXECUTION + READ TOOLS


def test_edits_from_one_execution_are_visible_to_the_next_and_in_the_git_diff(ready, monkeypatch):
    use_llm(monkeypatch, FakeLLM(call(*edit()), say("done")))
    assert execute(ready.client, ready.id, ready.headers).status_code == 200
    llm = use_llm(monkeypatch, FakeLLM(call("read_file", {"file_path": UI}), say("seen")))

    body = execute(ready.client, ready.id, ready.headers).json()

    result = tool_result(llm.calls[1])["output"]
    assert "label = 'Save'" in result["content"] and result["source"] == "workspace" and body["changes"] == []
    assert body["uncommitted_changes"] is True
    [entry] = diff_of(ready)["files"]
    assert entry["path"] == UI and "+label = 'Save'" in entry["diff"]


def test_execution_clones_the_workspace_when_it_is_missing(repo, monkeypatch):
    use_llm(monkeypatch, FakeLLM(call(*edit()), say("done")))

    body = execute(repo.client, repo.id, repo.headers).json()

    assert (repo.root / ".git").is_dir() and body["branch"] == "main" and [c["path"] for c in body["changes"]] == [UI]


def test_the_read_only_agent_reads_the_checkout_when_one_exists(ready, monkeypatch):
    ready.write(UI, "edited locally" + chr(10))
    llm = use_llm(monkeypatch, FakeLLM(call("read_file", {"file_path": UI}), say("ok")))

    ready.client.post(f"/api/v1/repositories/{ready.id}/agent", json={"message": "q"}, headers=ready.headers)

    output = tool_result(llm.calls[1])["output"]
    assert output["content"] == "edited locally" and output["source"] == "workspace"


def test_the_read_only_agent_falls_back_to_the_index_and_says_so(repo, monkeypatch):
    llm = use_llm(monkeypatch, FakeLLM(call("read_file", {"file_path": UI}), say("ok")))

    repo.client.post(f"/api/v1/repositories/{repo.id}/agent", json={"message": "q"}, headers=repo.headers)

    assert tool_result(llm.calls[1])["output"]["source"] == "index" and not repo.base.exists()


def test_planning_also_reads_the_checkout(ready, monkeypatch):
    ready.write(UI, "planning sees this" + chr(10))
    llm = use_llm(monkeypatch, FakeLLM(call("read_file", {"file_path": UI}), say(json.dumps(plan()))))

    ready.client.post(f"/api/v1/repositories/{ready.id}/agent/plan", json={"message": REQUEST}, headers=ready.headers)

    assert tool_result(llm.calls[1])["output"]["content"] == "planning sees this"


def test_read_file_keeps_its_security_rules_on_the_checkout(ready, monkeypatch):
    ready.write(".env", "TOPSECRET=hunter22secret" + chr(10))
    ready.write("app/config.py", chr(10).join(SECRET_LINES) + chr(10))
    llm = use_llm(
        monkeypatch,
        FakeLLM(calls(("read_file", {"file_path": ".env"}), ("read_file", {"file_path": "app/config.py"}), ("read_file", {"file_path": "../outside.py"}), ("read_file", {"file_path": "app/missing.py"})), say("ok")),
    )

    response = ready.client.post(f"/api/v1/repositories/{ready.id}/agent", json={"message": "q"}, headers=ready.headers)

    results = [json.loads(m["content"]) for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert results[0]["error"]["code"] == "FORBIDDEN" and results[2]["error"]["code"] == "INVALID_INPUT" and results[3]["error"]["code"] == "RESOURCE_NOT_FOUND"
    assert not [value for value in SECRET_VALUES if value in json.dumps(results)] and "TOPSECRET" not in response.text
    assert results[1]["output"]["redactions"] >= 1


def test_search_code_states_that_it_reflects_the_last_scan(ready, monkeypatch):
    llm = use_llm(monkeypatch, FakeLLM(call("search_code", {"query": "button label"}), say("ok")))

    ready.client.post(f"/api/v1/repositories/{ready.id}/agent", json={"message": "q"}, headers=ready.headers)

    assert tool_result(llm.calls[1])["output"]["source"] == "last_scan"
    assert "last repository scan" in [t for t in llm.calls[0]["tools"] if t["name"] == "search_code"][0]["description"]


def test_the_agent_cannot_commit_push_or_branch(ready, monkeypatch):
    remote_tip = git(ready.remote, "rev-parse", "refs/heads/main").strip()
    head = ready.git("rev-parse", "HEAD")
    attempts = ("git_commit", "git_push", "create_branch", "commit", "push", "git", "run_command", "shell")
    llm = use_llm(
        monkeypatch,
        FakeLLM(calls(*((name, {"message": "x", "approved": True, "name": "codefrog/x"}) for name in attempts[:4])), calls(*((name, {"approved": True}) for name in attempts[4:])), say("done")),
    )

    body = execute(ready.client, ready.id, ready.headers).json()

    assert body["status"] == "completed" and [c["path"] for c in body["changes"]] == []
    assert ready.git("rev-parse", "HEAD") == head and ready.git("branch", "--list").split() == ["*", "main"]
    assert git(ready.remote, "rev-parse", "refs/heads/main").strip() == remote_tip and ready.remote_branches() == ["main"]
    assert {t["name"] for t in llm.calls[0]["tools"]} == {"search_code", "read_file", "analyze_project", "edit_file", "create_file", "delete_file"}
    assert not [name for name in (t["name"] for t in tool_definitions(include_write=True)) if re.search("git|commit|push|branch|shell|run", name)]


def test_executing_after_the_user_commits_starts_from_the_new_commit(ready, monkeypatch):
    use_llm(monkeypatch, FakeLLM(call(*edit()), say("done")))
    execute(ready.client, ready.id, ready.headers)
    assert ready.branch().status_code == 200 and ready.commit().status_code == 200
    llm = use_llm(monkeypatch, FakeLLM(call("read_file", {"file_path": UI}), say("ok")))

    body = execute(ready.client, ready.id, ready.headers).json()

    assert "label = 'Save'" in tool_result(llm.calls[1])["output"]["content"] and body["uncommitted_changes"] is False and body["branch"] == "codefrog/task-1"


# ------------------------------------------------------------------ SECURITY


def python_files():
    return [path for path in APP.rglob("*.py")]


def test_git_is_started_only_from_the_runner_and_never_through_a_shell():
    offenders = []
    for path in python_files():
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(APP).as_posix()
        for pattern in (r"shell\s*=\s*True", r"os\.system", r"os\.popen", r"\bPopen\b", r"\bpexpect\b", r"asyncio\.create_subprocess"):
            if re.search(pattern, text):
                offenders.append(f"{relative}: {pattern}")
        if re.search(r"\bsubprocess\b", text) and relative != "git/runner.py":
            offenders.append(f"{relative}: subprocess outside the runner")
    assert offenders == []


def test_no_pull_request_code_exists():
    offenders = [p.name for p in python_files() if re.search(r"create_pull|/pulls\b|pulls\.create|gh pr", p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_every_git_command_is_an_argument_list_in_the_workspace_with_the_token_only_in_the_environment(ready, monkeypatch):
    calls_seen = []
    real = subprocess.run

    def recording(command, **kwargs):
        calls_seen.append((command, kwargs))
        return real(command, **kwargs)

    monkeypatch.setattr(git_runner.subprocess, "run", recording)
    committed_branch(ready)
    ready.push()
    ready.post("workspace/sync")
    ready.get("git/diff"), ready.get("git/status"), ready.get("git/log")

    assert calls_seen
    for command, kwargs in calls_seen:
        assert isinstance(command, list) and all(isinstance(part, str) for part in command)
        assert kwargs["shell"] is False and kwargs["timeout"] > 0 and kwargs["stdin"] == subprocess.DEVNULL
        assert Path(kwargs["cwd"]).resolve().is_relative_to(ready.base.resolve())
        assert not any(FAKE_TOKEN in part for part in command)
        assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0" and kwargs["env"]["GIT_CONFIG_GLOBAL"] and "credential.helper=" in command
        assert "SECRET" not in "".join(kwargs["env"]) and "LLM_API_KEY" not in kwargs["env"] and "DATABASE_URL" not in kwargs["env"]
    assert any(kwargs["env"].get("GIT_CONFIG_VALUE_0", "").startswith("Authorization: Basic ") for _, kwargs in calls_seen)  # network calls carry it in the environment


def test_a_git_timeout_is_a_controlled_error(ready, monkeypatch):
    def slow(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(git_runner.subprocess, "run", slow)

    response = ready.get("git/status")

    assert response.status_code == 504 and response.json()["error"]["code"] == "GIT_TIMEOUT" and not paths_leak(response.text, ready.tmp_path)


def test_a_missing_git_executable_is_a_controlled_error(ready, monkeypatch):
    monkeypatch.setattr(git_runner.shutil, "which", lambda name: None)

    response = ready.get("git/status")

    assert response.status_code == 503 and response.json()["error"]["code"] == "GIT_UNAVAILABLE"


def test_hooks_in_the_checkout_never_run(ready):
    marker = ready.tmp_path / "hook-ran"
    hook = ready.root / ".git" / "hooks" / "pre-commit"
    hook.write_bytes(("#!/bin/sh" + chr(10) + f"touch '{marker.as_posix()}'" + chr(10)).encode())
    hook.chmod(0o755)
    committed_branch(ready)

    assert not marker.exists()


def test_the_token_never_appears_in_logs_or_any_response(ready, caplog):
    with caplog.at_level("DEBUG"):
        changed_on_branch(ready)
        texts = [ready.commit().text, ready.push().text, ready.post("workspace/sync").text, ready.get("git/diff").text]

    assert FAKE_TOKEN not in caplog.text + "".join(texts) and "x-access-token" not in caplog.text + "".join(texts)


def test_the_stored_token_is_only_read_through_the_existing_github_integration():
    text = (APP / "workspace" / "service.py").read_text(encoding="utf-8")

    assert "decrypt_access_token" in text and "access_token_encrypted" in text
    assert not re.search(r"os\.environ|getenv|GITHUB_CLIENT_SECRET", text)


def test_every_git_write_is_routed_through_the_approval_check():
    text = (APP / "api" / "routes" / "repository_git.py").read_text(encoding="utf-8")
    writes = re.findall(r'@router\.post\("([^"]+)"', text)

    assert {"/{repository_id}/agent/branch", "/{repository_id}/agent/commit", "/{repository_id}/agent/push"} <= set(writes)
    for name in ("create_branch", "commit_changes", "push_branch"):
        body = text[text.index(f"def {name}("):]
        body = body[: body.index(chr(10) + chr(10) + chr(10))] if chr(10) + chr(10) + chr(10) in body else body
        assert body.index("get_owned_repository") < body.index("_require_approval") < body.index("exclusive_workspace")


def test_the_repository_directory_name_is_the_uuid_only(ready):
    assert [p.name for p in (ready.base / "repositories").iterdir()] == [str(ready.id)]
    assert Path(ready.root).is_relative_to(ready.base)
