"""Tests for repository scanning, chunking, filtering, and the scan endpoint."""

import base64
import hashlib
import logging
import uuid

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes import repositories as repository_routes
from app.auth.security import create_access_token
from app.core.config import get_settings
from app.db.base import Base
from app.db.database import get_db
from app.db.models import GitHubAccount, Repository, RepositoryChunk, RepositoryFile, User
from app.integrations.github.contents import (
    GitHubAuthError,
    GitHubContentClient,
    GitHubContentError,
    GitHubNotFoundError,
    GitHubTree,
    GitHubTreeEntry,
)
from app.scanner.chunking import chunk_source
from app.scanner.filters import detect_language
from main import app

GITHUB_TOKEN = "gho_plaintext_token_that_must_never_leak"


def blob_sha(content: bytes) -> str:
    return hashlib.sha1(content).hexdigest()


class FakeGitHubClient:
    """In-memory GitHub double; records blob downloads and never touches the network."""

    def __init__(self, files=None, *, tree_error=None, unavailable=(), truncated=False):
        self.files = {p: c.encode() if isinstance(c, str) else c for p, c in (files or {}).items()}
        self.tree_error = tree_error
        self.unavailable = set(unavailable)
        self.truncated = truncated
        self.downloads: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def get_tree(self, owner, name, ref):
        if self.tree_error:
            raise self.tree_error
        entries = [
            GitHubTreeEntry(path=path, sha=blob_sha(content), size=len(content))
            for path, content in self.files.items()
        ]
        return GitHubTree(entries=entries, truncated=self.truncated)

    def get_file_content(self, owner, name, sha):
        for path, content in self.files.items():
            if blob_sha(content) == sha:
                if path in self.unavailable:
                    raise GitHubNotFoundError("gone")
                self.downloads.append(path)
                return content
        raise GitHubNotFoundError("gone")


@pytest.fixture
def database():
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def override_get_db():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    yield factory
    app.dependency_overrides.clear()
    engine.dispose()


@pytest.fixture
def client(database):
    with TestClient(app) as test_client:
        yield test_client


def create_repository(factory, *, email="owner@example.com", encrypted_token=None):
    """Insert a user, GitHub account, and repository; return (repository_id, auth headers)."""

    with factory() as session:
        user = User(email=email, name="Owner", status="active")
        session.add(user)
        session.flush()
        account = GitHubAccount(
            user_id=user.id,
            github_user_id=abs(hash(email)) % 10**9,
            login=email.split("@")[0],
            access_token_encrypted=encrypted_token,
        )
        session.add(account)
        session.flush()
        repository = Repository(
            github_account_id=account.id,
            github_repository_id=abs(hash(email + "repo")) % 10**9,
            owner=account.login,
            name="project",
            default_branch="main",
        )
        session.add(repository)
        session.commit()
        headers = {"Authorization": f"Bearer {create_access_token(user.id)}"}
        return repository.id, headers


def use_github(monkeypatch, fake):
    monkeypatch.setattr(repository_routes, "GitHubContentClient", lambda _token: fake)
    return fake


def scan(client, repository_id, headers):
    return client.post(f"/api/v1/repositories/{repository_id}/scan", headers=headers)


def stored_files(factory, repository_id):
    with factory() as session:
        return {
            f.path: f
            for f in session.scalars(
                select(RepositoryFile).where(RepositoryFile.repository_id == repository_id)
            )
        }


def count(factory, model):
    with factory() as session:
        return session.scalar(select(func.count()).select_from(model))


def numbered_lines(total: int) -> str:
    return "".join(f"line {number}\n" for number in range(1, total + 1))


def test_successful_scan_returns_summary(client, database, monkeypatch):
    repository_id, headers = create_repository(database)
    use_github(
        monkeypatch,
        FakeGitHubClient(
            {"app/main.py": numbered_lines(120), "README.md": "docs", "logo.png": b"\x89PNG\x00"}
        ),
    )

    response = scan(client, repository_id, headers)

    assert response.status_code == 200
    assert response.json() == {
        "repository_id": str(repository_id),
        "status": "completed",
        "files_discovered": 3,
        "files_indexed": 1,
        "files_skipped": 2,
        "files_removed": 0,
        "chunks_created": 2,
    }


def test_file_metadata_is_stored(client, database, monkeypatch):
    repository_id, headers = create_repository(database)
    source = "print('hi')\n"
    use_github(monkeypatch, FakeGitHubClient({"src/app.py": source}))

    scan(client, repository_id, headers)

    stored = stored_files(database, repository_id)["src/app.py"]
    assert stored.language == "python"
    assert stored.sha == blob_sha(source.encode())
    assert stored.size_bytes == len(source.encode())


def test_chunks_have_correct_line_ranges_and_exact_content(client, database, monkeypatch):
    repository_id, headers = create_repository(database)
    source = numbered_lines(120)
    use_github(monkeypatch, FakeGitHubClient({"example.py": source}))

    scan(client, repository_id, headers)

    with database() as session:
        chunks = session.scalars(
            select(RepositoryChunk).order_by(RepositoryChunk.chunk_index)
        ).all()
    assert [(c.chunk_index, c.start_line, c.end_line) for c in chunks] == [(0, 1, 100), (1, 101, 120)]
    assert "".join(c.content for c in chunks) == source


def test_unsupported_files_and_ignored_directories_are_not_indexed(client, database, monkeypatch):
    repository_id, headers = create_repository(database)
    fake = use_github(
        monkeypatch,
        FakeGitHubClient(
            {
                "keep.py": "x = 1\n",
                "notes.txt": "text",
                "image.png": b"\x89PNG",
                "node_modules/lib/index.js": "module.exports = 1\n",
                ".git/config.json": "{}",
                "web/dist/bundle.js": "var a\n",
                "src/__pycache__/m.py": "x\n",
                "package-lock.json": "{}",
                "assets/app.min.js": "var a\n",
            }
        ),
    )

    response = scan(client, repository_id, headers)

    assert response.json()["files_indexed"] == 1
    assert set(stored_files(database, repository_id)) == {"keep.py"}
    assert fake.downloads == ["keep.py"]


def test_binary_and_undecodable_content_is_skipped_without_failing_scan(client, database, monkeypatch):
    repository_id, headers = create_repository(database)
    use_github(
        monkeypatch,
        FakeGitHubClient(
            {
                "good.py": "ok = True\n",
                "binary.py": b"abc\x00def",
                "latin1.py": "caf\xe9\n".encode("latin-1"),
                "gone.py": "removed = 1\n",
            },
            unavailable={"gone.py"},
        ),
    )

    response = scan(client, repository_id, headers)

    assert response.status_code == 200
    assert response.json()["files_indexed"] == 1
    assert set(stored_files(database, repository_id)) == {"good.py"}


def test_empty_repository_and_empty_file(client, database, monkeypatch):
    repository_id, headers = create_repository(database)
    use_github(monkeypatch, FakeGitHubClient({}))
    empty = scan(client, repository_id, headers)
    assert empty.status_code == 200
    assert empty.json()["files_discovered"] == 0
    assert empty.json()["chunks_created"] == 0

    use_github(monkeypatch, FakeGitHubClient({"empty.py": ""}))
    with_empty_file = scan(client, repository_id, headers)
    assert with_empty_file.json()["files_indexed"] == 1
    assert with_empty_file.json()["chunks_created"] == 0


def test_rescan_does_not_duplicate_and_only_refetches_changed_files(client, database, monkeypatch):
    repository_id, headers = create_repository(database)
    first = use_github(
        monkeypatch,
        FakeGitHubClient({"a.py": numbered_lines(150), "b.py": "b = 1\n", "gone.py": "g = 1\n"}),
    )
    scan(client, repository_id, headers)
    assert count(database, RepositoryFile) == 3
    assert count(database, RepositoryChunk) == 4

    second = use_github(
        monkeypatch,
        FakeGitHubClient({"a.py": numbered_lines(150), "b.py": "b = 2\nb2 = 3\n", "new.py": "n = 1\n"}),
    )
    response = scan(client, repository_id, headers)

    assert response.json()["files_removed"] == 1
    assert response.json()["chunks_created"] == 2
    assert second.downloads == ["b.py", "new.py"]
    assert set(stored_files(database, repository_id)) == {"a.py", "b.py", "new.py"}
    assert count(database, RepositoryFile) == 3
    assert count(database, RepositoryChunk) == 4
    with database() as session:
        b_chunk = session.scalars(
            select(RepositoryChunk).where(RepositoryChunk.content == "b = 2\nb2 = 3\n")
        ).one()
        assert (b_chunk.start_line, b_chunk.end_line) == (1, 2)

    third = use_github(monkeypatch, FakeGitHubClient({"a.py": numbered_lines(150), "b.py": "b = 2\nb2 = 3\n", "new.py": "n = 1\n"}))
    unchanged = scan(client, repository_id, headers)
    assert unchanged.json()["chunks_created"] == 0
    assert third.downloads == []
    assert count(database, RepositoryFile) == 3
    assert count(database, RepositoryChunk) == 4
    assert first.downloads  # sanity: first scan actually downloaded files


def test_repository_not_found(client, database):
    _, headers = create_repository(database)

    response = scan(client, uuid.uuid4(), headers)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "RESOURCE_NOT_FOUND"


def test_user_cannot_scan_another_users_repository(client, database, monkeypatch):
    other_repository_id, _ = create_repository(database, email="other@example.com")
    _, my_headers = create_repository(database, email="me@example.com")
    fake = use_github(monkeypatch, FakeGitHubClient({"a.py": "x = 1\n"}))

    response = scan(client, other_repository_id, my_headers)

    assert response.status_code == 404
    assert fake.downloads == []
    assert count(database, RepositoryFile) == 0


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (GitHubAuthError("bad credentials"), 403),
        (GitHubNotFoundError("missing"), 404),
        (GitHubContentError("boom"), 502),
    ],
)
def test_github_failures_map_to_safe_errors(client, database, monkeypatch, error, status_code):
    repository_id, headers = create_repository(database)
    use_github(monkeypatch, FakeGitHubClient(tree_error=error))

    response = scan(client, repository_id, headers)

    assert response.status_code == status_code
    assert set(response.json()["error"]) == {"code", "message", "details"}
    assert "bad credentials" not in response.text and "boom" not in response.text


def test_github_failure_rolls_back_partial_index(client, database, monkeypatch):
    repository_id, headers = create_repository(database)

    class FlakyClient(FakeGitHubClient):
        def get_file_content(self, owner, name, sha):
            if self.downloads:
                raise GitHubContentError("rate limited")
            return super().get_file_content(owner, name, sha)

    use_github(monkeypatch, FlakyClient({"a.py": "a = 1\n", "b.py": "b = 1\n"}))

    response = scan(client, repository_id, headers)

    assert response.status_code == 502
    assert count(database, RepositoryFile) == 0
    assert count(database, RepositoryChunk) == 0


def test_oversized_repository_is_rejected(client, database, monkeypatch):
    repository_id, headers = create_repository(database)
    use_github(monkeypatch, FakeGitHubClient({"a.py": "x\n"}, truncated=True))

    assert scan(client, repository_id, headers).status_code == 400


def test_authentication_is_required(client, database, monkeypatch):
    repository_id, _ = create_repository(database)
    fake = use_github(monkeypatch, FakeGitHubClient({"a.py": "x\n"}))

    missing = scan(client, repository_id, {})
    invalid = scan(client, repository_id, {"Authorization": "Bearer not-a-real-token"})

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert fake.downloads == []


def test_github_token_is_used_server_side_but_never_returned_or_logged(
    client, database, monkeypatch, caplog
):
    encrypted = (
        Fernet(get_settings().token_encryption_key.get_secret_value().encode())
        .encrypt(GITHUB_TOKEN.encode())
        .decode()
    )
    repository_id, headers = create_repository(database, encrypted_token=encrypted)
    received: list[str | None] = []
    fake = FakeGitHubClient({"a.py": "x = 1\n"})
    monkeypatch.setattr(
        repository_routes, "GitHubContentClient", lambda token: received.append(token) or fake
    )
    caplog.set_level(logging.DEBUG)

    response = scan(client, repository_id, headers)

    assert response.status_code == 200
    assert received == [GITHUB_TOKEN]
    assert GITHUB_TOKEN not in response.text
    assert encrypted not in response.text
    assert GITHUB_TOKEN not in caplog.text
    assert "access_token" not in response.text


def test_undecryptable_stored_token_is_reported_as_github_access_error(client, database):
    repository_id, headers = create_repository(database, encrypted_token="not-a-fernet-token")

    response = scan(client, repository_id, headers)

    assert response.status_code == 403
    assert "not-a-fernet-token" not in response.text


def test_chunker_handles_edge_cases_and_preserves_text():
    assert chunk_source("") == []
    single = chunk_source("one line, no newline")
    assert [(c.start_line, c.end_line) for c in single] == [(1, 1)]
    text = "a\r\nb\n\nlast"
    assert "".join(c.content for c in chunk_source(text, max_lines=2)) == text
    exact = chunk_source(numbered_lines(200))
    assert [(c.start_line, c.end_line) for c in exact] == [(1, 100), (101, 200)]


@pytest.mark.parametrize(
    ("path", "language"),
    [
        ("a.py", "python"),
        ("a.JSX", "javascript"),
        ("a.tsx", "typescript"),
        ("a.scss", "scss"),
        ("a.hpp", "cpp"),
        ("a.yml", "yaml"),
        ("deep/dir/main.rs", "rust"),
        ("README.md", None),
        ("build/out.js", None),
        ("pkg/venv/x.py", None),
        ("package-lock.json", None),
        ("Makefile", None),
    ],
)
def test_language_detection_and_filtering(path, language):
    assert detect_language(path) == language


def test_github_client_sends_token_and_maps_errors():
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        status = {"/repos/o/empty/git/trees/main": 409, "/repos/o/limited/git/trees/main": 403,
                  "/repos/o/denied/git/trees/main": 401}.get(request.url.path)
        if status == 403:
            return httpx.Response(403, headers={"X-RateLimit-Remaining": "0"})
        if status:
            return httpx.Response(status)
        content = base64.b64encode(b"data").decode()
        if "/git/blobs/" in request.url.path:
            return httpx.Response(200, json={"encoding": "base64", "content": content})
        return httpx.Response(
            200,
            json={
                "truncated": False,
                "tree": [
                    {"path": "a.py", "type": "blob", "sha": "1", "size": 4, "mode": "100644"},
                    {"path": "link", "type": "blob", "sha": "2", "size": 1, "mode": "120000"},
                    {"path": "dir", "type": "tree", "sha": "3"},
                ],
            },
        )

    with GitHubContentClient("secret-token", transport=httpx.MockTransport(handler)) as github:
        tree = github.get_tree("o", "project", "main")
        assert [e.path for e in tree.entries] == ["a.py"]
        assert seen["authorization"] == "Bearer secret-token"
        assert github.get_file_content("o", "project", "1") == b"data"
        assert github.get_tree("o", "empty", "main").entries == []
        with pytest.raises(GitHubAuthError):
            github.get_tree("o", "denied", "main")
        with pytest.raises(GitHubContentError, match="rate limit"):
            github.get_tree("o", "limited", "main")

    with GitHubContentClient(transport=httpx.MockTransport(handler)) as anonymous:
        anonymous.get_tree("o", "project", "main")
        assert seen["authorization"] is None
