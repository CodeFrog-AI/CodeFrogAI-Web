"""Tests for listing GitHub repositories and connecting one to CodeFrog."""

import logging

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes import repositories as repository_routes
from app.auth.security import create_access_token
from app.core.config import get_settings
from app.db.base import Base
from app.db.database import get_db
from app.db.models import GitHubAccount, Repository, User
from app.integrations.github.contents import (
    GitHubAuthError,
    GitHubContentClient,
    GitHubContentError,
    GitHubNotFoundError,
    GitHubRepository,
)
from main import app

GITHUB_TOKEN = "gho_plaintext_token_that_must_never_leak"


def make_repo(github_id: int, name: str = "project", *, owner="octocat", private=False, branch="main"):
    return GitHubRepository(github_id, owner, name, branch, private)


class FakeGitHub:
    """GitHub double: `accessible` is the user's list; `existing` are other visible repos."""

    def __init__(self, accessible=(), existing=(), *, error=None):
        self.accessible = list(accessible)
        self.existing = {repo.github_repository_id: repo for repo in existing}
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def list_repositories(self):
        if self.error:
            raise self.error
        return self.accessible

    def get_repository_by_id(self, github_repository_id):
        if github_repository_id in self.existing:
            return self.existing[github_repository_id]
        raise GitHubNotFoundError("missing")


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


def encrypt(token: str) -> str:
    key = get_settings().token_encryption_key.get_secret_value().encode()
    return Fernet(key).encrypt(token.encode()).decode()


def create_user(factory, email="owner@example.com", *, with_account=True, token=GITHUB_TOKEN, github_id=1):
    with factory() as session:
        user = User(email=email, name="Owner", status="active")
        session.add(user)
        session.flush()
        if with_account:
            session.add(
                GitHubAccount(
                    user_id=user.id,
                    github_user_id=github_id,
                    login=email.split("@")[0],
                    access_token_encrypted=encrypt(token) if token else None,
                )
            )
        session.commit()
        return {"Authorization": f"Bearer {create_access_token(user.id)}"}


def use_github(monkeypatch, fake):
    received: list[str | None] = []
    monkeypatch.setattr(
        repository_routes, "GitHubContentClient", lambda token: received.append(token) or fake
    )
    return received


def repository_count(factory) -> int:
    with factory() as session:
        return session.scalar(select(func.count()).select_from(Repository))


def connect(client, headers, github_id):
    return client.post(
        "/api/v1/repositories/connect", json={"github_repository_id": github_id}, headers=headers
    )


def test_authenticated_user_can_list_repositories(client, database, monkeypatch):
    headers = create_user(database)
    fake = FakeGitHub([make_repo(10, "public-app"), make_repo(11, "secret", private=True, branch="dev")])
    received = use_github(monkeypatch, fake)
    assert connect(client, headers, 10).status_code == 201

    response = client.get("/api/v1/repositories/github", headers=headers)

    assert response.status_code == 200
    assert received[-1] == GITHUB_TOKEN
    assert response.json()["repositories"] == [
        {"github_repository_id": 10, "owner": "octocat", "name": "public-app",
         "default_branch": "main", "private": False, "connected": True},
        {"github_repository_id": 11, "owner": "octocat", "name": "secret",
         "default_branch": "dev", "private": True, "connected": False},
    ]


def test_user_without_github_account_gets_conflict_error(client, database, monkeypatch):
    headers = create_user(database, with_account=False)
    use_github(monkeypatch, FakeGitHub())

    listing = client.get("/api/v1/repositories/github", headers=headers)
    connecting = connect(client, headers, 10)

    for response in (listing, connecting):
        assert response.status_code == 409
        assert "GitHub account" in response.json()["error"]["message"]


def test_account_without_stored_token_is_asked_to_reconnect(client, database, monkeypatch):
    headers = create_user(database, token=None)
    use_github(monkeypatch, FakeGitHub([make_repo(10)]))

    assert client.get("/api/v1/repositories/github", headers=headers).status_code == 403
    assert connect(client, headers, 10).status_code == 403


def test_authentication_is_required(client, database, monkeypatch):
    use_github(monkeypatch, FakeGitHub([make_repo(10)]))

    assert client.get("/api/v1/repositories/github").status_code == 401
    assert client.post("/api/v1/repositories/connect", json={"github_repository_id": 10}).status_code == 401
    bad = {"Authorization": "Bearer invalid"}
    assert client.get("/api/v1/repositories/github", headers=bad).status_code == 401
    assert repository_count(database) == 0


def test_user_can_connect_an_accessible_repository(client, database, monkeypatch):
    headers = create_user(database)
    use_github(monkeypatch, FakeGitHub([make_repo(10, "app", private=True, branch="trunk")]))

    response = connect(client, headers, 10)

    assert response.status_code == 201
    body = response.json()
    assert (body["github_repository_id"], body["owner"], body["name"]) == (10, "octocat", "app")
    assert (body["default_branch"], body["private"], body["connection_status"]) == ("trunk", True, "connected")
    with database() as session:
        stored = session.scalars(select(Repository)).one()
        assert str(stored.id) == body["id"]
        assert stored.github_account.login == "owner"
        assert stored.connection_metadata == {"private": True}


def test_connecting_twice_reuses_the_record_and_refreshes_metadata(client, database, monkeypatch):
    headers = create_user(database)
    use_github(monkeypatch, FakeGitHub([make_repo(10, "app", branch="main")]))
    first = connect(client, headers, 10)

    use_github(monkeypatch, FakeGitHub([make_repo(10, "app-renamed", branch="develop")]))
    second = connect(client, headers, 10)

    assert (first.status_code, second.status_code) == (201, 200)
    assert first.json()["id"] == second.json()["id"]
    assert second.json()["name"] == "app-renamed"
    assert second.json()["default_branch"] == "develop"
    assert repository_count(database) == 1


def test_inaccessible_repository_is_rejected_with_403(client, database, monkeypatch):
    headers = create_user(database)
    use_github(monkeypatch, FakeGitHub([make_repo(10)], existing=[make_repo(99, "someone-elses", owner="stranger")]))

    response = connect(client, headers, 99)

    assert response.status_code == 403
    assert repository_count(database) == 0


def test_nonexistent_repository_is_rejected_with_404(client, database, monkeypatch):
    headers = create_user(database)
    use_github(monkeypatch, FakeGitHub([make_repo(10)]))

    response = connect(client, headers, 12345)

    assert response.status_code == 404
    assert repository_count(database) == 0


def test_repository_connected_to_another_user_is_not_shared(client, database, monkeypatch):
    first = create_user(database, "first@example.com", github_id=1)
    second = create_user(database, "second@example.com", github_id=2)
    use_github(monkeypatch, FakeGitHub([make_repo(10)]))
    assert connect(client, first, 10).status_code == 201

    response = connect(client, second, 10)

    assert response.status_code == 409
    assert repository_count(database) == 1
    with database() as session:
        assert session.scalars(select(Repository)).one().github_account.login == "first"


@pytest.mark.parametrize(
    ("error", "status_code"),
    [(GitHubAuthError("bad credentials"), 403), (GitHubContentError("boom"), 502)],
)
def test_github_failures_are_handled_safely(client, database, monkeypatch, error, status_code):
    headers = create_user(database)
    use_github(monkeypatch, FakeGitHub(error=error))

    for response in (client.get("/api/v1/repositories/github", headers=headers), connect(client, headers, 10)):
        assert response.status_code == status_code
        assert "bad credentials" not in response.text and "boom" not in response.text
    assert repository_count(database) == 0


@pytest.mark.parametrize("bad_id", [0, -5, 2**31, "abc"])
def test_invalid_repository_id_is_rejected(client, database, monkeypatch, bad_id):
    headers = create_user(database)
    use_github(monkeypatch, FakeGitHub([make_repo(10)]))

    assert connect(client, headers, bad_id).status_code == 422


def test_github_token_is_never_returned_or_logged(client, database, monkeypatch, caplog):
    headers = create_user(database)
    use_github(monkeypatch, FakeGitHub([make_repo(10)]))
    caplog.set_level(logging.DEBUG)

    responses = [client.get("/api/v1/repositories/github", headers=headers), connect(client, headers, 10)]

    for response in responses:
        assert response.status_code in (200, 201)
        for secret in (GITHUB_TOKEN, encrypt(GITHUB_TOKEN)[:20], "access_token"):
            assert secret not in response.text
    assert GITHUB_TOKEN not in caplog.text


def test_github_client_lists_paginated_repositories_and_looks_up_by_id():
    def repo_json(number: int) -> dict:
        return {"id": number, "name": f"r{number}", "owner": {"login": "octocat"},
                "default_branch": "main", "private": number == 1}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {GITHUB_TOKEN}"
        if request.url.path == "/user/repos":
            page = int(request.url.params["page"])
            count = 100 if page == 1 else 3
            return httpx.Response(200, json=[repo_json(n) for n in range((page - 1) * 100 + 1, (page - 1) * 100 + count + 1)])
        if request.url.path == "/repositories/7":
            return httpx.Response(200, json=repo_json(7))
        return httpx.Response(404)

    with GitHubContentClient(GITHUB_TOKEN, transport=httpx.MockTransport(handler)) as github:
        repositories = github.list_repositories()
        assert len(repositories) == 103
        assert repositories[0].private is True and repositories[1].private is False
        assert repositories[0].owner == "octocat"
        assert github.get_repository_by_id(7).name == "r7"
        with pytest.raises(GitHubNotFoundError):
            github.get_repository_by_id(8)

    malformed = httpx.MockTransport(lambda _request: httpx.Response(200, json=[{"id": 1}]))
    with GitHubContentClient(GITHUB_TOKEN, transport=malformed) as github:
        with pytest.raises(GitHubContentError):
            github.list_repositories()
