"""Tests for GitHub OAuth safety, account linking, and CodeFrog JWT issuance."""

import logging
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from app.api.routes import github_oauth
from app.auth.github_service import resolve_github_identity
from app.auth.oauth_state import OAuthStateStore
from app.core.config import get_settings
from app.db.database import get_db
from app.db.models import GitHubAccount, User
from app.integrations.github.oauth import GitHubIdentity, GitHubOAuthError, GitHubOAuthClient
from main import app


class FakeQuery:
    """Minimal query support for user and GitHub identity lookup tests."""

    def __init__(self, session, model):  # type: ignore[no-untyped-def]
        self.session = session
        self.model = model
        self.field: str | None = None
        self.value = None

    def filter(self, criterion):  # type: ignore[no-untyped-def]
        self.field = criterion.left.name
        self.value = criterion.right.value
        return self

    def first(self):  # type: ignore[no-untyped-def]
        if self.model is User and self.field == "email":
            return self.session.users_by_email.get(self.value)
        if self.model is GitHubAccount and self.field == "github_user_id":
            return self.session.accounts.get(self.value)
        return None


class FakeSession:
    """In-memory database boundary for OAuth service and route tests."""

    def __init__(self):
        self.users_by_email: dict[str, User] = {}
        self.users_by_id: dict[object, User] = {}
        self.accounts: dict[int, GitHubAccount] = {}

    def query(self, model):  # type: ignore[no-untyped-def]
        return FakeQuery(self, model)

    def add(self, item):  # type: ignore[no-untyped-def]
        if isinstance(item, User):
            self.users_by_email[item.email] = item
        else:
            self.accounts[item.github_user_id] = item

    def flush(self) -> None:
        for user in self.users_by_email.values():
            if user.id is None:
                user.id = uuid4()
            self.users_by_id[user.id] = user

    def commit(self) -> None:
        self.flush()

    def rollback(self) -> None:
        return None

    def refresh(self, user: User) -> None:
        self.flush()
        user.created_at = datetime.now(timezone.utc)

    def get(self, model, identifier):  # type: ignore[no-untyped-def]
        return self.users_by_id.get(identifier) if model is User else None


class SuccessfulGitHubClient:
    """No-network OAuth provider double that deliberately retains no tokens."""

    def build_authorization_url(self, state: str) -> str:
        return GitHubOAuthClient().build_authorization_url(state)

    def exchange_code(self, _code: str) -> str:
        return "github-access-token-that-must-not-leak"

    def get_identity(self, _access_token: str) -> GitHubIdentity:
        return GitHubIdentity(
            github_user_id=12345,
            login="octocat",
            email="octocat@example.com",
            name="The Octocat",
        )


class FailingGitHubClient(SuccessfulGitHubClient):
    def exchange_code(self, _code: str) -> str:
        raise GitHubOAuthError("provider token and code must not leak")


@pytest.fixture
def oauth_client(monkeypatch):
    session = FakeSession()
    app.dependency_overrides[get_db] = lambda: session
    monkeypatch.setattr(github_oauth, "GitHubOAuthClient", SuccessfulGitHubClient)
    with TestClient(app) as test_client:
        yield test_client, session
    app.dependency_overrides.clear()


def begin_oauth(test_client: TestClient) -> str:
    response = test_client.get("/api/v1/auth/github/login", follow_redirects=False)
    assert response.status_code == 302
    return parse_qs(urlparse(response.headers["location"]).query)["state"][0]


def test_login_redirect_uses_configured_client_redirect_and_random_state(oauth_client):
    test_client, _ = oauth_client

    first = test_client.get("/api/v1/auth/github/login", follow_redirects=False)
    second = test_client.get("/api/v1/auth/github/login", follow_redirects=False)
    first_query = parse_qs(urlparse(first.headers["location"]).query)
    second_query = parse_qs(urlparse(second.headers["location"]).query)

    assert first.headers["location"].startswith("https://github.com/login/oauth/authorize?")
    assert first_query["client_id"] == ["test-github-client-id"]
    assert first_query["redirect_uri"] == ["http://localhost:8000/api/v1/auth/github/callback"]
    assert first_query["scope"] == ["read:user user:email"]
    assert first_query["state"][0] != second_query["state"][0]
    assert len(first_query["state"][0]) >= 32


def test_oauth_state_is_single_use_and_missing_or_invalid_state_is_rejected(oauth_client):
    test_client, _ = oauth_client
    state = begin_oauth(test_client)

    valid = test_client.get(f"/api/v1/auth/github/callback?state={state}&code=code")
    reused = test_client.get(f"/api/v1/auth/github/callback?state={state}&code=code")
    missing = test_client.get("/api/v1/auth/github/callback?code=code")
    invalid = test_client.get("/api/v1/auth/github/callback?state=invalid&code=code")

    assert valid.status_code == 200
    assert reused.status_code == 401
    assert missing.status_code == 401
    assert invalid.status_code == 401


def test_missing_code_and_provider_failures_return_safe_errors(oauth_client, monkeypatch):
    test_client, _ = oauth_client
    state = begin_oauth(test_client)

    missing_code = test_client.get(f"/api/v1/auth/github/callback?state={state}")
    assert missing_code.status_code == 400

    state = begin_oauth(test_client)
    monkeypatch.setattr(github_oauth, "GitHubOAuthClient", FailingGitHubClient)
    provider_failure = test_client.get(f"/api/v1/auth/github/callback?state={state}&code=unsafe-code")

    assert provider_failure.status_code == 400
    assert "unsafe-code" not in provider_failure.text
    assert "provider token" not in provider_failure.text


def test_existing_local_user_is_linked_and_oauth_user_has_no_invented_password():
    session = FakeSession()
    existing_user = User(email="octocat@example.com", name="Existing", status="active")
    session.add(existing_user)
    session.flush()
    identity = SuccessfulGitHubClient().get_identity("unused")

    linked_user = resolve_github_identity(session, identity)
    linked_again = resolve_github_identity(session, identity)

    assert linked_user is existing_user
    assert session.accounts[12345].user_id == existing_user.id
    assert linked_again is existing_user

    new_session = FakeSession()
    new_identity = GitHubIdentity(7, "new-user", "new@example.com", "New User")
    new_user = resolve_github_identity(new_session, new_identity)
    assert new_user.password_hash is None
    assert new_session.accounts[7].access_token_encrypted is None


def test_callback_issues_codefrog_jwt_and_the_existing_me_endpoint_accepts_it(oauth_client):
    test_client, _ = oauth_client
    state = begin_oauth(test_client)

    callback = test_client.get(f"/api/v1/auth/github/callback?state={state}&code=code")
    access_token = callback.json()["access_token"]
    current_user = test_client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {access_token}"}
    )

    assert callback.status_code == 200
    assert "github-access-token" not in callback.text
    assert current_user.status_code == 200
    assert current_user.json()["email"] == "octocat@example.com"
    assert "password_hash" not in current_user.text


def test_tokens_codes_and_client_secrets_are_not_logged(oauth_client, caplog):
    test_client, _ = oauth_client
    caplog.set_level(logging.INFO)
    state = begin_oauth(test_client)
    callback = test_client.get(f"/api/v1/auth/github/callback?state={state}&code=unsafe-code")

    assert callback.status_code == 200
    assert "unsafe-code" not in caplog.text
    assert "github-access-token" not in caplog.text
    assert "test-github-client-secret" not in caplog.text


def test_callback_stores_the_github_token_encrypted_only(oauth_client):
    test_client, session = oauth_client
    state = begin_oauth(test_client)

    callback = test_client.get(f"/api/v1/auth/github/callback?state={state}&code=code")

    stored = session.accounts[12345].access_token_encrypted
    key = get_settings().token_encryption_key.get_secret_value().encode()
    assert callback.status_code == 200
    assert stored is not None and "github-access-token" not in stored
    assert Fernet(key).decrypt(stored.encode()).decode() == "github-access-token-that-must-not-leak"
    assert "github-access-token" not in callback.text


def test_state_store_rejects_expired_values(monkeypatch):
    store = OAuthStateStore(ttl_seconds=1)
    state = store.create()
    monkeypatch.setattr("app.auth.oauth_state.time.monotonic", lambda: 10_000_000.0)

    assert not store.consume(state)
