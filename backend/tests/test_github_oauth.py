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
from app.core.config import Settings, get_settings
from app.db.database import get_db
from app.db.models import GitHubAccount, User
from app.integrations.github.oauth import GitHubIdentity, GitHubOAuthError, GitHubOAuthClient, oauth_scopes
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
        if self.model is GitHubAccount and self.field == "user_id":
            return next((a for a in self.session.accounts.values() if a.user_id == self.value), None)
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


FRONTEND_CALLBACK = "http://localhost:3000/auth/callback"


def finish_oauth(test_client: TestClient, query: str):
    """Call the callback without following the redirect to the (not running) frontend."""

    return test_client.get(f"/api/v1/auth/github/callback?{query}", follow_redirects=False)


def fragment_of(response) -> dict[str, str]:
    """The `#...` part of the redirect target, parsed."""

    assert response.status_code == 302
    return {key: values[0] for key, values in parse_qs(urlparse(response.headers["location"]).fragment).items()}


def error_of(response) -> str:
    fragment = fragment_of(response)
    assert set(fragment) == {"error"}
    assert response.headers["location"].startswith(f"{FRONTEND_CALLBACK}#error=")
    return fragment["error"]


def test_login_redirect_uses_configured_client_redirect_and_random_state(oauth_client):
    test_client, _ = oauth_client

    first = test_client.get("/api/v1/auth/github/login", follow_redirects=False)
    second = test_client.get("/api/v1/auth/github/login", follow_redirects=False)
    first_query = parse_qs(urlparse(first.headers["location"]).query)
    second_query = parse_qs(urlparse(second.headers["location"]).query)

    assert first.headers["location"].startswith("https://github.com/login/oauth/authorize?")
    assert first_query["client_id"] == ["test-github-client-id"]
    assert first_query["redirect_uri"] == ["http://localhost:8000/api/v1/auth/github/callback"]
    assert first_query["scope"] == ["read:user user:email repo"]
    assert first_query["state"][0] != second_query["state"][0]
    assert len(first_query["state"][0]) >= 32


def test_oauth_state_is_single_use_and_missing_or_invalid_state_is_rejected(oauth_client):
    test_client, _ = oauth_client
    state = begin_oauth(test_client)

    valid = finish_oauth(test_client, f"state={state}&code=code")
    reused = finish_oauth(test_client, f"state={state}&code=code")
    missing = finish_oauth(test_client, "code=code")
    invalid = finish_oauth(test_client, "state=invalid&code=code")

    assert "access_token" in fragment_of(valid)
    assert [error_of(response) for response in (reused, missing, invalid)] == ["invalid_state"] * 3


def test_missing_code_and_provider_failures_return_safe_errors(oauth_client, monkeypatch):
    test_client, _ = oauth_client
    state = begin_oauth(test_client)

    missing_code = finish_oauth(test_client, f"state={state}")
    assert error_of(missing_code) == "authorization_failed"

    state = begin_oauth(test_client)
    monkeypatch.setattr(github_oauth, "GitHubOAuthClient", FailingGitHubClient)
    provider_failure = finish_oauth(test_client, f"state={state}&code=unsafe-code")

    assert error_of(provider_failure) == "authorization_failed"
    assert "unsafe-code" not in provider_failure.headers["location"]
    assert "provider token" not in provider_failure.headers["location"]


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

    callback = finish_oauth(test_client, f"state={state}&code=code")
    access_token = fragment_of(callback)["access_token"]
    current_user = test_client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {access_token}"}
    )

    assert callback.status_code == 302
    assert "github-access-token" not in callback.headers["location"]
    assert current_user.status_code == 200
    assert current_user.json()["email"] == "octocat@example.com"
    assert "password_hash" not in current_user.text


def test_tokens_codes_and_client_secrets_are_not_logged(oauth_client, caplog):
    test_client, _ = oauth_client
    caplog.set_level(logging.INFO)
    state = begin_oauth(test_client)
    callback = finish_oauth(test_client, f"state={state}&code=unsafe-code")
    access_token = fragment_of(callback)["access_token"]

    assert callback.status_code == 302
    assert access_token not in caplog.text
    assert "unsafe-code" not in caplog.text
    assert "github-access-token" not in caplog.text
    assert "test-github-client-secret" not in caplog.text


def test_callback_stores_the_github_token_encrypted_only(oauth_client):
    test_client, session = oauth_client
    state = begin_oauth(test_client)

    callback = finish_oauth(test_client, f"state={state}&code=code")

    stored = session.accounts[12345].access_token_encrypted
    key = get_settings().token_encryption_key.get_secret_value().encode()
    assert callback.status_code == 302
    assert stored is not None and "github-access-token" not in stored
    assert Fernet(key).decrypt(stored.encode()).decode() == "github-access-token-that-must-not-leak"
    assert "github-access-token" not in callback.headers["location"]


def test_state_store_rejects_expired_values(monkeypatch):
    store = OAuthStateStore(ttl_seconds=1)
    state = store.create()
    monkeypatch.setattr("app.auth.oauth_state.time.monotonic", lambda: 10_000_000.0)

    assert not store.consume(state)


# ------------------------------------------------------------------ redirect to the web frontend


def test_successful_callback_redirects_to_the_frontend_with_the_jwt_in_the_fragment_only(oauth_client):
    test_client, _ = oauth_client
    state = begin_oauth(test_client)

    callback = finish_oauth(test_client, f"state={state}&code=code")

    location = urlparse(callback.headers["location"])
    token = fragment_of(callback)["access_token"]
    assert callback.status_code == 302
    assert f"{location.scheme}://{location.netloc}{location.path}" == FRONTEND_CALLBACK
    assert location.query == "" and location.params == ""  # the JWT is never in the query string
    assert location.fragment == f"access_token={token}"
    assert token.count(".") == 2 and "?" not in callback.headers["location"]
    assert callback.headers["cache-control"] == "no-store"
    assert callback.headers["referrer-policy"] == "no-referrer"
    assert "github-access-token" not in callback.headers["location"]


def test_the_redirect_target_comes_from_frontend_url(oauth_client, monkeypatch):
    test_client, _ = oauth_client
    monkeypatch.setattr(get_settings(), "frontend_url", "https://app.example.test")
    state = begin_oauth(test_client)

    callback = finish_oauth(test_client, f"state={state}&code=code")

    assert callback.headers["location"].startswith("https://app.example.test/auth/callback#access_token=")


def test_a_denied_authorization_redirects_with_a_safe_error_code(oauth_client):
    test_client, session = oauth_client
    state = begin_oauth(test_client)

    denied = finish_oauth(test_client, f"state={state}&error=access_denied&error_description=The+user+denied")

    assert error_of(denied) == "access_denied"
    assert "denied" not in denied.headers["location"].split("#", 1)[1].replace("access_denied", "")
    assert session.accounts == {}


def test_other_provider_errors_and_unknown_error_values_become_a_fixed_code(oauth_client):
    test_client, _ = oauth_client
    state = begin_oauth(test_client)

    failed = finish_oauth(test_client, f"state={state}&error=%3Cscript%3Ealert(1)%3C/script%3E")

    assert error_of(failed) == "authorization_failed"
    assert "script" not in failed.headers["location"]


def test_error_redirects_never_contain_secrets_or_stack_traces(oauth_client, monkeypatch):
    test_client, _ = oauth_client
    monkeypatch.setattr(github_oauth, "GitHubOAuthClient", FailingGitHubClient)
    state = begin_oauth(test_client)

    failed = finish_oauth(test_client, f"state={state}&code=unsafe-code")

    location = failed.headers["location"]
    for secret in ("github-access-token", "test-github-client-secret", "unsafe-code", "Traceback", "provider token"):
        assert secret not in location
    assert error_of(failed) == "authorization_failed" and "access_token" not in location


def test_state_cookie_is_bound_to_the_browser_and_cleared_after_the_callback(oauth_client):
    test_client, _ = oauth_client
    state = begin_oauth(test_client)
    test_client.cookies.clear()  # a different browser: the state exists on the server but not in this cookie jar

    other_browser = finish_oauth(test_client, f"state={state}&code=code")

    assert error_of(other_browser) == "invalid_state"


def test_the_state_is_consumed_even_when_authorization_fails(oauth_client):
    test_client, _ = oauth_client
    state = begin_oauth(test_client)

    assert error_of(finish_oauth(test_client, f"state={state}&error=access_denied")) == "access_denied"
    assert error_of(finish_oauth(test_client, f"state={state}&code=code")) == "invalid_state"


# ------------------------------------------------------------------ token encryption must not be skipped


def test_a_missing_encryption_key_fails_the_callback_without_linking_or_issuing_a_jwt(oauth_client, monkeypatch):
    test_client, session = oauth_client
    state = begin_oauth(test_client)
    monkeypatch.setattr(get_settings(), "token_encryption_key", None)

    callback = finish_oauth(test_client, f"state={state}&code=code")

    assert error_of(callback) == "server_error"
    assert "access_token" not in callback.headers["location"]
    assert session.accounts == {} and session.users_by_email == {}


def test_an_invalid_encryption_key_fails_the_callback(oauth_client, monkeypatch):
    from pydantic import SecretStr

    test_client, session = oauth_client
    state = begin_oauth(test_client)
    monkeypatch.setattr(get_settings(), "token_encryption_key", SecretStr("not-a-valid-fernet-key"))

    callback = finish_oauth(test_client, f"state={state}&code=code")

    assert error_of(callback) == "server_error"
    assert "not-a-valid-fernet-key" not in callback.headers["location"]
    assert session.accounts == {}


def test_login_does_not_send_the_user_to_github_when_tokens_could_not_be_stored(oauth_client, monkeypatch):
    test_client, _ = oauth_client
    monkeypatch.setattr(get_settings(), "token_encryption_key", None)

    login = test_client.get("/api/v1/auth/github/login", follow_redirects=False)

    assert login.headers["location"].startswith(FRONTEND_CALLBACK) and error_of(login) == "server_error"
    assert "github_oauth_state" not in login.headers.get("set-cookie", "")


def test_encryption_failures_are_never_silent_in_the_service(monkeypatch):
    from app.auth.github_service import store_github_token
    from app.integrations.github.tokens import TokenEncryptionError

    session = FakeSession()
    resolve_github_identity(session, GitHubIdentity(9, "someone", "someone@example.com", None))
    monkeypatch.setattr(get_settings(), "token_encryption_key", None)

    with pytest.raises(TokenEncryptionError):
        store_github_token(session, 9, "github-access-token-that-must-not-leak")

    assert session.accounts[9].access_token_encrypted is None


def test_the_encryption_key_and_token_are_not_logged_on_failure(oauth_client, monkeypatch, caplog):
    test_client, _ = oauth_client
    caplog.set_level(logging.DEBUG)
    state = begin_oauth(test_client)
    monkeypatch.setattr(get_settings(), "token_encryption_key", None)

    finish_oauth(test_client, f"state={state}&code=code")

    assert "github-access-token" not in caplog.text and "test-github-client-secret" not in caplog.text


# ------------------------------------------------------------------ configuration


def test_default_scopes_include_identity_and_repository_access():
    assert Settings.model_fields["github_oauth_scopes"].default == "read:user user:email repo"
    assert Settings.model_fields["frontend_url"].default == "http://localhost:3000"
    assert oauth_scopes() == "read:user user:email repo"


def test_the_authorization_url_uses_the_configured_scopes(oauth_client, monkeypatch):
    test_client, _ = oauth_client
    monkeypatch.setattr(get_settings(), "github_oauth_scopes", "repo workflow")

    query = parse_qs(urlparse(test_client.get("/api/v1/auth/github/login", follow_redirects=False).headers["location"]).query)

    assert query["scope"] == ["read:user user:email repo workflow"]


@pytest.mark.parametrize(
    ("configured", "requested"),
    [
        ("repo", "read:user user:email repo"),  # a .env with only `repo` still gets the identity scopes
        ("user:email read:user repo", "read:user user:email repo"),
        ("repo,workflow", "read:user user:email repo workflow"),
        ("repo repo", "read:user user:email repo"),
    ],
)
def test_identity_scopes_are_always_requested(monkeypatch, configured, requested):
    normalized = Settings(github_oauth_scopes=configured).github_oauth_scopes
    monkeypatch.setattr(get_settings(), "github_oauth_scopes", normalized)

    assert oauth_scopes() == requested


@pytest.mark.parametrize("bad", ["", "   ", "repo; rm -rf /", "repo&client_secret=x", "repo" + chr(0) + "admin", "a" * 65, "re po%20x"])
def test_invalid_scope_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        Settings(github_oauth_scopes=bad)


@pytest.mark.parametrize(
    ("configured", "normalized"),
    [
        ("http://localhost:3000", "http://localhost:3000"),
        ("http://localhost:3000/", "http://localhost:3000"),
        ("https://app.example.com", "https://app.example.com"),
        ("HTTP://LOCALHOST:3000", "http://localhost:3000"),
    ],
)
def test_frontend_url_is_normalized_to_an_origin(configured, normalized):
    assert Settings(frontend_url=configured).frontend_url == normalized


@pytest.mark.parametrize(
    "bad",
    ["", "localhost:3000", "ftp://localhost", "http://user:pass@localhost:3000", "http://localhost:3000/app", "http://localhost:3000?x=1", "http://localhost:3000#frag", "javascript:alert(1)", "http://"],
)
def test_invalid_frontend_urls_are_rejected(bad):
    with pytest.raises(ValueError):
        Settings(frontend_url=bad)


def test_cors_allows_only_the_configured_frontend_origin(oauth_client):
    test_client, _ = oauth_client

    allowed = test_client.get("/", headers={"Origin": "http://localhost:3000"})
    other_host = test_client.get("/", headers={"Origin": "http://127.0.0.1:3000"})
    unknown = test_client.get("/", headers={"Origin": "https://evil.example"})

    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-origin" not in other_host.headers
    assert "access-control-allow-origin" not in unknown.headers


# ------------------------------------------------------------------ the frontend learns who is connected


def test_me_reports_the_github_connection_without_any_token(oauth_client):
    test_client, _ = oauth_client
    state = begin_oauth(test_client)
    token = fragment_of(finish_oauth(test_client, f"state={state}&code=code"))["access_token"]

    me = test_client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}).json()

    assert me["github_login"] == "octocat" and me["github_connected"] is True
    assert "github-access-token" not in str(me) and "access_token" not in me
