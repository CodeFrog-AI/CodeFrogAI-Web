"""Security and API tests for the local authentication foundation."""

import logging
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.auth.security import hash_password, verify_password
from app.db.database import get_db
from app.db.models import User
from main import app


PASSWORD = "a-safe-local-password"


class FakeQuery:
    """Minimal query implementation for authentication route unit tests."""

    def __init__(self, users: dict[str, User]):
        self.users = users
        self.email: str | None = None

    def filter(self, criterion):  # type: ignore[no-untyped-def]
        self.email = criterion.right.value
        return self

    def first(self) -> User | None:
        return self.users.get(self.email or "")


class FakeSession:
    """In-memory session boundary; no local database is required for these tests."""

    def __init__(self):
        self.users: dict[str, User] = {}

    def query(self, _model):  # type: ignore[no-untyped-def]
        return FakeQuery(self.users)

    def add(self, user: User) -> None:
        if user.id is None:
            user.id = uuid4()
        self.users[user.email] = user

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def refresh(self, user: User) -> None:
        user.created_at = datetime.now(timezone.utc)

    def get(self, _model, user_id):  # type: ignore[no-untyped-def]
        return next((user for user in self.users.values() if user.id == user_id), None)


@pytest.fixture
def client():
    session = FakeSession()
    app.dependency_overrides[get_db] = lambda: session
    with TestClient(app) as test_client:
        yield test_client, session
    app.dependency_overrides.clear()


def registration_payload(**overrides: str) -> dict[str, str]:
    return {"email": "developer@example.com", "name": "Developer", "password": PASSWORD, **overrides}


def test_argon2_hashes_and_verifies_passwords():
    password_hash = hash_password(PASSWORD)

    assert password_hash != PASSWORD
    assert password_hash.startswith("$argon2")
    assert verify_password(PASSWORD, password_hash)
    assert not verify_password("incorrect-password", password_hash)


def test_registration_persists_hash_and_returns_only_public_user(client):
    test_client, session = client

    response = test_client.post("/api/v1/auth/register", json=registration_payload())

    assert response.status_code == 201
    assert response.json()["email"] == "developer@example.com"
    assert "password" not in response.text
    assert "password_hash" not in response.text
    stored_user = session.users["developer@example.com"]
    assert stored_user.password_hash != PASSWORD
    assert verify_password(PASSWORD, stored_user.password_hash or "")


def test_registration_rejects_duplicates_and_short_passwords(client):
    test_client, _ = client
    test_client.post("/api/v1/auth/register", json=registration_payload())

    duplicate = test_client.post("/api/v1/auth/register", json=registration_payload())
    submitted_password = "abc"
    short_password = test_client.post(
        "/api/v1/auth/register", json=registration_payload(password=submitted_password)
    )

    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "CONFLICT"
    assert short_password.status_code == 422
    assert submitted_password not in short_password.text


def test_login_and_protected_identity_flow(client):
    test_client, _ = client
    test_client.post("/api/v1/auth/register", json=registration_payload())

    login = test_client.post(
        "/api/v1/auth/login",
        json={"email": "developer@example.com", "password": PASSWORD},
    )
    token = login.json()["access_token"]
    current_user = test_client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
    )

    assert login.status_code == 200
    assert login.json()["token_type"] == "bearer"
    assert PASSWORD not in login.text
    assert current_user.status_code == 200
    assert current_user.json()["email"] == "developer@example.com"
    assert "password_hash" not in current_user.text


def test_invalid_credentials_and_missing_token_use_safe_unauthorized_response(client):
    test_client, _ = client
    test_client.post("/api/v1/auth/register", json=registration_payload())

    invalid_login = test_client.post(
        "/api/v1/auth/login",
        json={"email": "developer@example.com", "password": "wrong-password"},
    )
    missing_token = test_client.get("/api/v1/auth/me")

    for response in (invalid_login, missing_token):
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHORIZED"
    assert "wrong-password" not in invalid_login.text


def test_authentication_bodies_and_tokens_are_not_logged(client, caplog):
    test_client, _ = client
    caplog.set_level(logging.INFO, logger="codefrog.api")
    response = test_client.post("/api/v1/auth/register", json=registration_payload())

    assert response.status_code == 201
    assert PASSWORD not in caplog.text
    assert "password_hash" not in caplog.text
