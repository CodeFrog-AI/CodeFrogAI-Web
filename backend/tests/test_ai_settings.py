"""Tests for the per-user AI provider settings (storage, API, and provider resolution)."""

import logging

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agent.llm import LLMError, LLMNotConfiguredError
from app.ai_settings import service
from app.api.routes import repositories as repository_routes
from app.core.config import get_settings
from app.core.secret_box import decrypt_secret
from app.db.base import Base
from app.db.database import get_db
from app.db.models import User, UserAISettings
from app.embeddings.provider import EmbeddingNotConfiguredError
from main import app
from tests.test_agent_code_editing import REQUEST as EDIT_REQUEST, plan as edit_plan
from tests.test_pr_review_fix import FINDING
from tests.test_repository_scan import FakeGitHubClient, create_repository, scan, use_github
from tests.test_semantic_search import FakeProvider

URL = "/api/v1/settings/ai"
USER_LLM_KEY = "sk-user-llm-key-1111-AAAA"
USER_EMB_KEY = "sk-user-embedding-key-2222-BBBB"
SERVER_LLM_KEY = "sk-server-llm-key-9999"
SERVER_EMB_KEY = "sk-server-embedding-key-8888"
ALL_KEYS = (USER_LLM_KEY, USER_EMB_KEY, SERVER_LLM_KEY, SERVER_EMB_KEY)


@pytest.fixture
def database():
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
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


@pytest.fixture(autouse=True)
def server_config(monkeypatch):
    """Pin the server-side configuration so a developer's real .env never leaks into these tests."""

    settings = get_settings()
    monkeypatch.setattr(settings, "llm_api_key", SecretStr(SERVER_LLM_KEY))
    monkeypatch.setattr(settings, "llm_model", "server-llm-model")
    monkeypatch.setattr(settings, "llm_base_url", "https://llm.server.example/v1")
    monkeypatch.setattr(settings, "embedding_api_key", SecretStr(SERVER_EMB_KEY))
    monkeypatch.setattr(settings, "embedding_model", "server-embedding-model")
    monkeypatch.setattr(settings, "embedding_base_url", "https://emb.server.example/v1")
    monkeypatch.setattr(settings, "token_encryption_key", SecretStr(Fernet.generate_key().decode()))
    return settings


class RecordingLLM:
    """Stands in for OpenAICompatibleLLM: records how it was built and answers with a fixed error."""

    built: list[tuple[str, str, str]] = []

    def __init__(self, api_key, model, base_url, **_):
        self.model = model
        RecordingLLM.built.append((api_key, model, base_url))

    def complete(self, messages, tools=None):
        raise LLMError("recording llm stops here")


class RecordingEmbeddings(FakeProvider):
    built: list[tuple[str, str, str]] = []

    def __init__(self, api_key, model, base_url, **_):
        super().__init__()
        self.model = model
        RecordingEmbeddings.built.append((api_key, model, base_url))


@pytest.fixture
def providers(monkeypatch):
    RecordingLLM.built = []
    RecordingEmbeddings.built = []
    monkeypatch.setattr(service, "OpenAICompatibleLLM", RecordingLLM)
    monkeypatch.setattr(service, "OpenAIEmbeddingProvider", RecordingEmbeddings)


def put(client, headers, **body):
    return client.put(URL, json=body, headers=headers)


def stored(factory, email="owner@example.com"):
    with factory() as session:
        user = session.scalar(select(User).where(User.email == email))
        return session.scalar(select(UserAISettings).where(UserAISettings.user_id == user.id))


@pytest.fixture
def owner(database):
    repository_id, headers = create_repository(database)
    return repository_id, headers


# ------------------------------------------------------------------ authentication and isolation


@pytest.mark.parametrize(
    "method, path",
    [("get", ""), ("put", ""), ("delete", "/llm-key"), ("delete", "/embedding-key")],
)
def test_every_endpoint_requires_authentication(client, method, path):
    kwargs = {"json": {"llm_model": "gpt"}} if method == "put" else {}
    assert getattr(client, method)(URL + path, **kwargs).status_code == 401


def test_a_fresh_user_sees_the_server_defaults(client, owner):
    _, headers = owner
    body = client.get(URL, headers=headers).json()
    assert body == {
        "llm": {"api_key_configured": False, "api_key_hint": None, "model": "server-llm-model", "source": "server"},
        "embedding": {"api_key_configured": False, "api_key_hint": None, "model": "server-embedding-model", "source": "server"},
    }


def test_source_is_none_when_neither_user_nor_server_has_a_key(client, owner, server_config, monkeypatch):
    monkeypatch.setattr(server_config, "llm_api_key", None)
    monkeypatch.setattr(server_config, "embedding_api_key", None)
    body = client.get(URL, headers=owner[1]).json()
    assert body["llm"]["source"] == "none" and body["embedding"]["source"] == "none"


def test_users_cannot_see_or_change_each_others_settings(client, database):
    _, alice = create_repository(database, email="alice@example.com")
    _, bob = create_repository(database, email="bob@example.com")

    put(client, alice, llm_api_key=USER_LLM_KEY, llm_model="alice-model")

    bob_view = client.get(URL, headers=bob).json()
    assert bob_view["llm"]["api_key_configured"] is False
    assert bob_view["llm"]["model"] == "server-llm-model"
    client.delete(URL + "/llm-key", headers=bob)
    assert client.get(URL, headers=alice).json()["llm"]["api_key_configured"] is True
    assert stored(database, "bob@example.com") is None


# ------------------------------------------------------------------ storage and secrecy


def test_keys_are_encrypted_at_rest_and_never_returned(client, database, owner):
    _, headers = owner
    response = put(client, headers, llm_api_key=USER_LLM_KEY, embedding_api_key=USER_EMB_KEY)

    assert response.status_code == 200
    for text in (response.text, client.get(URL, headers=headers).text):
        assert not any(key in text for key in ALL_KEYS)
    row = stored(database)
    assert row.llm_api_key_encrypted != USER_LLM_KEY and USER_LLM_KEY not in row.llm_api_key_encrypted
    assert decrypt_secret(row.llm_api_key_encrypted) == USER_LLM_KEY
    assert decrypt_secret(row.embedding_api_key_encrypted) == USER_EMB_KEY


def test_only_the_last_four_characters_are_shown(client, owner):
    body = put(client, owner[1], llm_api_key=USER_LLM_KEY, embedding_api_key=USER_EMB_KEY).json()
    assert body["llm"]["api_key_hint"] == "AAAA"
    assert body["embedding"]["api_key_hint"] == "BBBB"
    assert body["llm"]["api_key_configured"] is True and body["llm"]["source"] == "user"


def test_keys_are_never_logged(client, owner, caplog):
    with caplog.at_level(logging.DEBUG):
        put(client, owner[1], llm_api_key=USER_LLM_KEY, embedding_api_key=USER_EMB_KEY)
        client.get(URL, headers=owner[1])
        client.delete(URL + "/llm-key", headers=owner[1])
    assert not any(key in caplog.text for key in ALL_KEYS)


def test_validation_errors_do_not_echo_the_key(client, owner):
    bad = "sk short key with spaces"
    response = put(client, owner[1], llm_api_key=bad)
    assert response.status_code == 422
    assert bad not in response.text and "short key" not in response.text


def test_one_row_per_user(client, database, owner):
    put(client, owner[1], llm_api_key=USER_LLM_KEY)
    put(client, owner[1], embedding_api_key=USER_EMB_KEY)
    with database() as session:
        assert len(session.scalars(select(UserAISettings)).all()) == 1
    table = UserAISettings.__table__
    assert any(c.name == "user_id" for u in table.constraints if u.__class__.__name__ == "UniqueConstraint" for c in u.columns)
    assert next(iter(table.c.user_id.foreign_keys)).ondelete == "CASCADE"


# ------------------------------------------------------------------ independence and partial updates


def test_llm_and_embedding_are_independent(client, database, owner):
    _, headers = owner
    put(client, headers, llm_api_key=USER_LLM_KEY, llm_model="my-llm")
    body = client.get(URL, headers=headers).json()
    assert body["llm"]["source"] == "user" and body["llm"]["model"] == "my-llm"
    assert body["embedding"]["api_key_configured"] is False and body["embedding"]["source"] == "server"

    put(client, headers, embedding_api_key=USER_EMB_KEY, embedding_model="my-embed")
    row = stored(database)
    assert decrypt_secret(row.llm_api_key_encrypted) == USER_LLM_KEY
    assert row.llm_model == "my-llm" and row.embedding_model == "my-embed"


def test_the_same_key_can_be_used_on_both_sides(client, database, owner):
    put(client, owner[1], llm_api_key=USER_LLM_KEY, embedding_api_key=USER_LLM_KEY)
    row = stored(database)
    assert decrypt_secret(row.llm_api_key_encrypted) == decrypt_secret(row.embedding_api_key_encrypted) == USER_LLM_KEY


def test_different_keys_are_kept_separately(client, database, owner):
    put(client, owner[1], llm_api_key=USER_LLM_KEY, embedding_api_key=USER_EMB_KEY)
    row = stored(database)
    assert decrypt_secret(row.llm_api_key_encrypted) == USER_LLM_KEY
    assert decrypt_secret(row.embedding_api_key_encrypted) == USER_EMB_KEY


def test_a_blank_key_keeps_the_existing_key(client, database, owner):
    put(client, owner[1], llm_api_key=USER_LLM_KEY)
    body = put(client, owner[1], llm_api_key="", llm_model="other-model").json()
    assert decrypt_secret(stored(database).llm_api_key_encrypted) == USER_LLM_KEY
    assert body["llm"]["api_key_hint"] == "AAAA" and body["llm"]["model"] == "other-model"


def test_an_omitted_field_is_unchanged_and_a_new_key_replaces_the_old(client, database, owner):
    put(client, owner[1], llm_api_key=USER_LLM_KEY, llm_model="keep-me")
    put(client, owner[1], embedding_model="only-this")
    assert stored(database).llm_model == "keep-me"
    body = put(client, owner[1], llm_api_key="sk-replacement-key-ZZZZ").json()
    assert body["llm"]["api_key_hint"] == "ZZZZ"
    assert decrypt_secret(stored(database).llm_api_key_encrypted) == "sk-replacement-key-ZZZZ"


def test_a_blank_model_resets_to_the_server_default(client, owner):
    put(client, owner[1], llm_model="my-llm")
    body = put(client, owner[1], llm_model="").json()
    assert body["llm"]["model"] == "server-llm-model"


def test_deleting_a_key_removes_only_that_side(client, database, owner):
    put(client, owner[1], llm_api_key=USER_LLM_KEY, llm_model="my-llm", embedding_api_key=USER_EMB_KEY)

    body = client.delete(URL + "/llm-key", headers=owner[1]).json()

    assert body["llm"]["api_key_configured"] is False and body["llm"]["source"] == "server"
    assert body["llm"]["model"] == "my-llm"
    assert body["embedding"]["api_key_configured"] is True
    row = stored(database)
    assert row.llm_api_key_encrypted is None and row.llm_api_key_hint is None
    assert row.embedding_api_key_encrypted is not None

    body = client.delete(URL + "/embedding-key", headers=owner[1]).json()
    assert body["embedding"]["api_key_configured"] is False


def test_deleting_a_key_that_does_not_exist_is_harmless(client, owner):
    assert client.delete(URL + "/llm-key", headers=owner[1]).status_code == 200


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize("field", ["llm_api_key", "embedding_api_key"])
@pytest.mark.parametrize(
    "value",
    ["short", "a" * 513, "has space in it 123", "tab\tinside-key-123", "new\nline-key-1234", "nul\x00-key-12345"],
)
def test_bad_keys_are_rejected(client, owner, field, value):
    assert put(client, owner[1], **{field: value}).status_code == 422


def test_keys_at_the_length_limits_are_accepted(client, owner):
    assert put(client, owner[1], llm_api_key="k" * 8, embedding_api_key="k" * 512).status_code == 200


@pytest.mark.parametrize("field", ["llm_model", "embedding_model"])
@pytest.mark.parametrize("value", ["has space", "semi;colon", "m" * 129, "modèl", "a\nb", "$(x)"])
def test_bad_models_are_rejected(client, owner, field, value):
    assert put(client, owner[1], **{field: value}).status_code == 422


@pytest.mark.parametrize("value", ["gpt-4o", "text-embedding-3-small", "org/model:tag_1.5", "m" * 128])
def test_good_models_are_accepted(client, owner, value):
    assert put(client, owner[1], llm_model=value).json()["llm"]["model"] == value


@pytest.mark.parametrize(
    "field", ["llm_base_url", "embedding_base_url", "base_url", "user_id", "api_key"]
)
def test_unknown_fields_are_rejected(client, owner, field):
    assert put(client, owner[1], **{field: "https://evil.example"}).status_code == 422


def test_nothing_is_stored_when_validation_fails(client, database, owner):
    put(client, owner[1], llm_api_key=USER_LLM_KEY, embedding_model="bad model")
    assert stored(database) is None


# ------------------------------------------------------------------ encryption failure


def test_saving_a_key_fails_cleanly_when_encryption_is_not_configured(client, database, owner, server_config, monkeypatch, caplog):
    monkeypatch.setattr(server_config, "token_encryption_key", None)
    with caplog.at_level(logging.DEBUG):
        response = put(client, owner[1], llm_api_key=USER_LLM_KEY, llm_model="my-llm")
    assert response.status_code == 503
    assert USER_LLM_KEY not in response.text and USER_LLM_KEY not in caplog.text
    assert stored(database) is None  # nothing was partially written


def test_an_invalid_encryption_key_also_fails_cleanly(client, owner, server_config, monkeypatch):
    monkeypatch.setattr(server_config, "token_encryption_key", SecretStr("not-a-fernet-key"))
    assert put(client, owner[1], llm_api_key=USER_LLM_KEY).status_code == 503


def test_models_can_still_be_saved_and_settings_read_without_an_encryption_key(client, owner, server_config, monkeypatch):
    monkeypatch.setattr(server_config, "token_encryption_key", None)
    assert put(client, owner[1], llm_model="my-llm").status_code == 200
    assert client.get(URL, headers=owner[1]).json()["llm"]["model"] == "my-llm"


# ------------------------------------------------------------------ resolving providers


def resolve_llm(database, email="owner@example.com", fallback=None):
    with database() as session:
        user = session.scalar(select(User).where(User.email == email))
        return service.llm_provider_for(session, user, fallback or (lambda: "server-llm"))


def resolve_embeddings(database, email="owner@example.com", fallback=None):
    with database() as session:
        user = session.scalar(select(User).where(User.email == email))
        return service.embedding_factory_for(session, user, fallback or (lambda: "server-embeddings"))


def test_without_user_settings_the_server_configuration_is_used(database, owner, providers):
    assert resolve_llm(database) == "server-llm"
    assert resolve_embeddings(database)() == "server-embeddings"
    assert RecordingLLM.built == [] and RecordingEmbeddings.built == []


def test_the_users_settings_override_the_server_configuration(client, database, owner, providers):
    put(client, owner[1], llm_api_key=USER_LLM_KEY, llm_model="my-llm", embedding_api_key=USER_EMB_KEY, embedding_model="my-embed")

    resolve_llm(database)
    resolve_embeddings(database)()

    assert RecordingLLM.built == [(USER_LLM_KEY, "my-llm", "https://llm.server.example/v1")]
    assert RecordingEmbeddings.built == [(USER_EMB_KEY, "my-embed", "https://emb.server.example/v1")]


def test_a_user_key_without_a_model_uses_the_server_model(client, database, owner, providers):
    put(client, owner[1], llm_api_key=USER_LLM_KEY)
    resolve_llm(database)
    assert RecordingLLM.built == [(USER_LLM_KEY, "server-llm-model", "https://llm.server.example/v1")]


def test_a_model_without_a_key_uses_the_server_key_of_the_same_side(client, database, owner, providers):
    put(client, owner[1], llm_model="my-llm", embedding_model="my-embed", embedding_api_key=USER_EMB_KEY)
    resolve_llm(database)
    resolve_embeddings(database)()
    assert RecordingLLM.built[0][:2] == (SERVER_LLM_KEY, "my-llm")
    assert RecordingEmbeddings.built[0][:2] == (USER_EMB_KEY, "my-embed")


def test_the_llm_key_is_never_used_for_embeddings_and_the_reverse(client, database, owner, server_config, monkeypatch, providers):
    monkeypatch.setattr(server_config, "llm_api_key", None)
    monkeypatch.setattr(server_config, "embedding_api_key", None)
    put(client, owner[1], llm_api_key=USER_LLM_KEY)

    assert resolve_llm(database).model == "server-llm-model"
    with pytest.raises(EmbeddingNotConfiguredError):
        resolve_embeddings(database, fallback=lambda: (_ for _ in ()).throw(EmbeddingNotConfiguredError("x")))()
    put(client, owner[1], embedding_model="my-embed")
    with pytest.raises(EmbeddingNotConfiguredError):
        resolve_embeddings(database)()
    assert all(call[0] != USER_LLM_KEY for call in RecordingEmbeddings.built)

    client.delete(URL + "/llm-key", headers=owner[1])
    put(client, owner[1], embedding_api_key=USER_EMB_KEY, llm_model="my-llm")
    with pytest.raises(LLMNotConfiguredError):
        resolve_llm(database)
    assert all(call[0] != USER_EMB_KEY for call in RecordingLLM.built)


def test_a_model_only_llm_setting_without_any_key_is_not_configured(client, database, owner, server_config, monkeypatch, providers):
    monkeypatch.setattr(server_config, "llm_api_key", None)
    put(client, owner[1], llm_model="my-llm")
    with pytest.raises(LLMNotConfiguredError):
        resolve_llm(database)


def test_a_stored_key_that_cannot_be_decrypted_counts_as_not_configured(client, database, owner, server_config, monkeypatch, providers):
    put(client, owner[1], llm_api_key=USER_LLM_KEY, embedding_api_key=USER_EMB_KEY)
    monkeypatch.setattr(server_config, "token_encryption_key", SecretStr(Fernet.generate_key().decode()))

    with pytest.raises(LLMNotConfiguredError):
        resolve_llm(database)
    with pytest.raises(EmbeddingNotConfiguredError):
        resolve_embeddings(database)()
    assert client.get(URL, headers=owner[1]).status_code == 200


def test_one_users_settings_never_apply_to_another_user(client, database, providers):
    _, alice = create_repository(database, email="alice@example.com")
    create_repository(database, email="bob@example.com")
    put(client, alice, llm_api_key=USER_LLM_KEY, llm_model="alice-model")

    assert resolve_llm(database, "bob@example.com") == "server-llm"
    assert RecordingLLM.built == []


# ------------------------------------------------------------------ the routes use the right configuration


def ask(client, repository_id, headers, suffix="agent", **body):
    return client.post(f"/api/v1/repositories/{repository_id}/{suffix}", json={"message": "hello", **body}, headers=headers)


@pytest.mark.parametrize("suffix", ["agent", "agent/plan", "agent/execute"])
def test_the_agent_routes_use_the_users_llm_configuration(client, owner, providers, suffix):
    repository_id, headers = owner
    put(client, headers, llm_api_key=USER_LLM_KEY, llm_model="my-llm", embedding_api_key=USER_EMB_KEY)

    if suffix.endswith("execute"):
        body = {"message": EDIT_REQUEST, "plan": edit_plan(), "approved": True}
        client.post(f"/api/v1/repositories/{repository_id}/{suffix}", json=body, headers=headers)
    else:
        ask(client, repository_id, headers, suffix)

    assert RecordingLLM.built
    assert RecordingLLM.built[0][:2] == (USER_LLM_KEY, "my-llm")
    assert all(call[0] != USER_EMB_KEY for call in RecordingLLM.built)


@pytest.mark.parametrize(
    "suffix, body",
    [("agent/pr/1/review", {"approved": True}), ("agent/pr/1/fix-plan", {"finding": {**FINDING, "head_sha": "a" * 40, "signature": "b" * 64}})],
)
def test_review_and_fix_routes_use_the_users_llm_configuration(client, owner, providers, suffix, body):
    repository_id, headers = owner
    put(client, headers, llm_api_key=USER_LLM_KEY, llm_model="my-llm", embedding_api_key=USER_EMB_KEY)

    client.post(f"/api/v1/repositories/{repository_id}/{suffix}", json=body, headers=headers)

    assert RecordingLLM.built and RecordingLLM.built[0][:2] == (USER_LLM_KEY, "my-llm")


def test_the_agent_route_falls_back_to_the_server_provider(client, owner, monkeypatch):
    repository_id, headers = owner
    calls = []

    def server_provider():
        calls.append("server")
        raise LLMNotConfiguredError("LLM provider is not configured")

    monkeypatch.setattr(repository_routes, "get_llm_provider", server_provider)
    response = ask(client, repository_id, headers)
    assert calls == ["server"] and response.status_code == 503


def test_a_missing_llm_key_keeps_the_existing_503(client, owner, server_config, monkeypatch):
    monkeypatch.setattr(server_config, "llm_api_key", None)
    response = ask(client, *owner)
    assert response.status_code == 503


def test_scan_indexes_with_the_users_embedding_configuration(client, database, owner, providers, monkeypatch):
    repository_id, headers = owner
    use_github(monkeypatch, FakeGitHubClient({"app/a.py": "def a():\n    return 1\n"}))
    put(client, headers, embedding_api_key=USER_EMB_KEY, embedding_model="my-embed", llm_api_key=USER_LLM_KEY)

    response = scan(client, repository_id, headers)

    assert response.status_code == 200
    assert RecordingEmbeddings.built
    assert RecordingEmbeddings.built[0][:2] == (USER_EMB_KEY, "my-embed")
    assert all(call[0] != USER_LLM_KEY for call in RecordingEmbeddings.built)


def test_scan_reports_embeddings_not_configured_when_only_an_llm_key_exists(client, owner, server_config, monkeypatch, providers):
    repository_id, headers = owner
    monkeypatch.setattr(server_config, "embedding_api_key", None)
    use_github(monkeypatch, FakeGitHubClient({"app/a.py": "def a():\n    return 1\n"}))
    put(client, headers, llm_api_key=USER_LLM_KEY, embedding_model="my-embed")

    response = scan(client, repository_id, headers)

    assert response.status_code == 200
    assert response.json()["embeddings"]["status"] == "not_configured"
    assert RecordingEmbeddings.built == []


def test_scan_still_uses_the_server_embedding_provider_without_user_settings(client, owner, monkeypatch):
    repository_id, headers = owner
    use_github(monkeypatch, FakeGitHubClient({"app/a.py": "def a():\n    return 1\n"}))
    calls = []

    def server_provider():
        calls.append("server")
        raise EmbeddingNotConfiguredError("Embedding provider is not configured")

    monkeypatch.setattr(repository_routes, "get_embedding_provider", server_provider)
    assert scan(client, repository_id, headers).status_code == 200
    assert calls == ["server"]
