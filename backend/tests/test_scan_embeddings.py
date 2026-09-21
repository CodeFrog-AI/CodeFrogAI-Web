"""Tests for automatic embedding generation after a repository scan."""

import logging

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes import repositories as repository_routes
from app.db.base import Base
from app.db.database import get_db
from app.db.models import RepositoryChunk, RepositoryFile
from app.embeddings import EMBEDDING_DIMENSIONS
from app.embeddings.indexing import BATCH_SIZE
from app.embeddings.provider import EmbeddingError, EmbeddingNotConfiguredError
from app.scanner import semantic as semantic_module
from main import app
from tests.test_repository_scan import FakeGitHubClient, create_repository, scan, use_github
from tests.test_semantic_search import FakeProvider, python_similarity

API_KEY = "sk-test-embedding-key-that-must-never-leak"

AUTH_CODE = "def github_callback():\n    exchange oauth token for login\n"
DB_CODE = "engine = create_engine()\nsession query connection\n"
UI_CODE = "render button with css layout\n"
FILES = {"app/github_oauth.py": AUTH_CODE, "app/database.py": DB_CODE, "app/ui.py": UI_CODE}


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


@pytest.fixture
def embeddings(monkeypatch):
    """Configured fake embedding provider plus the SQLite stand-in for pgvector ranking."""

    fake = FakeProvider()
    monkeypatch.setattr(repository_routes, "get_embedding_provider", lambda: fake)
    monkeypatch.setattr(semantic_module, "find_similar_chunks", python_similarity)
    return fake


def chunks(factory):
    with factory() as session:
        return session.scalars(
            select(RepositoryChunk).join(RepositoryFile).order_by(RepositoryFile.path, RepositoryChunk.chunk_index)
        ).all()


def semantic_paths(client, repository_id, headers, query="github authentication login"):
    response = client.get(
        f"/api/v1/repositories/{repository_id}/semantic-search",
        params={"query": query, "limit": 25}, headers=headers,
    )
    assert response.status_code == 200
    return [result["file_path"] for result in response.json()["results"]]


def test_scan_automatically_generates_embeddings_for_new_chunks(client, database, monkeypatch, embeddings):
    repository_id, headers = create_repository(database)
    use_github(monkeypatch, FakeGitHubClient(FILES))

    response = scan(client, repository_id, headers)

    assert response.status_code == 200
    assert response.json()["chunks_created"] == 3
    assert response.json()["embeddings"] == {
        "status": "completed", "chunks_embedded": 3, "chunks_reused": 0, "chunks_skipped": 0, "message": None,
    }
    stored = chunks(database)
    assert len(stored) == 3
    for chunk in stored:
        assert len(chunk.embedding) == EMBEDDING_DIMENSIONS
        assert len(chunk.embedding_content_hash) == 64


def test_semantic_search_works_immediately_after_scan_without_manual_embedding(
    client, database, monkeypatch, embeddings
):
    repository_id, headers = create_repository(database)
    use_github(monkeypatch, FakeGitHubClient(FILES))
    scan(client, repository_id, headers)

    paths = semantic_paths(client, repository_id, headers)

    assert paths[0] == "app/github_oauth.py"


def test_unchanged_chunks_are_reused_without_calling_the_provider(client, database, monkeypatch, embeddings):
    repository_id, headers = create_repository(database)
    use_github(monkeypatch, FakeGitHubClient(FILES))
    scan(client, repository_id, headers)
    before = {c.id: (c.embedding_content_hash, list(c.embedding)) for c in chunks(database)}
    calls_before = len(embeddings.calls)

    response = scan(client, repository_id, headers)

    assert response.json()["chunks_created"] == 0
    assert response.json()["embeddings"]["chunks_embedded"] == 0
    assert response.json()["embeddings"]["chunks_reused"] == 3
    assert len(embeddings.calls) == calls_before
    assert {c.id: (c.embedding_content_hash, list(c.embedding)) for c in chunks(database)} == before


def test_changed_file_gets_new_embeddings_and_untouched_files_do_not(client, database, monkeypatch, embeddings):
    repository_id, headers = create_repository(database)
    use_github(monkeypatch, FakeGitHubClient(FILES))
    scan(client, repository_id, headers)
    embeddings.calls.clear()

    changed = dict(FILES, **{"app/ui.py": "def github_login():\n    oauth token\n"})
    use_github(monkeypatch, FakeGitHubClient(changed))
    response = scan(client, repository_id, headers)

    assert response.json()["embeddings"]["chunks_embedded"] == 1
    assert response.json()["embeddings"]["chunks_reused"] == 2
    assert len(embeddings.calls) == 1 and len(embeddings.calls[0]) == 1
    assert "def github_login" in embeddings.calls[0][0]
    assert "app/ui.py" in semantic_paths(client, repository_id, headers)[:2]


def test_chunk_with_missing_embedding_is_regenerated_on_rescan(client, database, monkeypatch, embeddings):
    repository_id, headers = create_repository(database)
    use_github(monkeypatch, FakeGitHubClient(FILES))
    scan(client, repository_id, headers)
    with database() as session:
        victim = session.scalars(select(RepositoryChunk)).first()
        victim.embedding = None
        session.commit()
    embeddings.calls.clear()

    response = scan(client, repository_id, headers)

    assert response.json()["embeddings"]["chunks_embedded"] == 1
    assert len(embeddings.calls) == 1
    assert all(chunk.embedding is not None for chunk in chunks(database))


def test_changing_the_embedding_model_re_embeds_on_rescan(client, database, monkeypatch, embeddings):
    repository_id, headers = create_repository(database)
    use_github(monkeypatch, FakeGitHubClient(FILES))
    scan(client, repository_id, headers)
    embeddings.model = "another-model"

    response = scan(client, repository_id, headers)

    assert response.json()["embeddings"]["chunks_embedded"] == 3


def test_deleted_files_and_their_embeddings_are_gone_and_not_searchable(client, database, monkeypatch, embeddings):
    repository_id, headers = create_repository(database)
    use_github(monkeypatch, FakeGitHubClient(FILES))
    scan(client, repository_id, headers)
    assert "app/github_oauth.py" in semantic_paths(client, repository_id, headers)

    remaining = {path: code for path, code in FILES.items() if path != "app/github_oauth.py"}
    use_github(monkeypatch, FakeGitHubClient(remaining))
    response = scan(client, repository_id, headers)

    assert response.json()["files_removed"] == 1
    assert "app/github_oauth.py" not in semantic_paths(client, repository_id, headers)
    assert len(chunks(database)) == 2
    with database() as session:
        embedded = session.scalar(select(func.count()).select_from(RepositoryChunk).where(RepositoryChunk.embedding.is_not(None)))
        assert embedded == 2


def test_many_chunks_are_embedded_in_batches_not_one_request_per_chunk(client, database, monkeypatch, embeddings):
    repository_id, headers = create_repository(database)
    many = {f"pkg/file_{n:03d}.py": f"github token {n}\n" for n in range(BATCH_SIZE + 6)}
    use_github(monkeypatch, FakeGitHubClient(many))

    response = scan(client, repository_id, headers)

    assert response.json()["embeddings"]["chunks_embedded"] == BATCH_SIZE + 6
    assert [len(call) for call in embeddings.calls] == [BATCH_SIZE, 6]


def test_provider_failure_keeps_the_scan_and_reports_failure_safely(client, database, monkeypatch, caplog):
    repository_id, headers = create_repository(database)
    use_github(monkeypatch, FakeGitHubClient(FILES))

    class FailingProvider:
        model = "fake"

        def embed(self, texts):
            raise EmbeddingError(f"upstream 401 for {API_KEY}: raw provider body")

    monkeypatch.setattr(repository_routes, "get_embedding_provider", lambda: FailingProvider())
    caplog.set_level(logging.DEBUG)

    response = scan(client, repository_id, headers)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed" and body["files_indexed"] == 3 and body["chunks_created"] == 3
    assert body["embeddings"]["status"] == "failed"
    assert body["embeddings"]["chunks_embedded"] is None
    assert API_KEY not in response.text and "raw provider body" not in response.text
    assert API_KEY not in caplog.text and "raw provider body" not in caplog.text
    assert len(chunks(database)) == 3 and all(c.embedding is None for c in chunks(database))


def test_failed_embedding_is_recovered_by_the_next_scan(client, database, monkeypatch, embeddings):
    repository_id, headers = create_repository(database)
    use_github(monkeypatch, FakeGitHubClient(FILES))

    class DownProvider:
        model = "fake"

        def embed(self, texts):
            raise EmbeddingError("down")

    healthy = embeddings
    monkeypatch.setattr(repository_routes, "get_embedding_provider", lambda: DownProvider())
    assert scan(client, repository_id, headers).json()["embeddings"]["status"] == "failed"

    monkeypatch.setattr(repository_routes, "get_embedding_provider", lambda: healthy)
    response = scan(client, repository_id, headers)

    assert response.json()["files_indexed"] == 3 and response.json()["chunks_created"] == 0
    assert response.json()["embeddings"]["status"] == "completed"
    assert response.json()["embeddings"]["chunks_embedded"] == 3
    assert "app/github_oauth.py" in semantic_paths(client, repository_id, headers)


def test_later_batch_failure_keeps_earlier_batches_and_resumes(client, database, monkeypatch, embeddings):
    repository_id, headers = create_repository(database)
    many = {f"pkg/file_{n:03d}.py": f"github token {n}\n" for n in range(BATCH_SIZE + 6)}
    use_github(monkeypatch, FakeGitHubClient(many))

    class SecondBatchFails(FakeProvider):
        def embed(self, texts):
            if self.calls:
                raise EmbeddingError("second batch failed")
            return super().embed(texts)

    monkeypatch.setattr(repository_routes, "get_embedding_provider", lambda: SecondBatchFails())
    failed = scan(client, repository_id, headers)

    assert failed.json()["embeddings"]["status"] == "failed"
    stored = chunks(database)
    assert sum(c.embedding is not None for c in stored) == BATCH_SIZE
    assert all((c.embedding is None) == (c.embedding_content_hash is None) for c in stored)

    monkeypatch.setattr(repository_routes, "get_embedding_provider", lambda: embeddings)
    resumed = scan(client, repository_id, headers)

    assert resumed.json()["embeddings"]["chunks_embedded"] == 6
    assert resumed.json()["embeddings"]["chunks_reused"] == BATCH_SIZE
    assert [len(call) for call in embeddings.calls] == [6]


def test_missing_embedding_configuration_keeps_the_scan_and_says_so(client, database, monkeypatch):
    repository_id, headers = create_repository(database)
    use_github(monkeypatch, FakeGitHubClient(FILES))

    def unconfigured():
        raise EmbeddingNotConfiguredError("EMBEDDING_API_KEY missing")

    monkeypatch.setattr(repository_routes, "get_embedding_provider", unconfigured)

    response = scan(client, repository_id, headers)

    assert response.status_code == 200
    assert response.json()["chunks_created"] == 3
    assert response.json()["embeddings"]["status"] == "not_configured"
    assert "EMBEDDING_API_KEY" not in response.text
    assert len(chunks(database)) == 3


def test_github_failure_does_not_trigger_embedding_generation(client, database, monkeypatch, embeddings):
    from app.integrations.github.contents import GitHubContentError

    repository_id, headers = create_repository(database)
    use_github(monkeypatch, FakeGitHubClient(tree_error=GitHubContentError("boom")))

    response = scan(client, repository_id, headers)

    assert response.status_code == 502
    assert embeddings.calls == []


def test_scan_ownership_still_blocks_embedding_of_another_users_repository(client, database, monkeypatch, embeddings):
    other_id, _ = create_repository(database, email="other@example.com")
    _, my_headers = create_repository(database, email="me@example.com")
    use_github(monkeypatch, FakeGitHubClient(FILES))

    assert scan(client, other_id, my_headers).status_code == 404
    assert scan(client, other_id, {}).status_code == 401
    assert embeddings.calls == []
    assert chunks(database) == []


def test_manual_embeddings_endpoint_still_works_and_reuses_scan_embeddings(client, database, monkeypatch, embeddings):
    repository_id, headers = create_repository(database)
    use_github(monkeypatch, FakeGitHubClient(FILES))
    scan(client, repository_id, headers)
    calls_after_scan = len(embeddings.calls)

    reused = client.post(f"/api/v1/repositories/{repository_id}/embeddings", headers=headers)

    assert reused.status_code == 200
    assert (reused.json()["chunks_embedded"], reused.json()["chunks_reused"]) == (0, 3)
    assert len(embeddings.calls) == calls_after_scan

    with database() as session:
        session.query(RepositoryChunk).update({RepositoryChunk.embedding: None, RepositoryChunk.embedding_content_hash: None})
        session.commit()
    backfilled = client.post(f"/api/v1/repositories/{repository_id}/embeddings", headers=headers)
    assert backfilled.json()["chunks_embedded"] == 3


def test_keys_tokens_and_vectors_never_appear_in_scan_responses_or_logs(client, database, monkeypatch, caplog):
    from cryptography.fernet import Fernet
    from app.core.config import get_settings

    github_token = "gho_plaintext_github_token_that_must_never_leak"
    key = get_settings().token_encryption_key.get_secret_value().encode()
    encrypted = Fernet(key).encrypt(github_token.encode()).decode()
    repository_id, headers = create_repository(database, encrypted_token=encrypted)
    use_github(monkeypatch, FakeGitHubClient(FILES))
    leaky = FakeProvider()
    original = leaky.embed

    def embed(texts):
        original(texts)
        raise EmbeddingError(f"provider said key {API_KEY} and token {github_token}")

    leaky.embed = embed
    monkeypatch.setattr(repository_routes, "get_embedding_provider", lambda: leaky)
    caplog.set_level(logging.DEBUG)

    failed = scan(client, repository_id, headers)
    monkeypatch.setattr(repository_routes, "get_embedding_provider", lambda: FakeProvider())
    monkeypatch.setattr(semantic_module, "find_similar_chunks", python_similarity)
    ok = scan(client, repository_id, headers)

    for response in (failed, ok):
        assert response.status_code == 200
        for secret in (API_KEY, github_token, encrypted, '"embedding"', "access_token"):
            assert secret not in response.text
    assert API_KEY not in caplog.text and github_token not in caplog.text
