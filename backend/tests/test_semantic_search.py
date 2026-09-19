"""Tests for embedding generation, indexing, and the semantic search endpoint.

The suite runs on SQLite, which cannot evaluate pgvector distance operators, so the
ranking query is replaced by an equivalent Python cosine ranking here. The real SQL
is covered by test_semantic_search_postgres.py.
"""

import logging
import math
import uuid
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes import repositories as repository_routes
from app.auth.security import create_access_token
from app.db.base import Base
from app.db.database import get_db
from app.db.models import GitHubAccount, Repository, RepositoryChunk, RepositoryFile, User
from app.embeddings import EMBEDDING_DIMENSIONS
from app.embeddings import provider as provider_module
from app.embeddings.indexing import BATCH_SIZE, build_embedding_text, content_hash, index_repository_embeddings
from app.embeddings.provider import (
    EmbeddingError,
    EmbeddingNotConfiguredError,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)
from app.scanner import semantic as semantic_module
from app.scanner.semantic import MAX_SEMANTIC_LIMIT, SemanticHit
from main import app

API_KEY = "sk-test-embedding-key-that-must-never-leak"
CONCEPTS = [
    {"github", "oauth", "auth", "authentication", "login", "token", "callback", "credentials"},
    {"database", "sql", "query", "session", "engine", "connection"},
    {"button", "css", "render", "layout", "ui"},
]


def concept_vector(text: str) -> list[float]:
    """Deterministic stand-in embedding: one axis per concept, a spare axis for anything else."""

    words = set(text.lower().replace("_", " ").replace("(", " ").replace(")", " ").replace("?", " ").split())
    vector = [float(len(words & concept)) for concept in CONCEPTS]
    vector.append(0.0 if any(vector) else 1.0)
    vector += [0.0] * (EMBEDDING_DIMENSIONS - len(vector))
    return vector


class FakeProvider:
    model = "fake-embedding-model"

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [concept_vector(text) for text in texts]


def python_similarity(session, repository, query_embedding, limit, min_score=None):
    """SQLite stand-in for the pgvector cosine ranking."""

    rows = (
        session.query(RepositoryFile, RepositoryChunk)
        .join(RepositoryChunk, RepositoryChunk.repository_file_id == RepositoryFile.id)
        .filter(RepositoryFile.repository_id == repository.id, RepositoryChunk.embedding.is_not(None))
        .all()
    )

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

    hits = [
        SemanticHit(file.path, file.language, chunk.start_line, chunk.end_line, chunk.content,
                    round(cosine(list(chunk.embedding), query_embedding), 4))
        for file, chunk in rows
    ]
    hits = [hit for hit in hits if min_score is None or hit.score >= min_score]
    return sorted(hits, key=lambda hit: (-hit.score, hit.file_path))[:limit]


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
def provider(monkeypatch):
    fake = FakeProvider()
    monkeypatch.setattr(repository_routes, "get_embedding_provider", lambda: fake)
    monkeypatch.setattr(semantic_module, "find_similar_chunks", python_similarity)
    return fake


DEFAULT_FILES = {
    "app/github_oauth.py": [(1, "def github_callback():\n    exchange oauth token for login\n")],
    "app/database.py": [(1, "engine = create_engine()\nsession query connection\n")],
    "app/ui.py": [(1, "render button with css layout\n")],
}


def create_repository(factory, email="me@example.com", files=None):
    with factory() as session:
        user = User(email=email, name="Owner", status="active")
        session.add(user)
        session.flush()
        account = GitHubAccount(user_id=user.id, github_user_id=abs(hash(email)) % 10**9, login=email.split("@")[0])
        session.add(account)
        session.flush()
        repository = Repository(
            github_account_id=account.id, github_repository_id=abs(hash(email + "r")) % 10**9,
            owner=account.login, name="project",
        )
        session.add(repository)
        session.flush()
        for path, chunks in (DEFAULT_FILES if files is None else files).items():
            file = RepositoryFile(repository_id=repository.id, path=path, language="python", sha=uuid.uuid4().hex)
            session.add(file)
            session.flush()
            for index, (start_line, content) in enumerate(chunks):
                session.add(RepositoryChunk(
                    repository_file_id=file.id, chunk_index=index, start_line=start_line,
                    end_line=start_line + content.count("\n") - 1 if content.endswith("\n") else start_line,
                    content=content,
                ))
        session.commit()
        return repository.id, {"Authorization": f"Bearer {create_access_token(user.id)}"}


def index(client, repository_id, headers):
    return client.post(f"/api/v1/repositories/{repository_id}/embeddings", headers=headers)


def search(client, repository_id, headers, **params):
    return client.get(f"/api/v1/repositories/{repository_id}/semantic-search", params=params, headers=headers)


def chunk_rows(factory):
    with factory() as session:
        return session.scalars(select(RepositoryChunk).order_by(RepositoryChunk.start_line)).all()


def indexed_repository(client, database, provider, **kwargs):
    repository_id, headers = create_repository(database, **kwargs)
    assert index(client, repository_id, headers).status_code == 200
    return repository_id, headers


# ---------------------------------------------------------------- semantic search API


def test_semantic_search_returns_relevant_chunk_with_all_fields(client, database, provider):
    repository_id, headers = indexed_repository(client, database, provider)

    response = search(client, repository_id, headers, query="Where is GitHub authentication handled?")

    assert response.status_code == 200
    body = response.json()
    assert body["repository_id"] == str(repository_id)
    assert body["query"] == "Where is GitHub authentication handled?"
    top = body["results"][0]
    assert top["file_path"] == "app/github_oauth.py"
    assert top["language"] == "python"
    assert (top["start_line"], top["end_line"]) == (1, 2)
    assert "github_callback" in top["snippet"]
    assert 0.5 < top["score"] <= 1.0
    scores = [result["score"] for result in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert set(top) == {"file_path", "language", "start_line", "end_line", "snippet", "score"}


def test_result_limit_is_enforced(client, database, provider):
    repository_id, headers = indexed_repository(client, database, provider)

    results = search(client, repository_id, headers, query="github token", limit=2).json()["results"]

    assert len(results) == 2


@pytest.mark.parametrize("limit", [0, -1, MAX_SEMANTIC_LIMIT + 1, "many"])
def test_invalid_limit_is_rejected(client, database, provider, limit):
    repository_id, headers = indexed_repository(client, database, provider)

    assert search(client, repository_id, headers, query="x", limit=limit).status_code == 422


def test_min_score_filters_to_an_empty_successful_result(client, database, provider):
    repository_id, headers = indexed_repository(client, database, provider)

    response = search(client, repository_id, headers, query="weather forecast", min_score=0.5)

    assert response.status_code == 200
    assert response.json()["results"] == []


@pytest.mark.parametrize("query", ["", "   ", "\t\n", "x" * 501])
def test_empty_whitespace_or_oversized_query_is_rejected(client, database, provider, query):
    repository_id, headers = indexed_repository(client, database, provider)
    calls_before = len(provider.calls)

    response = search(client, repository_id, headers, query=query)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert len(provider.calls) == calls_before


def test_authentication_is_required(client, database, provider):
    repository_id, _ = create_repository(database)

    assert search(client, repository_id, {}, query="x").status_code == 401
    assert search(client, repository_id, {"Authorization": "Bearer bad"}, query="x").status_code == 401
    assert index(client, repository_id, {}).status_code == 401
    assert provider.calls == []


def test_unknown_repository_returns_not_found(client, database, provider):
    _, headers = create_repository(database)

    assert search(client, uuid.uuid4(), headers, query="x").status_code == 404
    assert index(client, uuid.uuid4(), headers).status_code == 404


def test_another_users_repository_is_not_searchable_or_indexable(client, database, provider):
    other_id, _ = indexed_repository(client, database, provider, email="other@example.com")
    _, my_headers = create_repository(database, "me@example.com", files={})
    calls_before = len(provider.calls)

    searched = search(client, other_id, my_headers, query="github authentication")
    indexed = index(client, other_id, my_headers)

    for response in (searched, indexed):
        assert response.status_code == 404
        assert "github_oauth" not in response.text
    assert len(provider.calls) == calls_before


def test_search_never_returns_another_repositorys_chunks(client, database, provider):
    _, headers = indexed_repository(client, database, provider, files={"mine.py": [(1, "github login\n")]})
    indexed_repository(client, database, provider, email="other@example.com", files={"theirs.py": [(1, "github login\n")]})
    with database() as session:
        my_repo = session.scalars(select(Repository).where(Repository.name == "project").order_by(Repository.created_at)).first()

    results = search(client, my_repo.id, headers, query="github login").json()["results"]

    assert {r["file_path"] for r in results} == {"mine.py"}


def test_repository_without_embeddings_returns_clear_conflict(client, database, provider):
    repository_id, headers = create_repository(database)

    response = search(client, repository_id, headers, query="github")

    assert response.status_code == 409
    assert "embeddings" in response.json()["error"]["message"].lower()
    assert provider.calls == []


def test_embedding_provider_failure_is_handled_safely(client, database, monkeypatch, caplog):
    repository_id, headers = create_repository(database)
    with database() as session:
        chunk = session.scalars(select(RepositoryChunk)).first()
        chunk.embedding = concept_vector("seed")
        session.commit()

    class FailingProvider:
        model = "fake"

        def embed(self, texts):
            raise EmbeddingError(f"boom {API_KEY} raw provider body")

    monkeypatch.setattr(repository_routes, "get_embedding_provider", lambda: FailingProvider())
    caplog.set_level(logging.DEBUG)

    searched = search(client, repository_id, headers, query="github")
    indexed = index(client, repository_id, headers)

    for response in (searched, indexed):
        assert response.status_code == 502
        assert API_KEY not in response.text and "raw provider body" not in response.text
    assert API_KEY not in caplog.text


def test_unconfigured_provider_returns_service_unavailable(client, database, monkeypatch):
    repository_id, headers = create_repository(database)

    def unconfigured():
        raise EmbeddingNotConfiguredError("no key")

    monkeypatch.setattr(repository_routes, "get_embedding_provider", unconfigured)

    assert index(client, repository_id, headers).status_code == 503
    assert search(client, repository_id, headers, query="x").status_code == 503


def test_keys_vectors_and_tokens_are_never_returned_or_logged(client, database, monkeypatch, caplog):
    repository_id, headers = create_repository(database)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
        return httpx.Response(500, text=f"upstream error echoing {API_KEY}")

    real = OpenAIEmbeddingProvider(API_KEY, "text-embedding-3-small", "https://api.example.test/v1",
                                   transport=httpx.MockTransport(handler))
    monkeypatch.setattr(repository_routes, "get_embedding_provider", lambda: real)
    caplog.set_level(logging.DEBUG)

    failed = index(client, repository_id, headers)

    assert failed.status_code == 502
    assert API_KEY not in failed.text and API_KEY not in caplog.text

    monkeypatch.setattr(repository_routes, "get_embedding_provider", lambda: FakeProvider())
    monkeypatch.setattr(semantic_module, "find_similar_chunks", python_similarity)
    ok_index = index(client, repository_id, headers)
    ok_search = search(client, repository_id, headers, query="github")
    for response in (ok_index, ok_search):
        assert "embedding\"" not in response.text and "access_token" not in response.text
        assert API_KEY not in response.text
    assert API_KEY not in caplog.text


# ---------------------------------------------------------------- indexing service


def test_indexing_stores_embeddings_and_reports_counts(client, database, provider):
    repository_id, headers = create_repository(database)

    response = index(client, repository_id, headers)

    assert response.json() == {
        "repository_id": str(repository_id), "status": "completed",
        "chunks_total": 3, "chunks_embedded": 3, "chunks_reused": 0, "chunks_skipped": 0,
    }
    for chunk in chunk_rows(database):
        assert chunk.embedding is not None and len(chunk.embedding) == EMBEDDING_DIMENSIONS
        assert len(chunk.embedding_content_hash) == 64


def test_reindexing_unchanged_chunks_reuses_embeddings_without_calling_provider(client, database, provider):
    repository_id, headers = indexed_repository(client, database, provider)
    calls_after_first = len(provider.calls)

    response = index(client, repository_id, headers)

    assert (response.json()["chunks_embedded"], response.json()["chunks_reused"]) == (0, 3)
    assert len(provider.calls) == calls_after_first


def test_changed_chunk_content_is_re_embedded(client, database, provider):
    repository_id, headers = indexed_repository(client, database, provider)
    with database() as session:
        chunk = session.scalars(select(RepositoryChunk).join(RepositoryFile).where(RepositoryFile.path == "app/ui.py")).one()
        old_hash = chunk.embedding_content_hash
        chunk.content = "def github_login():\n    oauth token\n"
        session.commit()
    provider.calls.clear()

    response = index(client, repository_id, headers)

    assert (response.json()["chunks_embedded"], response.json()["chunks_reused"]) == (1, 2)
    assert len(provider.calls) == 1 and len(provider.calls[0]) == 1
    with database() as session:
        chunk = session.scalars(select(RepositoryChunk).join(RepositoryFile).where(RepositoryFile.path == "app/ui.py")).one()
        assert chunk.embedding_content_hash != old_hash


def test_changing_the_embedding_model_re_embeds_everything(client, database, provider):
    repository_id, headers = indexed_repository(client, database, provider)
    provider.model = "another-model"
    provider.calls.clear()

    response = index(client, repository_id, headers)

    assert response.json()["chunks_embedded"] == 3


def test_blank_chunks_are_never_embedded(client, database, provider):
    repository_id, headers = create_repository(database, files={"blank.py": [(1, "   \n\n"), (3, "github\n")]})

    response = index(client, repository_id, headers)

    assert (response.json()["chunks_embedded"], response.json()["chunks_skipped"]) == (1, 1)
    assert [text for call in provider.calls for text in call] == [build_embedding_text("blank.py", "github\n")]
    assert chunk_rows(database)[0].embedding is None


def test_embedding_is_generated_in_batches_and_partial_progress_survives_failure(database):
    files = {f"f{n:03d}.py": [(1, f"github token {n}\n")] for n in range(BATCH_SIZE + 5)}
    repository_id, _ = create_repository(database, files=files)

    class FlakyProvider(FakeProvider):
        def embed(self, texts):
            if self.calls:
                raise EmbeddingError("second batch fails")
            return super().embed(texts)

    flaky = FlakyProvider()
    with database() as session:
        repository = session.get(Repository, repository_id)
        with pytest.raises(EmbeddingError):
            index_repository_embeddings(session, repository, flaky)
        embedded = session.scalar(select(func.count()).select_from(RepositoryChunk).where(RepositoryChunk.embedding.is_not(None)))
        assert embedded == BATCH_SIZE

        healthy = FakeProvider()
        summary = index_repository_embeddings(session, repository, healthy)
        assert (summary.chunks_reused, summary.chunks_embedded) == (BATCH_SIZE, 5)


# ---------------------------------------------------------------- provider


def test_provider_sends_expected_request_and_parses_vectors():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        seen["url"] = str(request.url)
        data = [{"index": 1, "embedding": [0.2] * EMBEDDING_DIMENSIONS}, {"index": 0, "embedding": [0.1] * EMBEDDING_DIMENSIONS}]
        return httpx.Response(200, json={"data": data})

    with_provider = OpenAIEmbeddingProvider(API_KEY, "text-embedding-3-small", "https://api.example.test/v1/",
                                            transport=httpx.MockTransport(handler))
    vectors = with_provider.embed(["a", "b"])

    assert vectors[0][0] == 0.1 and vectors[1][0] == 0.2
    assert seen["url"] == "https://api.example.test/v1/embeddings"
    assert seen["auth"] == f"Bearer {API_KEY}"
    assert seen["body"] == {"model": "text-embedding-3-small", "input": ["a", "b"], "dimensions": EMBEDDING_DIMENSIONS}
    assert with_provider.embed([]) == []


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"error": {"message": f"bad key {API_KEY}"}}),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]}),
        httpx.Response(200, json={"data": []}),
        httpx.Response(200, json={"unexpected": True}),
    ],
)
def test_provider_errors_are_safe(response):
    provider = OpenAIEmbeddingProvider(API_KEY, "text-embedding-3-small", "https://api.example.test/v1",
                                       transport=httpx.MockTransport(lambda _request: response))

    with pytest.raises(EmbeddingError) as error:
        provider.embed(["text"])

    assert API_KEY not in str(error.value)


def test_provider_network_failure_is_safe():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    provider = OpenAIEmbeddingProvider(API_KEY, "m", "https://api.example.test/v1", transport=httpx.MockTransport(handler))

    with pytest.raises(EmbeddingError):
        provider.embed(["text"])


def test_get_embedding_provider_requires_a_configured_key(monkeypatch):
    settings = SimpleNamespace(embedding_api_key=None, embedding_model="m", embedding_base_url="https://x.test/v1")
    monkeypatch.setattr(provider_module, "get_settings", lambda: settings)

    with pytest.raises(EmbeddingNotConfiguredError):
        get_embedding_provider()

    from pydantic import SecretStr
    settings.embedding_api_key = SecretStr(API_KEY)
    assert get_embedding_provider().model == "m"


def test_content_hash_depends_on_model_and_text():
    text = build_embedding_text("a.py", "code")
    assert content_hash("m1", text) != content_hash("m2", text)
    assert content_hash("m1", text) != content_hash("m1", build_embedding_text("a.py", "code2"))
    assert content_hash("m1", text) == content_hash("m1", text)
