"""Semantic search against real PostgreSQL + pgvector.

Skipped unless PGVECTOR_TEST_DATABASE_URL points at a disposable database that has
the `vector` extension installed. All tables in it are created and dropped by the test.
"""

import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.routes import repositories as repository_routes
from app.auth.security import create_access_token
from app.db.base import Base
from app.db.database import get_db
from app.db.models import GitHubAccount, Repository, RepositoryChunk, RepositoryFile, User
from app.embeddings import EMBEDDING_DIMENSIONS
from app.scanner.semantic import find_similar_chunks
from main import app
from tests.test_repository_scan import FakeGitHubClient, create_repository, scan, use_github
from tests.test_semantic_search import FakeProvider

pytestmark = pytest.mark.skipif(
    not os.environ.get("PGVECTOR_TEST_DATABASE_URL"),
    reason="PGVECTOR_TEST_DATABASE_URL is not set",
)


def axis(index: int, weight: float = 1.0, other: int | None = None, other_weight: float = 0.0) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[index] = weight
    if other is not None:
        vector[other] = other_weight
    return vector


class AxisProvider:
    """Embeds every text onto axis 0, so chunks on axis 0 are the best match."""

    model = "axis"

    def embed(self, texts):
        return [axis(0) for _ in texts]


@pytest.fixture
def factory():
    engine = create_engine(os.environ["PGVECTOR_TEST_DATABASE_URL"])
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.drop_all(engine)
    engine.dispose()


def seed(factory, email, chunks):
    """chunks: list of (path, start_line, content, embedding)."""

    with factory() as session:
        user = User(email=email, status="active")
        session.add(user)
        session.flush()
        account = GitHubAccount(user_id=user.id, github_user_id=abs(hash(email)) % 10**9, login=email.split("@")[0])
        session.add(account)
        session.flush()
        repository = Repository(github_account_id=account.id, github_repository_id=abs(hash(email + "r")) % 10**9,
                                owner=account.login, name="project")
        session.add(repository)
        session.flush()
        files: dict[str, RepositoryFile] = {}
        for path, start_line, content, embedding in chunks:
            if path not in files:
                files[path] = RepositoryFile(repository_id=repository.id, path=path, language="python", sha=uuid.uuid4().hex)
                session.add(files[path])
                session.flush()
            session.add(RepositoryChunk(repository_file_id=files[path].id, chunk_index=start_line, start_line=start_line,
                                        end_line=start_line + 1, content=content, embedding=embedding))
        session.commit()
        return repository.id, {"Authorization": f"Bearer {create_access_token(user.id)}"}


def test_pgvector_ranks_by_cosine_similarity_scoped_to_one_repository(factory):
    repository_id, _ = seed(factory, "me@example.com", [
        ("best.py", 1, "best", axis(0)),
        ("close.py", 1, "close", axis(0, 1.0, other=1, other_weight=1.0)),
        ("far.py", 1, "far", axis(2)),
        ("opposite.py", 1, "opposite", axis(0, -1.0)),
        ("none.py", 1, "not embedded", None),
    ])
    seed(factory, "other@example.com", [("intruder.py", 1, "other user", axis(0))])

    with factory() as session:
        repository = session.get(Repository, repository_id)
        hits = find_similar_chunks(session, repository, axis(0), limit=10)
        filtered = find_similar_chunks(session, repository, axis(0), limit=10, min_score=0.5)
        limited = find_similar_chunks(session, repository, axis(0), limit=1)

    assert [hit.file_path for hit in hits] == ["best.py", "close.py", "far.py", "opposite.py"]
    assert [hit.score for hit in hits] == [1.0, 0.7071, 0.0, -1.0]
    assert [hit.file_path for hit in filtered] == ["best.py", "close.py"]
    assert [hit.file_path for hit in limited] == ["best.py"]


def test_semantic_search_endpoint_end_to_end_on_postgres(factory, monkeypatch):
    repository_id, headers = seed(factory, "me@example.com", [
        ("best.py", 1, "def github_callback(): ...", axis(0)),
        ("far.py", 1, "render button", axis(2)),
    ])
    monkeypatch.setattr(repository_routes, "get_embedding_provider", lambda: AxisProvider())

    def override_get_db():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            response = client.get(f"/api/v1/repositories/{repository_id}/semantic-search",
                                  params={"query": "Where is GitHub authentication handled?", "limit": 1}, headers=headers)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["results"] == [{
        "file_path": "best.py", "language": "python", "start_line": 1, "end_line": 2,
        "snippet": "def github_callback(): ...", "score": 1.0,
    }]


def test_scan_automatically_embeds_and_search_works_on_postgres(factory, monkeypatch):
    files = {
        "app/github_oauth.py": "def github_callback():\n    exchange oauth token for login\n",
        "app/database.py": "engine = create_engine()\nsession query connection\n",
        "app/ui.py": "render button with css layout\n",
    }
    repository_id, headers = create_repository(factory)
    provider = FakeProvider()
    monkeypatch.setattr(repository_routes, "get_embedding_provider", lambda: provider)

    def override_get_db():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    def semantic(client):
        response = client.get(f"/api/v1/repositories/{repository_id}/semantic-search",
                              params={"query": "Where is GitHub authentication handled?"}, headers=headers)
        assert response.status_code == 200
        return response.json()["results"]

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            use_github(monkeypatch, FakeGitHubClient(files))
            first = scan(client, repository_id, headers).json()
            assert first["embeddings"]["status"] == "completed"
            assert first["embeddings"]["chunks_embedded"] == 3
            results = semantic(client)
            assert results[0]["file_path"] == "app/github_oauth.py" and results[0]["score"] > 0.5

            calls_before = len(provider.calls)
            second = scan(client, repository_id, headers).json()
            assert (second["embeddings"]["chunks_embedded"], second["embeddings"]["chunks_reused"]) == (0, 3)
            assert len(provider.calls) == calls_before

            use_github(monkeypatch, FakeGitHubClient({k: v for k, v in files.items() if k != "app/github_oauth.py"}))
            assert scan(client, repository_id, headers).json()["files_removed"] == 1
            assert "app/github_oauth.py" not in [r["file_path"] for r in semantic(client)]
    finally:
        app.dependency_overrides.clear()
