"""Tests for repository code search over indexed chunks."""

import logging
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth.security import create_access_token
from app.db.base import Base
from app.db.database import get_db
from app.db.models import GitHubAccount, Repository, RepositoryChunk, RepositoryFile, User
from app.scanner.search import DEFAULT_RESULT_LIMIT, MAX_RESULT_LIMIT
from main import app


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


def create_repository(factory, email: str, files: dict[str, list[tuple[int, str]]] | None = None):
    """Insert a user and repository with indexed files: {path: [(start_line, content), ...]}."""

    with factory() as session:
        user = User(email=email, name="Owner", status="active")
        session.add(user)
        session.flush()
        account = GitHubAccount(user_id=user.id, github_user_id=abs(hash(email)) % 10**9, login=email.split("@")[0])
        session.add(account)
        session.flush()
        repository = Repository(
            github_account_id=account.id,
            github_repository_id=abs(hash(email + "r")) % 10**9,
            owner=account.login,
            name="project",
        )
        session.add(repository)
        session.flush()
        for path, chunks in (files or {}).items():
            file = RepositoryFile(repository_id=repository.id, path=path, language="python", sha=uuid.uuid4().hex)
            session.add(file)
            session.flush()
            for index, (start_line, content) in enumerate(chunks):
                line_count = len(content.rstrip("\n").split("\n"))
                session.add(
                    RepositoryChunk(
                        repository_file_id=file.id,
                        chunk_index=index,
                        start_line=start_line,
                        end_line=start_line + line_count - 1,
                        content=content,
                    )
                )
        session.commit()
        headers = {"Authorization": f"Bearer {create_access_token(user.id)}"}
        return repository.id, headers


def search(client, repository_id, headers, **params):
    return client.get(f"/api/v1/repositories/{repository_id}/search", params=params, headers=headers)


def numbered(start: int, count: int, marker_line: int | None = None, marker="TARGET") -> str:
    return "".join(
        f"{marker if n == marker_line else 'filler'} {n}\n" for n in range(start, start + count)
    )


def test_authenticated_user_can_search_and_gets_path_lines_and_snippet(client, database):
    content = numbered(1, 20, marker_line=10, marker="def github_callback():")
    repository_id, headers = create_repository(database, "me@example.com", {"app/routes.py": [(1, content)]})

    response = search(client, repository_id, headers, query="github_callback")

    assert response.status_code == 200
    body = response.json()
    assert body["repository_id"] == str(repository_id)
    assert body["query"] == "github_callback"
    assert body["results"] == [
        {
            "file_path": "app/routes.py",
            "language": "python",
            "start_line": 8,
            "end_line": 12,
            "snippet": "filler 8\nfiller 9\ndef github_callback(): 10\nfiller 11\nfiller 12",
        }
    ]


def test_line_numbers_account_for_chunk_offset(client, database):
    chunk = numbered(101, 30, marker_line=125, marker="needle")
    repository_id, headers = create_repository(database, "me@example.com", {"big.py": [(1, numbered(1, 100)), (101, chunk)]})

    result = search(client, repository_id, headers, query="needle").json()["results"][0]

    assert (result["start_line"], result["end_line"]) == (123, 127)
    assert "needle 125" in result["snippet"]


def test_search_is_case_insensitive_and_matches_first_or_last_line(client, database):
    repository_id, headers = create_repository(
        database, "me@example.com", {"a.py": [(1, "first_TARGET\nmiddle\nlast_target")]}
    )

    upper = search(client, repository_id, headers, query="FIRST_target").json()["results"][0]
    lower = search(client, repository_id, headers, query="LAST_TARGET").json()["results"][0]

    assert (upper["start_line"], upper["end_line"]) == (1, 3)
    assert (lower["start_line"], lower["end_line"]) == (1, 3)


def test_multiple_matching_chunks_and_files_are_returned_in_order(client, database):
    repository_id, headers = create_repository(
        database,
        "me@example.com",
        {
            "b.py": [(1, "nothing\n"), (2, "b has needle\n")],
            "a.py": [(1, "a has needle\n"), (2, "unrelated\n"), (3, "needle again\n")],
        },
    )

    results = search(client, repository_id, headers, query="needle").json()["results"]

    assert [(r["file_path"], r["start_line"]) for r in results] == [
        ("a.py", 1),
        ("a.py", 3),
        ("b.py", 2),
    ]


def test_multiline_query_matches_across_lines(client, database):
    repository_id, headers = create_repository(
        database, "me@example.com", {"a.py": [(1, "x = 1\nif ready:\n    go()\ny = 2\n")]}
    )

    response = search(client, repository_id, headers, query="if ready:\n    go()")

    assert len(response.json()["results"]) == 1


def test_no_results_is_a_successful_empty_list(client, database):
    repository_id, headers = create_repository(database, "me@example.com", {"a.py": [(1, "hello\n")]})

    response = search(client, repository_id, headers, query="absent-term")

    assert response.status_code == 200
    assert response.json()["results"] == []


def test_like_wildcards_in_query_are_matched_literally(client, database):
    repository_id, headers = create_repository(
        database, "me@example.com", {"a.py": [(1, "plain text\n"), (2, "100% done\n"), (3, "snake_case\n")]}
    )

    percent = search(client, repository_id, headers, query="%").json()["results"]
    underscore = search(client, repository_id, headers, query="e_c").json()["results"]
    unmatched = search(client, repository_id, headers, query="p_ain").json()["results"]

    assert [r["start_line"] for r in percent] == [2]
    assert [r["start_line"] for r in underscore] == [3]
    assert unmatched == []


def test_only_the_selected_repository_is_searched(client, database):
    first_id, headers = create_repository(database, "me@example.com", {"mine.py": [(1, "shared needle\n")]})
    create_repository(database, "other@example.com", {"theirs.py": [(1, "shared needle\n")]})

    results = search(client, first_id, headers, query="needle").json()["results"]

    assert [r["file_path"] for r in results] == ["mine.py"]


def test_result_limit_defaults_and_is_enforced(client, database):
    chunks = [(n, f"needle {n}\n") for n in range(1, 31)]
    repository_id, headers = create_repository(database, "me@example.com", {"a.py": chunks})

    default = search(client, repository_id, headers, query="needle").json()["results"]
    limited = search(client, repository_id, headers, query="needle", limit=3).json()["results"]

    assert DEFAULT_RESULT_LIMIT == 20 and len(default) == 20
    assert [r["start_line"] for r in limited] == [1, 2, 3]


@pytest.mark.parametrize("limit", [0, -1, MAX_RESULT_LIMIT + 1, "many"])
def test_invalid_limit_is_rejected(client, database, limit):
    repository_id, headers = create_repository(database, "me@example.com")

    assert search(client, repository_id, headers, query="x", limit=limit).status_code == 422


@pytest.mark.parametrize("query", ["", "   ", "\t\n", "x" * 201])
def test_empty_or_oversized_query_is_rejected(client, database, query):
    repository_id, headers = create_repository(database, "me@example.com", {"a.py": [(1, "x\n")]})

    response = search(client, repository_id, headers, query=query)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_missing_query_parameter_is_rejected(client, database):
    repository_id, headers = create_repository(database, "me@example.com")

    assert search(client, repository_id, headers).status_code == 422


def test_authentication_is_required(client, database):
    repository_id, _ = create_repository(database, "me@example.com", {"a.py": [(1, "needle\n")]})

    missing = search(client, repository_id, {}, query="needle")
    invalid = search(client, repository_id, {"Authorization": "Bearer bad-token"}, query="needle")

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert "needle" not in missing.text


def test_unknown_repository_returns_not_found(client, database):
    _, headers = create_repository(database, "me@example.com")

    response = search(client, uuid.uuid4(), headers, query="needle")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "RESOURCE_NOT_FOUND"


def test_another_users_repository_is_not_searchable(client, database):
    other_id, _ = create_repository(database, "other@example.com", {"secret.py": [(1, "top_secret needle\n")]})
    _, my_headers = create_repository(database, "me@example.com")

    response = search(client, other_id, my_headers, query="needle")

    assert response.status_code == 404
    assert "secret.py" not in response.text and "top_secret" not in response.text


def test_invalid_repository_id_is_rejected(client, database):
    _, headers = create_repository(database, "me@example.com")

    response = client.get("/api/v1/repositories/not-a-uuid/search", params={"query": "x"}, headers=headers)

    assert response.status_code == 422


def test_search_terms_are_not_written_to_logs(client, database, caplog):
    repository_id, headers = create_repository(database, "me@example.com", {"a.py": [(1, "hunter2-secret\n")]})
    caplog.set_level(logging.DEBUG)

    response = search(client, repository_id, headers, query="hunter2-secret")

    assert response.status_code == 200
    assert "hunter2-secret" not in caplog.text
