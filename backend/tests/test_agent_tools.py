"""Tests for the read-only agent tools: search_code, read_file, and analyze_project."""

import json
import re
import uuid

import pytest
from sqlalchemy import event, func, select

from app.analyzer import service as analyzer_service
from app.core.exceptions import NotFoundError
from app.db.models import Repository, RepositoryAnalysis, RepositoryChunk, RepositoryFile, User
from app.embeddings.indexing import index_repository_embeddings
from app.embeddings.provider import EmbeddingError
from app.tools import TOOLS, Tool, ToolContext, execute_tool, get_tool, tool_definitions
from app.tools.base import ToolInput
from app.tools.read_file import MAX_READ_CHARS, MAX_READ_LINES
from app.tools.search_code import SearchCodeInput
from tests.test_repository_context import (  # noqa: F401  (helpers/fixtures used by name)
    BASE_FILES,
    QUESTION,
    SECRET_LINES,
    SECRET_QUESTION,
    SECRET_VALUES,
    WITHHELD_VALUES,
    add_analysis,
)
from tests.test_semantic_search import client, create_repository, database, provider  # noqa: F401

NUL = chr(0)
BACKSLASH = chr(92)
API_KEY = "sk-test-embedding-key-that-must-never-leak"


@pytest.fixture
def make_context(database, provider):
    """Build a ToolContext for a seeded user; sessions are closed after the test."""

    opened = []

    def make(email="me@example.com", *, provider_factory=None):
        session = database()
        opened.append(session)
        user = session.query(User).filter(User.email == email).one()
        return ToolContext(session, user, provider_factory or (lambda: provider))

    yield make
    for session in opened:
        session.close()


def run(context, name, **arguments):
    return execute_tool(name, arguments, context)


def embed(database, repository_id, provider):
    with database() as session:
        index_repository_embeddings(session, session.get(Repository, repository_id), provider)


def chunked(lines, size=100):
    return [(start + 1, "\n".join(lines[start : start + size]) + "\n") for start in range(0, len(lines), size)]


def error_code(result):
    assert not result.ok and result.output is None
    return result.error.code


# --------------------------------------------------------------- registry and abstraction


def test_registry_discovers_the_three_tools_with_json_schemas():
    definitions = tool_definitions()

    assert [d["name"] for d in definitions] == ["search_code", "read_file", "analyze_project"]
    assert [tool.name for tool in TOOLS] == [d["name"] for d in definitions]
    for definition in definitions:
        assert set(definition) == {"name", "description", "input_schema"}
        assert definition["description"]
        schema = definition["input_schema"]
        assert schema["type"] == "object" and "repository_id" in schema["required"]
        assert schema["additionalProperties"] is False
    json.dumps(definitions)
    limit = get_tool("search_code").input_schema["properties"]["limit"]
    assert (limit["minimum"], limit["maximum"]) == (1, 20)
    assert set(get_tool("read_file").input_schema["required"]) == {"repository_id", "file_path"}
    assert get_tool("nope") is None


def test_unknown_tool_is_a_structured_error_that_does_not_echo_the_name(make_context, database):
    create_repository(database, files=BASE_FILES)

    result = execute_tool("rm_rf_slash", {}, make_context())

    assert error_code(result) == "UNKNOWN_TOOL"
    assert "search_code" in result.error.message and "rm_rf_slash" not in result.error.message


def test_results_are_json_serializable_with_a_stable_shape(make_context, database, provider):
    repository_id, _ = create_repository(database, files=BASE_FILES)
    context = make_context()

    ok = run(context, "search_code", repository_id=str(repository_id), query="github").to_dict()
    bad = run(context, "read_file", repository_id=str(repository_id), file_path="nope.py").to_dict()

    assert set(ok) == {"ok", "output"} and ok["ok"] is True
    assert set(bad) == {"ok", "error"} and bad["ok"] is False and set(bad["error"]) == {"code", "message"}
    json.dumps([ok, bad])


@pytest.mark.parametrize("tool", [t.name for t in TOOLS])
def test_unauthenticated_and_inactive_callers_are_rejected_before_anything_runs(make_context, database, tool):
    repository_id, _ = create_repository(database, files=BASE_FILES)
    session = make_context().session
    arguments = {"repository_id": str(repository_id), "query": "github", "file_path": "app/ui.py"}
    arguments = {k: v for k, v in arguments.items() if k in get_tool(tool).input_schema["properties"]}

    anonymous = execute_tool(tool, arguments, ToolContext(session, None))
    inactive = execute_tool(tool, arguments, ToolContext(session, User(email="x@example.com", status="disabled")))

    assert error_code(anonymous) == "UNAUTHORIZED" and error_code(inactive) == "UNAUTHORIZED"
    assert anonymous.to_dict()["error"]["message"] == "Authentication is required"


@pytest.mark.parametrize("tool", [t.name for t in TOOLS])
@pytest.mark.parametrize(
    "arguments",
    [{}, {"repository_id": "not-a-uuid"}, {"repository_id": 12345}, {"repository_id": None}, None, [], "text"],
)
def test_invalid_arguments_report_field_errors(make_context, database, tool, arguments):
    create_repository(database, files=BASE_FILES)

    result = execute_tool(tool, arguments, make_context())

    assert error_code(result) == "INVALID_INPUT"
    assert result.error.details and all({"location", "message", "type"} <= set(d) for d in result.error.details)


def test_submitted_values_are_never_echoed_in_validation_errors(make_context, database):
    create_repository(database, files=BASE_FILES)
    secret = "sk-secret-value-that-must-not-be-echoed"

    result = run(make_context(), "search_code", repository_id=secret, query="x", limit=secret)

    assert error_code(result) == "INVALID_INPUT"
    assert secret not in json.dumps(result.to_dict())


def test_unknown_fields_are_rejected(make_context, database):
    repository_id, _ = create_repository(database, files=BASE_FILES)

    result = run(make_context(), "analyze_project", repository_id=str(repository_id), delete_everything=True)

    assert error_code(result) == "INVALID_INPUT"


class _NoInput(ToolInput):
    pass


def test_application_errors_become_results_but_unexpected_errors_propagate(make_context, database):
    create_repository(database, files=BASE_FILES)

    def missing(context, arguments):
        raise NotFoundError("gone")

    def broken(context, arguments):
        raise RuntimeError("bug")

    context = make_context()
    assert error_code(Tool("t", "d", _NoInput, missing).execute(context, {})) == "RESOURCE_NOT_FOUND"
    with pytest.raises(RuntimeError):
        Tool("t", "d", _NoInput, broken).execute(context, {})


def test_tools_are_read_only(make_context, database, provider):
    repository_id, _ = create_repository(database, files=BASE_FILES)
    add_analysis(database, repository_id)
    embed(database, repository_id, provider)

    def snapshot():
        with database() as session:
            counts = [session.scalar(select(func.count()).select_from(m)) for m in (RepositoryAnalysis, RepositoryFile, RepositoryChunk)]
            return counts, session.scalars(select(RepositoryAnalysis.updated_at)).one()

    before = snapshot()
    context = make_context()
    run(context, "search_code", repository_id=str(repository_id), query=QUESTION)
    run(context, "read_file", repository_id=str(repository_id), file_path="app/ui.py")
    run(context, "analyze_project", repository_id=str(repository_id))

    assert snapshot() == before


# --------------------------------------------------------------- search_code


def search(context, repository_id, query=QUESTION, **extra):
    return run(context, "search_code", repository_id=str(repository_id), query=query, **extra)


def test_search_code_returns_structured_results_with_provenance(make_context, database, provider):
    repository_id, _ = create_repository(database, files=BASE_FILES)
    embed(database, repository_id, provider)

    result = search(make_context(), repository_id)

    assert result.ok
    output = result.output
    assert set(output) == {"repository_id", "query", "results", "retrieval"}
    assert output["repository_id"] == str(repository_id) and output["query"] == QUESTION
    top = output["results"][0]
    assert set(top) == {"file_path", "language", "start_line", "end_line", "snippet", "score", "sources", "matched_terms", "truncated"}
    assert top["file_path"] == "app/github_oauth.py" and top["language"] == "python"
    assert (top["start_line"], top["end_line"]) == (1, 3) and "github_callback" in top["snippet"]
    assert top["sources"] == ["exact", "semantic"] and top["matched_terms"] == ["GitHub"] and top["score"] > 0
    retrieval = output["retrieval"]
    assert retrieval["exact"]["status"] == "used" and retrieval["semantic"]["status"] == "used"
    assert retrieval["analysis"] == "skipped" and "project" not in output


def test_search_code_limit_is_respected_and_defaults_to_eight(make_context, database, provider):
    files = {f"f{n:02d}.py": [(1, "needle x\n")] for n in range(12)}
    repository_id, _ = create_repository(database, files=files)
    embed(database, repository_id, provider)
    context = make_context()

    default = search(context, repository_id, query="needle")
    limited = search(context, repository_id, query="needle", limit=3)

    assert len(default.output["results"]) == 8 and len(limited.output["results"]) == 3
    assert limited.output["retrieval"]["truncated"] is True


def test_search_code_limit_is_an_upper_bound_when_only_exact_search_is_available(make_context, database):
    files = {f"f{n:02d}.py": [(1, "needle x\n")] for n in range(12)}
    repository_id, _ = create_repository(database, files=files)

    result = search(make_context(), repository_id, query="needle", limit=20)

    # Exact search returns at most 5 hits per extracted term; the limit never forces more.
    assert result.ok and len(result.output["results"]) == 5


def test_search_code_still_works_without_embeddings_or_when_the_provider_fails(make_context, database):
    repository_id, _ = create_repository(database, files=BASE_FILES)

    class FailingProvider:
        model = "fake"

        def embed(self, texts):
            raise EmbeddingError(f"boom {API_KEY} raw provider body")

    no_embeddings = search(make_context(), repository_id)
    assert no_embeddings.ok and no_embeddings.output["retrieval"]["semantic"]["status"] == "no_embeddings"
    assert no_embeddings.output["results"][0]["sources"] == ["exact"]

    with_failure = search(make_context(provider_factory=lambda: FailingProvider()), repository_id)
    assert with_failure.ok and API_KEY not in json.dumps(with_failure.to_dict())


def test_search_code_never_returns_vectors_or_keys(make_context, database, provider):
    repository_id, _ = create_repository(database, files=BASE_FILES)
    embed(database, repository_id, provider)

    text = json.dumps(search(make_context(), repository_id).to_dict())

    assert '"embedding"' not in text and API_KEY not in text
    assert not re.search(r"\[\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+", text)


def test_search_code_redacts_secrets_and_withholds_sensitive_files(make_context, database, provider):
    files = {
        "app/config.py": [(1, "\n".join(SECRET_LINES) + "\n")],
        "config/secrets.yaml": [(1, "github token credentials login\npassword: supersecretvalue1\n")],
        ".env.local": [(1, "github token credentials login\nKEY=abcdef123456\n")],
        "certs/server.pem": [(1, "github token credentials login\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSj\n")],
    }
    repository_id, _ = create_repository(database, files=files)
    embed(database, repository_id, provider)

    result = search(make_context(), repository_id, query=SECRET_QUESTION)

    text = json.dumps(result.to_dict())
    assert result.ok
    for leaked in SECRET_VALUES + WITHHELD_VALUES:
        assert leaked not in text
    assert [r["file_path"] for r in result.output["results"]] == ["app/config.py"]
    assert result.output["retrieval"]["withheld_chunks"] == 3 and result.output["retrieval"]["redactions"] >= 8


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": ""}, {"query": "   "}, {"query": "a" + NUL + "b"}, {"query": "x" * 1001}, {},
        {"query": "ok", "limit": 0}, {"query": "ok", "limit": 21}, {"query": "ok", "limit": "many"}, {"query": "ok", "limit": 1.5},
    ],
)
def test_search_code_validates_its_input(make_context, database, arguments):
    repository_id, _ = create_repository(database, files=BASE_FILES)

    result = run(make_context(), "search_code", repository_id=str(repository_id), **arguments)

    assert error_code(result) == "INVALID_INPUT"


def test_search_code_query_limit_boundaries_are_accepted(make_context, database):
    repository_id, _ = create_repository(database, files=BASE_FILES)

    for arguments in ({"query": "a" * 1000, "limit": 1}, {"query": "github", "limit": 20}):
        assert run(make_context(), "search_code", repository_id=str(repository_id), **arguments).ok
    assert SearchCodeInput.model_validate({"repository_id": str(uuid.uuid4()), "query": "naïve café 日本語 🚀"}).limit == 8


def test_search_code_enforces_ownership(make_context, database, provider):
    other_id, _ = create_repository(database, email="other@example.com", files=BASE_FILES)
    embed(database, other_id, provider)
    create_repository(database, email="me@example.com", files={"mine.py": [(1, "nothing\n")]})
    calls_before = len(provider.calls)

    foreign = search(make_context(), other_id)
    unknown = search(make_context(), uuid.uuid4())

    assert error_code(foreign) == error_code(unknown) == "RESOURCE_NOT_FOUND"
    assert "github_callback" not in json.dumps(foreign.to_dict()) and len(provider.calls) == calls_before


def test_search_code_reports_an_unscanned_repository(make_context, database):
    repository_id, _ = create_repository(database, files={})

    assert error_code(search(make_context(), repository_id)) == "CONFLICT"


def test_search_code_is_deterministic(make_context, database, provider):
    repository_id, _ = create_repository(database, files=BASE_FILES)
    embed(database, repository_id, provider)

    assert search(make_context(), repository_id).to_dict() == search(make_context(), repository_id).to_dict()


# --------------------------------------------------------------- read_file


LINES = [f"line {n}" for n in range(1, 351)]


@pytest.fixture
def reading(make_context, database):
    """A repository with a 350-line file split into 100-line chunks plus small files."""

    files = {
        "src/big.py": chunked(LINES),
        "src/small.py": [(1, "a = 1\nb = 2\n")],
        "src/no_newline.py": [(1, "a = 1\nb = 2")],
        "src/empty.py": [],
    }
    repository_id, _ = create_repository(database, files=files)
    context = make_context()

    def read(file_path, **extra):
        return run(context, "read_file", repository_id=str(repository_id), file_path=file_path, **extra)

    read.repository_id = repository_id
    return read


def test_read_file_returns_the_requested_lines_across_chunk_boundaries(reading):
    result = reading("src/big.py", start_line=95, end_line=105)

    assert result.ok
    assert result.output == {
        "repository_id": str(reading.repository_id), "file_path": "src/big.py", "language": "python",
        "start_line": 95, "end_line": 105, "total_lines": 350, "has_more": True, "truncated": False,
        "redactions": 0, "content": "\n".join(LINES[94:105]),
    }


def test_read_file_defaults_to_the_first_300_lines_and_can_continue(reading):
    first = reading("src/big.py")
    rest = reading("src/big.py", start_line=first.output["end_line"] + 1)

    assert MAX_READ_LINES == 300
    assert (first.output["start_line"], first.output["end_line"], first.output["has_more"]) == (1, 300, True)
    assert first.output["content"] == "\n".join(LINES[:300])
    assert (rest.output["start_line"], rest.output["end_line"], rest.output["has_more"]) == (301, 350, False)
    assert first.output["content"] + "\n" + rest.output["content"] == "\n".join(LINES)


def test_read_file_range_variants_and_clamping(reading):
    assert reading("src/big.py", end_line=10).output["content"] == "\n".join(LINES[:10])
    clamped = reading("src/big.py", start_line=340, end_line=400)
    assert (clamped.output["end_line"], clamped.output["has_more"]) == (350, False)
    assert reading("src/big.py", start_line=350, end_line=350).output["content"] == "line 350"


def test_read_file_handles_trailing_newlines_and_empty_files(reading):
    for path in ("src/small.py", "src/no_newline.py"):
        output = reading(path).output
        assert (output["content"], output["total_lines"], output["end_line"], output["has_more"]) == ("a = 1\nb = 2", 2, 2, False)

    empty = reading("src/empty.py").output
    assert (empty["content"], empty["total_lines"], empty["start_line"], empty["end_line"], empty["has_more"]) == ("", 0, 1, 0, False)


@pytest.mark.parametrize(
    "arguments",
    [
        {"start_line": 1, "end_line": MAX_READ_LINES + 1},
        {"start_line": 50, "end_line": 350},
        {"start_line": 10, "end_line": 9},
        {"start_line": 0}, {"end_line": 0}, {"start_line": -5}, {"start_line": "one"}, {"end_line": 2.5},
    ],
)
def test_read_file_rejects_oversized_or_invalid_ranges(reading, arguments):
    assert error_code(reading("src/big.py", **arguments)) == "INVALID_INPUT"


def test_read_file_accepts_exactly_the_maximum_range(reading):
    result = reading("src/big.py", start_line=51, end_line=350)

    assert result.ok and result.output["end_line"] - result.output["start_line"] + 1 == MAX_READ_LINES


def test_read_file_start_beyond_the_end_is_a_clean_error(reading):
    result = reading("src/big.py", start_line=351)

    assert error_code(result) == "BAD_REQUEST" and "350 lines" in result.error.message


def test_read_file_caps_the_response_size_on_whole_lines(make_context, database):
    lines = ["x" * 99] * 300
    repository_id, _ = create_repository(database, files={"src/wide.py": chunked(lines)})

    output = run(make_context(), "read_file", repository_id=str(repository_id), file_path="src/wide.py").output

    kept = output["content"].count("\n") + 1
    assert len(output["content"]) <= MAX_READ_CHARS and output["truncated"] is True
    assert output["content"] == "\n".join(lines[:kept]) and output["end_line"] == kept
    assert output["has_more"] is True and output["total_lines"] == 300


@pytest.mark.parametrize(
    "path",
    [
        "../etc/passwd", "app/../../etc/passwd", "..", ".", "./src/small.py", "src/./small.py", "src//small.py", "src/small.py/",
        "/etc/passwd", "/src/small.py", "C:/Windows/win.ini", "c:" + BACKSLASH + "windows", "src" + BACKSLASH + "small.py",
        "src/small.py" + NUL, "src/../src/small.py",
    ],
)
def test_read_file_rejects_path_traversal_and_non_repository_paths(reading, path):
    assert error_code(reading(path)) == "INVALID_INPUT"


def test_read_file_only_serves_indexed_data_never_the_local_disk(reading):
    # These exist on the machine running the tests but were never indexed for the repository.
    for path in ("backend/main.py", "README.md", "backend/app/tools/read_file.py"):
        result = reading(path)
        assert error_code(result) == "RESOURCE_NOT_FOUND"
        assert result.error.message == "File not found in the indexed repository"
    assert error_code(reading(__file__.replace(BACKSLASH, "/"))) == "INVALID_INPUT"


def test_read_file_paths_are_exact_and_case_sensitive(reading):
    assert error_code(reading("SRC/BIG.PY")) == "RESOURCE_NOT_FOUND"
    assert error_code(reading("src/big")) == "RESOURCE_NOT_FOUND"


@pytest.mark.parametrize(
    "path",
    [".env", ".env.local", "config/secrets.yaml", "certs/server.pem", "keys/id_rsa", ".npmrc", "aws/credentials", "gcp/service-account.json", "secrets/db.yaml"],
)
def test_sensitive_files_cannot_be_read_even_when_indexed(make_context, database, path):
    repository_id, _ = create_repository(database, files={path: [(1, "TOPSECRET=hunter22secret\n")], "src/ok.py": [(1, "x = 1\n")]})
    context = make_context()

    existing = run(context, "read_file", repository_id=str(repository_id), file_path=path)
    absent = run(context, "read_file", repository_id=str(repository_id), file_path="never/" + path)

    assert error_code(existing) == "FORBIDDEN"
    assert "hunter22secret" not in json.dumps(existing.to_dict())
    # The answer depends only on the path, so it cannot reveal which sensitive files exist.
    assert existing.to_dict() == absent.to_dict()


def test_secrets_inside_readable_files_are_redacted_without_shifting_lines(make_context, database):
    repository_id, _ = create_repository(database, files={"app/config.py": [(1, "\n".join(SECRET_LINES) + "\n")]})

    output = run(make_context(), "read_file", repository_id=str(repository_id), file_path="app/config.py").output

    for leaked in SECRET_VALUES:
        assert leaked not in output["content"]
    assert output["redactions"] >= 8 and output["content"].count("\n") + 1 == len(SECRET_LINES) == output["end_line"]


def test_read_file_enforces_ownership_and_repository_isolation(make_context, database):
    other_id, _ = create_repository(database, email="other@example.com", files={"src/theirs.py": [(1, "x = 1\n")], ".env": [(1, "K=1\n")]})
    mine_a, _ = create_repository(database, email="me@example.com", files={"src/a.py": [(1, "a = 1\n")]})
    context = make_context()
    ctx_user = context.user

    foreign = run(context, "read_file", repository_id=str(other_id), file_path="src/theirs.py")
    foreign_sensitive = run(context, "read_file", repository_id=str(other_id), file_path=".env")
    unknown = run(context, "read_file", repository_id=str(uuid.uuid4()), file_path="src/a.py")
    assert {error_code(foreign), error_code(foreign_sensitive), error_code(unknown)} == {"RESOURCE_NOT_FOUND"}

    with database() as session:
        second = Repository(github_account_id=session.get(Repository, mine_a).github_account_id, github_repository_id=999_001, owner="me", name="second")
        session.add(second)
        session.flush()
        file = RepositoryFile(repository_id=second.id, path="src/only_in_second.py", language="python", sha="x")
        session.add(file)
        session.flush()
        session.add(RepositoryChunk(repository_file_id=file.id, chunk_index=0, start_line=1, end_line=1, content="s = 1\n"))
        session.commit()
        second_id = second.id
    assert ctx_user is not None
    assert error_code(run(context, "read_file", repository_id=str(mine_a), file_path="src/only_in_second.py")) == "RESOURCE_NOT_FOUND"
    assert run(context, "read_file", repository_id=str(second_id), file_path="src/only_in_second.py").ok


def test_read_file_never_loads_or_returns_embedding_vectors(make_context, database, provider):
    repository_id, _ = create_repository(database, files={"src/big.py": chunked(LINES)})
    embed(database, repository_id, provider)
    context = make_context()
    statements = []

    def record(connection, cursor, statement, parameters, execution_context, executemany):
        statements.append(statement.lower())

    engine = context.session.get_bind()
    event.listen(engine, "before_cursor_execute", record)
    try:
        result = run(context, "read_file", repository_id=str(repository_id), file_path="src/big.py")
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert result.ok and statements
    assert not any("embedding" in statement for statement in statements)
    assert '"embedding"' not in json.dumps(result.to_dict())


# --------------------------------------------------------------- analyze_project


def test_analyze_project_returns_the_stored_analysis_and_matches_the_endpoint(make_context, database, client):
    repository_id, headers = create_repository(database, files=BASE_FILES)
    add_analysis(database, repository_id, frameworks=["FastAPI", "Next.js"], package_managers=["npm", "pip"])

    result = run(make_context(), "analyze_project", repository_id=str(repository_id))

    assert result.ok
    output = result.output
    assert output["repository_id"] == str(repository_id) and output["status"] == "completed"
    assert (output["project_type"], output["languages"], output["frameworks"]) == ("web_application", ["python"], ["FastAPI", "Next.js"])
    assert output["dependencies"] == [{"name": "fastapi", "ecosystem": "pypi", "dev": False}]
    assert output["entry_points"] == [{"kind": "python_file", "name": "main.py", "path": "main.py"}]
    assert set(output) == {
        "repository_id", "status", "project_type", "languages", "frameworks", "package_managers", "dependencies",
        "important_files", "entry_points", "skipped_manifests", "updated_at",
    }
    assert client.get(f"/api/v1/repositories/{repository_id}/analysis", headers=headers).json() == output


def test_analyze_project_reads_the_stored_analysis_without_recomputing(make_context, database, monkeypatch):
    repository_id, _ = create_repository(database, files=BASE_FILES)
    add_analysis(database, repository_id)

    def boom(*args, **kwargs):
        raise AssertionError("analysis must not be recomputed")

    monkeypatch.setattr(analyzer_service, "analyze_repository", boom)
    monkeypatch.setattr(analyzer_service, "analyze_after_scan", boom)

    assert run(make_context(), "analyze_project", repository_id=str(repository_id)).ok


def test_analyze_project_reports_an_unanalyzed_repository(make_context, database):
    repository_id, _ = create_repository(database, files=BASE_FILES)

    result = run(make_context(), "analyze_project", repository_id=str(repository_id))

    assert error_code(result) == "CONFLICT" and "not been analyzed" in result.error.message


def test_analyze_project_enforces_ownership(make_context, database):
    other_id, _ = create_repository(database, email="other@example.com", files=BASE_FILES)
    add_analysis(database, other_id)
    create_repository(database, email="me@example.com", files={"mine.py": [(1, "x\n")]})
    context = make_context()

    foreign = run(context, "analyze_project", repository_id=str(other_id))
    unknown = run(context, "analyze_project", repository_id=str(uuid.uuid4()))

    assert error_code(foreign) == error_code(unknown) == "RESOURCE_NOT_FOUND"
    assert "web_application" not in json.dumps(foreign.to_dict())


def test_analyze_project_surfaces_a_failed_analysis_status(make_context, database):
    repository_id, _ = create_repository(database, files=BASE_FILES)
    add_analysis(database, repository_id, status="failed")

    result = run(make_context(), "analyze_project", repository_id=str(repository_id))

    assert result.ok and result.output["status"] == "failed"
