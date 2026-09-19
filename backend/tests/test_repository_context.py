"""Tests for the repository Context Engine and its endpoint."""

import json
import logging
import re
import time
import uuid

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select

from app.api.routes import repositories as repository_routes
from app.context.query import MAX_TERMS, STOPWORDS, extract_terms
from app.context.redaction import REDACTION, is_sensitive_path, redact_secrets
from app.context.service import (
    MAX_CHUNKS_PER_FILE,
    MAX_SNIPPET_CHARS,
    FusedChunk,
    assemble_chunks,
    build_layout,
    fuse_hits,
    truncate_at_line,
)
from app.db.models import RepositoryAnalysis, RepositoryChunk, RepositoryFile
from app.embeddings.provider import EmbeddingError, EmbeddingNotConfiguredError
from app.scanner.search import CodeSearchHit
from app.scanner.semantic import SemanticHit
from app.schemas.context import ContextRequest
from tests.test_repository_scan import FakeGitHubClient, scan, use_github
from tests.test_repository_scan import create_repository as create_empty_repository
from tests.test_semantic_search import (  # noqa: F401  (fixtures are used by name)
    FakeProvider,
    client,
    create_repository,
    database,
    index,
    provider,
)

API_KEY = "sk-test-embedding-key-that-must-never-leak"
QUESTION = "Where is GitHub authentication handled?"

BASE_FILES = {
    "app/github_oauth.py": [(1, "def github_callback():\n    exchange oauth token for login\n    return session\n")],
    "app/database.py": [(1, "engine = create_engine()\nsession query connection\n")],
    "app/ui.py": [(1, "render button with css layout\n")],
}


@pytest.fixture(autouse=True)
def embeddings_not_configured_by_default(monkeypatch):
    """Never reach a real provider; tests that need one request the `provider` fixture."""

    def unconfigured():
        raise EmbeddingNotConfiguredError("not configured")

    monkeypatch.setattr(repository_routes, "get_embedding_provider", unconfigured)


def ask(client, repository_id, headers, question=QUESTION, **options):
    return client.post(
        f"/api/v1/repositories/{repository_id}/context",
        json={"question": question, **options},
        headers=headers,
    )


def indexed_repository(client, database, provider, files=None, **kwargs):
    repository_id, headers = create_repository(database, files=BASE_FILES if files is None else files, **kwargs)
    assert index(client, repository_id, headers).status_code == 200
    return repository_id, headers


def add_analysis(factory, repository_id, **overrides):
    fields = dict(
        status="completed", project_type="web_application", languages=["python"], frameworks=["FastAPI"],
        package_managers=["pip"], dependencies=[{"name": "fastapi", "ecosystem": "pypi", "dev": False}],
        important_files=["README.md"], entry_points=[{"kind": "python_file", "name": "main.py", "path": "main.py"}],
        analysis_metadata={"manifests_skipped": []},
    )
    fields.update(overrides)
    with factory() as session:
        session.add(RepositoryAnalysis(repository_id=repository_id, **fields))
        session.commit()


def row_counts(factory):
    with factory() as session:
        return tuple(
            session.scalar(select(func.count()).select_from(model))
            for model in (RepositoryAnalysis, RepositoryFile, RepositoryChunk)
        )


# ------------------------------------------------------------------ term extraction


def test_terms_from_a_natural_language_question():
    assert extract_terms(QUESTION) == ["GitHub", "authentication"]


def test_quoted_and_backticked_spans_are_taken_verbatim_and_first():
    terms = extract_terms('Where is "auth flow" and `github_callback` used with oauth?')

    assert terms[:2] == ["auth flow", "github_callback"]
    assert "oauth" in terms


def test_single_quotes_count_but_apostrophes_do_not():
    assert extract_terms("show 'exchange_code' please")[0] == "exchange_code"
    assert extract_terms("What's the user's token?") == ["user", "token"]


def test_identifiers_snake_camel_dotted_and_paths_come_before_plain_words():
    terms = extract_terms("Check get_user_data, parseJSON, UserService.create_user, backend/app/main.py and sessions")

    assert terms == ["get_user_data", "parseJSON", "UserService.create_user", "backend/app/main.py", "Check"]


def test_stopwords_and_short_words_are_dropped():
    assert extract_terms(" ".join(sorted(word for word in STOPWORDS if len(word) >= 4))) == []
    assert extract_terms("How do we go to it?") == []
    assert extract_terms("Where is authentication handled") == ["authentication"]


def test_at_most_five_terms_are_returned_in_priority_order():
    question = "alpha_one beta_two gamma_three delta_four epsilon_five zeta_six eta_seven `quoted term`"

    terms = extract_terms(question)

    assert len(terms) == MAX_TERMS == 5
    assert terms[0] == "quoted term"
    assert terms[1:] == ["alpha_one", "beta_two", "gamma_three", "delta_four"]


def test_terms_are_deduplicated_case_insensitively_and_deterministic():
    assert extract_terms("Auth auth AUTH `auth`") == ["auth"]
    assert extract_terms(QUESTION) == extract_terms(QUESTION)
    assert extract_terms("") == [] and extract_terms("?!... ,,") == []


# ------------------------------------------------------------------ redaction


@pytest.mark.parametrize(
    ("secret", "text"),
    [
        ("AKIAABCDEFGHIJKLMNOP", 'aws = "AKIAABCDEFGHIJKLMNOP"'),
        ("ghp_" + "a" * 36, "token: ghp_" + "a" * 36),
        ("github_pat_" + "B" * 30, "GH=github_pat_" + "B" * 30),
        ("sk-proj-" + "c" * 30, 'client = OpenAI(api_key="sk-proj-' + "c" * 30 + '")'),
        ("xoxb-1234567890-abcdefghij", "slack = 'xoxb-1234567890-abcdefghij'"),
        ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnopqrstuvwx",
         "jwt = eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnopqrstuvwx"),
        ("abcdefghijklmnop1234", "Authorization: Bearer abcdefghijklmnop1234"),
        ("hunter22secret", 'password = "hunter22secret"'),
        ("hunter22secret", 'password: str = "hunter22secret"'),
        ("abcdef123456", '{"api_key": "abcdef123456"}'),
        ("s3cr3tvalue99", "client_secret: 's3cr3tvalue99'"),
        ("abc123def456ghi", "export API_KEY=abc123def456ghi"),
        ("hunter2secret", "password: hunter2secret"),
        ("dbpass123", "url = 'postgresql+psycopg://user:dbpass123@localhost/db'"),
    ],
)
def test_common_secret_formats_are_redacted(secret, text):
    redacted, count = redact_secrets(text)

    assert secret not in redacted
    assert REDACTION in redacted
    assert count >= 1


AWS_SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"


@pytest.mark.parametrize(
    ("secret", "text"),
    [
        (AWS_SECRET, f'AWS_SECRET_ACCESS_KEY = "{AWS_SECRET}"'),
        (AWS_SECRET, f"aws_secret_access_key={AWS_SECRET}"),
        ("Zm9vYmFyMTIzNDU2Nzg5", 'private_key = "Zm9vYmFyMTIzNDU2Nzg5"'),
        ("Zm9vYmFyMTIzNDU2Nzg5", "PRIVATE_KEY: 'Zm9vYmFyMTIzNDU2Nzg5'"),
        ("s3cretpass99", 'DB_PASS = "s3cretpass99"'),
        ("s3cretpass99", "MYSQL_PASS: s3cretpass99"),
        ("password", 'CACHE_URL = "redis://:password@host"'),
        ("redispw123456", "redis://:redispw123456@cache:6379/0"),
        ("hunter22secret", 'password = "hunter22secret"'),
        ("abcdefghijk", 'secret = "abcdefghijk"'),
        ("abcdefghijk", 'token = "abcdefghijk"'),
        ("abcdefghijk", 'api_key = "abcdefghijk"'),
        ("abcdefghij", "a" * 70 + '_token = "abcdefghij"'),
    ],
)
def test_secret_assignment_variants_are_redacted(secret, text):
    redacted, count = redact_secrets(text)

    assert secret not in redacted
    assert REDACTION in redacted
    assert count >= 1


@pytest.mark.parametrize(
    "code",
    [
        'bypass = "enabled123"',
        'compass = "north-west1"',
        'private_key_path = "keys/id.pem"',
        "pass_count = 3",
        "DB_PASS = get_db_pass()",
        "private_key = load_key()",
        'ssh_private_key = os.environ["KEY"]',
        "http://localhost:8080/path",
        "https://example.com:443/a",
        "ssh://git@github.com:org/repo.git",
    ],
)
def test_lookalike_names_and_urls_are_not_redacted(code):
    assert redact_secrets(code) == (code, 0)


@pytest.mark.parametrize(
    "text",
    ["a.b-" * 750 + "=x", "a-" * 1500 + "=x", "a." * 1500 + "=x", "a_b." * 750 + "token=", "a.b-" * 750],
)
def test_redaction_stays_fast_on_crafted_dotted_and_hyphenated_identifiers(text):
    # A snippet is at most 3,000 characters. The unbounded identifier prefix took ~1s here.
    started = time.perf_counter()
    redact_secrets(text[:3_000])

    assert time.perf_counter() - started < 0.5


def test_pem_private_keys_are_redacted_including_partial_ones_and_keep_line_count():
    full = "a = 1\n-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAxK7abcdefgh\nqrstuvwxyz0123456789ABCD\n-----END RSA PRIVATE KEY-----\nb = 2"
    cut_at_end = "x\n-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\nKcwggSjAgEAAoIBAQC7abcdefghijkl"
    cut_at_start = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC\nKcwggSjAgEAAoIBAQC7abcdefghijkl\n-----END PRIVATE KEY-----\ny = 3"

    for text, secret in ((full, "MIIEowIBAAKCAQEA"), (cut_at_end, "MIIEvQIBADANBg"), (cut_at_start, "KcwggSjAgEAAoIB")):
        redacted, count = redact_secrets(text)
        assert secret not in redacted and count >= 1
        assert redacted.count("\n") == text.count("\n")
    assert redact_secrets(full)[0].startswith("a = 1\n") and redact_secrets(full)[0].endswith("\nb = 2")


@pytest.mark.parametrize(
    "code",
    [
        "token_count = 5",
        "max_tokens = 4096",
        "token_limit = 100000",
        "tokens = tokens[0]",
        "password = get_password()",
        "password = user_password",
        "password_hash: Mapped[str | None] = mapped_column(String(512))",
        "auth_secret_key: SecretStr = Field(min_length=32)",
        'api_key = os.environ["API_KEY"]',
        "def validate_password(value: str) -> bool:",
        'token_type = "bearer"',
        "secret = secrets.token_hex(16)",
        'headers = {"Authorization": f"Bearer {access_token}"}',
        "results = [hit for hit in hits if hit.score > 0.5]",
        "http://localhost:8000/api/v1/health",
    ],
)
def test_normal_code_is_not_redacted(code):
    assert redact_secrets(code) == (code, 0)


@pytest.mark.parametrize(
    "path",
    [
        ".env", ".env.local", ".env.example", "backend/.env.production", "certs/server.pem", "keys/app.key",
        "id_rsa", "home/.ssh/id_rsa", "config/secrets.yaml", "app/client_secrets.json", "aws/credentials",
        "credentials.json", "gcp/service-account.json", "service-account-prod.json", ".npmrc",
        "frontend/.npmrc", ".pypirc", "secrets/db.yaml", "Config/SECRETS.yml",
    ],
)
def test_sensitive_paths_are_denied(path):
    assert is_sensitive_path(path)


@pytest.mark.parametrize(
    "path",
    ["app/auth/github_oauth.py", "src/monkey.py", "app/environment.py", "src/keyboard.ts", "README.md", "docs/setup.md"],
)
def test_ordinary_paths_are_allowed(path):
    assert not is_sensitive_path(path)


# ------------------------------------------------------------------ fusion and dedup


def exact_hit(path, start, end, text="exact text"):
    return CodeSearchHit(path, "python", start, end, text)


def semantic_hit(path, start, end, text="semantic chunk text", score=0.9):
    return SemanticHit(path, "python", start, end, text, score)


def test_reciprocal_rank_fusion_scores_and_order():
    fusion = fuse_hits(
        [("term", [exact_hit("a.py", 1, 3), exact_hit("b.py", 1, 3)])],
        [semantic_hit("b.py", 1, 3), semantic_hit("c.py", 1, 3)],
    )

    scores = {chunk.file_path: chunk.score for chunk in fusion.chunks}
    assert scores["b.py"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a.py"] == pytest.approx(1 / 61)
    assert scores["c.py"] == pytest.approx(1 / 62)
    assert [chunk.file_path for chunk in fusion.chunks] == ["b.py", "a.py", "c.py"]


def test_each_exact_term_is_its_own_ranked_list_that_adds_to_the_score():
    fusion = fuse_hits(
        [("alpha", [exact_hit("a.py", 1, 3)]), ("beta", [exact_hit("a.py", 1, 3), exact_hit("b.py", 1, 3)])],
        [],
    )

    top = fusion.chunks[0]
    assert top.file_path == "a.py"
    assert top.score == pytest.approx(2 / 61)
    assert top.matched_terms == ("alpha", "beta")
    assert fusion.chunks[1].matched_terms == ("beta",)


def test_wider_semantic_chunk_absorbs_the_narrower_exact_hit_and_sources_merge():
    fusion = fuse_hits(
        [("needle", [exact_hit("a.py", 8, 12, "narrow exact snippet")])],
        [semantic_hit("a.py", 1, 20, "whole semantic chunk")],
    )

    assert len(fusion.chunks) == 1
    chunk = fusion.chunks[0]
    assert (chunk.start_line, chunk.end_line, chunk.snippet) == (1, 20, "whole semantic chunk")
    assert chunk.sources == ("exact", "semantic")
    assert chunk.matched_terms == ("needle",)
    assert fusion.duplicates_merged == 1


def test_same_range_prefers_the_semantic_snippet():
    chunk = fuse_hits([("t", [exact_hit("a.py", 1, 5, "from exact")])], [semantic_hit("a.py", 1, 5, "from semantic")]).chunks[0]

    assert chunk.snippet == "from semantic"


def test_chained_overlaps_collapse_but_adjacent_and_cross_file_ranges_do_not():
    chained = fuse_hits([("t", [exact_hit("a.py", 1, 5), exact_hit("a.py", 5, 9), exact_hit("a.py", 9, 12)])], [])
    assert len(chained.chunks) == 1 and chained.duplicates_merged == 2

    separate = fuse_hits(
        [("t", [exact_hit("a.py", 1, 5), exact_hit("a.py", 6, 9), exact_hit("b.py", 1, 5)])], []
    )
    assert len(separate.chunks) == 3 and separate.duplicates_merged == 0


def test_fusion_is_deterministic_and_handles_no_results():
    exact = [("t", [exact_hit("b.py", 1, 3), exact_hit("a.py", 1, 3)])]
    semantic = [semantic_hit("c.py", 1, 3)]

    assert fuse_hits(exact, semantic) == fuse_hits(exact, semantic)
    assert fuse_hits([], []).chunks == [] and fuse_hits([("t", [])], []).duplicates_merged == 0


# ------------------------------------------------------------------ budgets and safety


def fused(path="a.py", start=1, text="line\n", score=0.5, end=None):
    return FusedChunk(path, "python", start, end or start + text.count("\n"), text, score, ("exact",), ("t",))


def test_at_most_two_chunks_per_file():
    chunks = [fused("a.py", start=n * 10, score=1 - n / 100) for n in range(4)] + [fused("b.py", score=0.1)]

    result = assemble_chunks(chunks, max_chunks=20, max_chars=60_000)

    assert MAX_CHUNKS_PER_FILE == 2
    assert [c.file_path for c in result.chunks] == ["a.py", "a.py", "b.py"]
    assert result.omitted == 2 and not result.truncated


def test_max_chunks_limits_the_total_and_flags_truncation():
    result = assemble_chunks([fused(f"f{n}.py") for n in range(5)], max_chunks=2, max_chars=60_000)

    assert len(result.chunks) == 2 and result.omitted == 3 and result.truncated


def test_long_snippets_are_cut_on_a_line_boundary_and_end_line_is_adjusted():
    lines = ["x" * 39] * 200
    text = "\n".join(lines) + "\n"

    chunk = assemble_chunks([fused(text=text, end=200)], max_chunks=5, max_chars=60_000).chunks[0]

    assert len(chunk.snippet) <= MAX_SNIPPET_CHARS
    assert chunk.truncated
    assert chunk.snippet == "\n".join(lines[: chunk.snippet.count("\n") + 1])
    assert chunk.end_line == chunk.start_line + chunk.snippet.count("\n") < 200


def test_a_single_overlong_line_is_hard_cut():
    chunk = assemble_chunks([fused(text="y" * 10_000)], max_chunks=5, max_chars=60_000).chunks[0]

    assert len(chunk.snippet) == MAX_SNIPPET_CHARS and chunk.truncated


def test_untruncated_chunks_keep_their_original_range_and_text():
    chunk = assemble_chunks([fused(start=7, text="a = 1\nb = 2\n", end=8)], max_chunks=5, max_chars=60_000).chunks[0]

    assert (chunk.start_line, chunk.end_line, chunk.snippet, chunk.truncated) == (7, 8, "a = 1\nb = 2", False)


def test_total_characters_never_exceed_the_budget():
    block = "\n".join(["github " + "x" * 22] * 50)
    chunks = [fused(f"{name}.py", text=block + "\n", score=1 - i / 10) for i, name in enumerate("abc")]

    result = assemble_chunks(chunks, max_chunks=20, max_chars=2_000)

    assert result.chars == sum(len(c.snippet) for c in result.chunks) <= 2_000
    assert [c.truncated for c in result.chunks] == [False, True]
    assert result.omitted == 1 and result.truncated
    second = result.chunks[1]
    assert second.snippet == "\n".join(block.split("\n")[: second.snippet.count("\n") + 1])


def test_a_sliver_of_budget_is_not_spent_on_a_useless_fragment():
    chunks = [fused("a.py", text="a" * 1_900, score=0.9), fused("b.py", text="b\n" * 400, score=0.8)]

    result = assemble_chunks(chunks, max_chunks=20, max_chars=2_000)

    assert [c.file_path for c in result.chunks] == ["a.py"]
    assert result.omitted == 1 and result.truncated


def test_sensitive_files_are_withheld_and_secrets_redacted_without_shifting_lines():
    pem = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBg\nKcwggSjAgEAAo\n-----END PRIVATE KEY-----"
    text = f'key = "AKIAABCDEFGHIJKLMNOP"\n{pem}\nlast = 1\n'
    chunks = [fused("app/config.py", text=text, score=0.9, end=6), fused("config/secrets.yaml", score=0.8), fused(".env", score=0.7)]

    result = assemble_chunks(chunks, max_chunks=20, max_chars=60_000)

    assert [c.file_path for c in result.chunks] == ["app/config.py"]
    assert result.withheld == 2 and result.redactions == 2
    chunk = result.chunks[0]
    assert "AKIA" not in chunk.snippet and "MIIEvQ" not in chunk.snippet
    assert chunk.end_line == 6 and chunk.snippet.count("\n") == 5


def test_empty_input_gives_an_empty_assembly():
    result = assemble_chunks([], max_chunks=5, max_chars=2_000)

    assert result.chunks == [] and (result.chars, result.omitted, result.withheld) == (0, 0, 0)


def test_truncate_at_line_edge_cases():
    assert truncate_at_line("abc", 3) == ("abc", False)
    assert truncate_at_line("ab\ncd", 3) == ("ab", True)
    assert truncate_at_line("ab\ncd\nef", 5) == ("ab\ncd", True)
    assert truncate_at_line("abcdef", 3) == ("abc", True)


def test_layout_counts_top_level_directories_and_skips_sensitive_paths():
    paths = ["a/x.py", "a/y.py", "b/z.py", "README.py", "secrets/db.yaml", ".env"]

    assert build_layout(paths) == [
        {"path": "a/", "files": 2}, {"path": "(root)", "files": 1}, {"path": "b/", "files": 1},
    ]
    many = [f"d{n:02d}/f.py" for n in range(40)]
    assert len(build_layout(many)) == 30


# ------------------------------------------------------------------ endpoint: happy path


def test_context_combines_analysis_exact_and_semantic_sources(client, database, provider):
    repository_id, headers = indexed_repository(client, database, provider)
    add_analysis(database, repository_id)

    response = ask(client, repository_id, headers)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"repository_id", "question", "project", "relevant_chunks", "relevant_files", "retrieval"}
    assert body["repository_id"] == str(repository_id) and body["question"] == QUESTION
    top = body["relevant_chunks"][0]
    assert top["file_path"] == "app/github_oauth.py" and top["language"] == "python"
    assert (top["start_line"], top["end_line"]) == (1, 3)
    assert "github_callback" in top["snippet"]
    assert top["sources"] == ["exact", "semantic"] and top["matched_terms"] == ["GitHub"]
    assert top["truncated"] is False and top["score"] == pytest.approx(2 / 61, abs=1e-6)
    assert set(top) == {"file_path", "language", "start_line", "end_line", "snippet", "score", "sources", "matched_terms", "truncated"}
    assert body["relevant_files"][0] == {
        "file_path": "app/github_oauth.py", "language": "python", "chunk_count": 1, "sources": ["exact", "semantic"],
    }
    retrieval = body["retrieval"]
    assert retrieval["analysis"] == "included"
    assert retrieval["exact"] == {"status": "used", "terms": ["GitHub", "authentication"], "hits": 1}
    assert retrieval["semantic"]["status"] == "used"
    assert retrieval["returned"] == len(body["relevant_chunks"]) and retrieval["truncated"] is False
    assert retrieval["chars"] == sum(len(chunk["snippet"]) for chunk in body["relevant_chunks"])
    assert retrieval["max_chars"] == 24_000 and retrieval["withheld_chunks"] == 0


def test_project_analysis_section_is_included_with_layout(client, database, provider):
    repository_id, headers = indexed_repository(client, database, provider)
    add_analysis(database, repository_id)

    project = ask(client, repository_id, headers).json()["project"]

    assert project == {
        "status": "completed", "project_type": "web_application", "languages": ["python"],
        "frameworks": ["FastAPI"], "package_managers": ["pip"],
        "dependencies": [{"name": "fastapi", "ecosystem": "pypi", "dev": False}], "dependencies_total": 1,
        "important_files": ["README.md"],
        "entry_points": [{"kind": "python_file", "name": "main.py", "path": "main.py"}],
        "layout": [{"path": "app/", "files": 3}],
    }


def test_project_is_null_without_analysis_and_skipped_when_disabled(client, database, provider):
    repository_id, headers = indexed_repository(client, database, provider)

    absent = ask(client, repository_id, headers).json()
    assert absent["project"] is None and absent["retrieval"]["analysis"] == "not_available"

    add_analysis(database, repository_id)
    skipped = ask(client, repository_id, headers, include_analysis=False).json()
    assert skipped["project"] is None and skipped["retrieval"]["analysis"] == "skipped"


def test_project_lists_are_capped(client, database, provider):
    repository_id, headers = indexed_repository(client, database, provider)
    add_analysis(
        database, repository_id,
        dependencies=[{"name": f"pkg{n:03d}", "ecosystem": "pypi", "dev": False} for n in range(60)],
        important_files=[f"f{n}.md" for n in range(30)],
        entry_points=[{"kind": "python_file", "name": f"m{n}.py", "path": f"m{n}.py"} for n in range(15)],
    )

    project = ask(client, repository_id, headers).json()["project"]

    assert len(project["dependencies"]) == 40 and project["dependencies_total"] == 60
    assert project["dependencies"][0]["name"] == "pkg000"
    assert len(project["important_files"]) == 20 and len(project["entry_points"]) == 10


def test_layout_is_capped_ordered_and_hides_sensitive_directories(client, database, provider):
    files = {f"d{n:02d}/f.py": [(1, "x = 1\n")] for n in range(35)}
    files.update({"d00/g.py": [(1, "y = 1\n")], "d00/h.py": [(1, "z = 1\n")], "README.py": [(1, "r = 1\n")], "secrets/db.yaml": [(1, "p: 1\n")]})
    repository_id, headers = create_repository(database, files=files)
    add_analysis(database, repository_id)

    layout = ask(client, repository_id, headers, include_exact=False, include_semantic=False).json()["project"]["layout"]

    assert len(layout) == 30 and layout[0] == {"path": "d00/", "files": 3}
    assert all(entry["path"] != "secrets/" for entry in layout)


def test_context_works_end_to_end_after_a_real_scan(client, database, provider, monkeypatch):
    repository_id, headers = create_empty_repository(database)
    use_github(monkeypatch, FakeGitHubClient({
        "app/github_oauth.py": "def github_callback():\n    exchange oauth token for login\n    return session\n",
        "app/database.py": "engine = create_engine()\nsession query connection\n",
        "requirements.txt": "fastapi>=0.115\n",
    }))
    assert scan(client, repository_id, headers).status_code == 200

    body = ask(client, repository_id, headers).json()

    assert body["project"]["project_type"] == "backend_service" and body["project"]["frameworks"] == ["FastAPI"]
    assert body["project"]["layout"] == [{"path": "app/", "files": 2}]
    top = body["relevant_chunks"][0]
    assert top["file_path"] == "app/github_oauth.py" and top["sources"] == ["exact", "semantic"]


# ------------------------------------------------------------------ retrieval modes


def test_semantic_can_be_disabled_and_the_provider_is_not_called(client, database, provider):
    repository_id, headers = indexed_repository(client, database, provider)
    calls_before = len(provider.calls)

    body = ask(client, repository_id, headers, include_semantic=False).json()

    assert len(provider.calls) == calls_before
    assert body["retrieval"]["semantic"] == {"status": "skipped", "hits": 0}
    assert [c["file_path"] for c in body["relevant_chunks"]] == ["app/github_oauth.py"]
    assert body["relevant_chunks"][0]["sources"] == ["exact"]


def test_exact_can_be_disabled(client, database, provider):
    repository_id, headers = indexed_repository(client, database, provider)

    body = ask(client, repository_id, headers, include_exact=False).json()

    assert body["retrieval"]["exact"] == {"status": "skipped", "terms": [], "hits": 0}
    assert all(c["sources"] == ["semantic"] and c["matched_terms"] == [] for c in body["relevant_chunks"])
    assert body["relevant_chunks"][0]["file_path"] == "app/github_oauth.py"


def test_both_sources_disabled_still_returns_the_project(client, database, provider):
    repository_id, headers = indexed_repository(client, database, provider)
    add_analysis(database, repository_id)

    response = ask(client, repository_id, headers, include_exact=False, include_semantic=False)

    assert response.status_code == 200
    assert response.json()["relevant_chunks"] == [] and response.json()["project"] is not None


def test_no_extractable_terms_falls_back_to_semantic_only(client, database, provider):
    repository_id, headers = indexed_repository(client, database, provider)

    body = ask(client, repository_id, headers, question="What is this?").json()

    assert body["retrieval"]["exact"] == {"status": "no_terms", "terms": [], "hits": 0}
    assert body["retrieval"]["semantic"]["status"] == "used"
    assert body["relevant_chunks"] and all(c["sources"] == ["semantic"] for c in body["relevant_chunks"])


def test_no_embeddings_still_returns_exact_results(client, database, provider):
    repository_id, headers = create_repository(database, files=BASE_FILES)
    calls_before = len(provider.calls)

    response = ask(client, repository_id, headers)

    assert response.status_code == 200
    body = response.json()
    assert body["retrieval"]["semantic"] == {"status": "no_embeddings", "hits": 0}
    assert [c["file_path"] for c in body["relevant_chunks"]] == ["app/github_oauth.py"]
    assert body["relevant_chunks"][0]["sources"] == ["exact"]
    assert len(provider.calls) == calls_before


def test_unconfigured_provider_still_returns_exact_results(client, database, provider, monkeypatch):
    repository_id, headers = indexed_repository(client, database, provider)

    def unconfigured():
        raise EmbeddingNotConfiguredError("EMBEDDING_API_KEY missing")

    monkeypatch.setattr(repository_routes, "get_embedding_provider", unconfigured)
    response = ask(client, repository_id, headers)

    assert response.status_code == 200
    assert response.json()["retrieval"]["semantic"]["status"] == "not_configured"
    assert response.json()["relevant_chunks"][0]["sources"] == ["exact"]
    assert "EMBEDDING_API_KEY" not in response.text


@pytest.mark.parametrize("error", [EmbeddingError(f"boom {API_KEY} raw provider body"), RuntimeError(f"kaboom {API_KEY}")])
def test_provider_failure_never_makes_the_endpoint_fail_or_leak(client, database, provider, monkeypatch, caplog, error):
    repository_id, headers = indexed_repository(client, database, provider)

    class FailingProvider:
        model = "fake"

        def embed(self, texts):
            raise error

    monkeypatch.setattr(repository_routes, "get_embedding_provider", lambda: FailingProvider())
    caplog.set_level(logging.DEBUG)

    response = ask(client, repository_id, headers)

    assert response.status_code == 200
    assert response.json()["retrieval"]["semantic"] == {"status": "failed", "hits": 0}
    assert response.json()["relevant_chunks"][0]["sources"] == ["exact"]
    assert API_KEY not in response.text + caplog.text and "raw provider body" not in response.text + caplog.text


def test_semantic_failure_does_not_poison_the_session(client, database, provider, monkeypatch):
    repository_id, headers = indexed_repository(client, database, provider)
    add_analysis(database, repository_id)
    from app.scanner import semantic as semantic_module

    def broken(*_args, **_kwargs):
        raise RuntimeError("vector query failed")

    monkeypatch.setattr(semantic_module, "find_similar_chunks", broken)

    body = ask(client, repository_id, headers).json()

    assert body["retrieval"]["semantic"]["status"] == "failed"
    assert body["project"] is not None and body["relevant_chunks"]


# ------------------------------------------------------------------ endpoint: budgets


def test_per_file_limit_applies_through_the_endpoint(client, database, provider):
    files = {
        "app/many.py": [(1, "needle a\n"), (2, "needle b\n"), (3, "needle c\n")],
        "app/other.py": [(1, "needle d\n")],
    }
    repository_id, headers = create_repository(database, files=files)

    body = ask(client, repository_id, headers, question="Where is needle used?", include_semantic=False).json()

    paths = [c["file_path"] for c in body["relevant_chunks"]]
    assert paths.count("app/many.py") == 2 and paths.count("app/other.py") == 1
    assert body["retrieval"]["omitted_chunks"] == 1 and body["retrieval"]["candidates"] == 4


def test_max_chunks_is_respected(client, database, provider):
    repository_id, headers = create_repository(database, files={f"f{n}.py": [(1, "needle x\n")] for n in range(5)})

    body = ask(client, repository_id, headers, question="needle", include_semantic=False, max_chunks=2).json()

    assert len(body["relevant_chunks"]) == 2 and body["retrieval"]["omitted_chunks"] == 3
    assert body["retrieval"]["truncated"] is True


def test_long_chunks_are_truncated_on_line_boundaries(client, database, provider):
    line = "github " + "x" * 33
    files = {"app/big.py": [(1, "\n".join([line] * 200) + "\n")]}
    repository_id, headers = indexed_repository(client, database, provider, files=files)

    body = ask(client, repository_id, headers, question="github token", include_exact=False).json()

    chunk = body["relevant_chunks"][0]
    assert len(chunk["snippet"]) <= 3_000 and chunk["truncated"] is True
    assert all(text == line for text in chunk["snippet"].split("\n"))
    assert chunk["end_line"] == chunk["start_line"] + chunk["snippet"].count("\n") < 200
    assert body["retrieval"]["truncated"] is True


def test_overall_character_budget_is_enforced(client, database, provider):
    block = "\n".join(["github " + "x" * 22] * 50) + "\n"
    repository_id, headers = indexed_repository(client, database, provider, files={f"app/{n}.py": [(1, block)] for n in "abc"})

    body = ask(client, repository_id, headers, question="github token", include_exact=False, max_chars=2_000).json()

    total = sum(len(c["snippet"]) for c in body["relevant_chunks"])
    assert total == body["retrieval"]["chars"] <= 2_000
    assert len(body["relevant_chunks"]) == 2 and body["relevant_chunks"][1]["truncated"] is True
    assert body["retrieval"]["truncated"] is True


def test_duplicates_are_merged_and_counted(client, database, provider):
    files = {"app/x.py": [(1, "".join(f"filler {n}\n" for n in range(1, 10)) + "github_callback here\n" + "".join(f"more {n}\n" for n in range(11, 20)))]}
    repository_id, headers = indexed_repository(client, database, provider, files=files)

    body = ask(client, repository_id, headers, question="github_callback").json()

    assert len(body["relevant_chunks"]) == 1
    chunk = body["relevant_chunks"][0]
    assert (chunk["start_line"], chunk["end_line"]) == (1, 19) and chunk["sources"] == ["exact", "semantic"]
    assert body["retrieval"]["duplicates_merged"] == 1


# ------------------------------------------------------------------ endpoint: security


SECRET_LINES = [
    "# github token credentials for login",
    'AWS_KEY = "AKIAABCDEFGHIJKLMNOP"',
    'password = "hunter22secret"',
    'GITHUB = "ghp_' + "a" * 36 + '"',
    'OPENAI = "sk-proj-' + "b" * 30 + '"',
    'SLACK = "xoxb-1234567890-abcdefghij"',
    'JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnopqrstuvwx"',
    'headers = {"Authorization": "Bearer abcdefghijklmnop1234"}',
    'DATABASE_URL = "postgresql://user:dbpass123@host/db"',
    "-----BEGIN RSA PRIVATE KEY-----",
    "MIIEowIBAAKCAQEAxK7abcdefghijklmnop",
    "-----END RSA PRIVATE KEY-----",
]
SECRET_VALUES = [
    "AKIAABCDEFGHIJKLMNOP", "hunter22secret", "ghp_" + "a" * 36, "sk-proj-" + "b" * 30, "xoxb-1234567890-abcdefghij",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0", "abcdefghijklmnop1234", "dbpass123", "MIIEowIBAAKCAQEA",
]
WITHHELD_VALUES = ["supersecretvalue1", "abcdef123456", "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC", "secrets.yaml", ".env.local", "server.pem"]
SECRET_QUESTION = "github token credentials login"


def secret_repository(client, database, provider):
    files = {
        "app/config.py": [(1, "\n".join(SECRET_LINES) + "\n")],
        "config/secrets.yaml": [(1, "github token credentials login\npassword: supersecretvalue1\n")],
        ".env.local": [(1, "github token credentials login\nKEY=abcdef123456\n")],
        "certs/server.pem": [(1, "github token credentials login\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n")],
    }
    return indexed_repository(client, database, provider, files=files)


@pytest.mark.parametrize(
    ("options", "minimum_redactions"),
    [({}, 8), ({"include_exact": False}, 8), ({"include_semantic": False}, 2)],
)
def test_secrets_and_sensitive_files_never_appear_in_the_response(client, database, provider, options, minimum_redactions):
    repository_id, headers = secret_repository(client, database, provider)
    add_analysis(database, repository_id, important_files=[".env.example", "README.md"])

    response = ask(client, repository_id, headers, question=SECRET_QUESTION, **options)

    assert response.status_code == 200
    body = response.json()
    for leaked in SECRET_VALUES + WITHHELD_VALUES:
        assert leaked not in response.text
    assert [c["file_path"] for c in body["relevant_chunks"]] == ["app/config.py"]
    assert body["retrieval"]["withheld_chunks"] == 3
    assert body["retrieval"]["redactions"] >= minimum_redactions
    assert REDACTION in body["relevant_chunks"][0]["snippet"]


def test_redaction_keeps_line_numbers_valid(client, database, provider):
    repository_id, headers = secret_repository(client, database, provider)

    chunk = ask(client, repository_id, headers, question=SECRET_QUESTION, include_exact=False).json()["relevant_chunks"][0]

    assert (chunk["start_line"], chunk["end_line"]) == (1, len(SECRET_LINES))
    assert chunk["snippet"].count("\n") + 1 == len(SECRET_LINES)


def test_vectors_keys_and_tokens_are_never_returned(client, database, provider):
    repository_id, headers = indexed_repository(client, database, provider)
    add_analysis(database, repository_id)

    response = ask(client, repository_id, headers)

    assert response.status_code == 200
    assert '"embedding"' not in response.text and "access_token" not in response.text
    assert not re.search(r"\[\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+", response.text)
    assert API_KEY not in response.text


def test_the_question_is_never_logged(client, database, provider, monkeypatch, caplog):
    repository_id, headers = indexed_repository(client, database, provider)
    question = "Where does the quokka-unique-question-marker live?"
    caplog.set_level(logging.DEBUG)

    assert ask(client, repository_id, headers, question=question).status_code == 200

    class FailingProvider:
        model = "fake"

        def embed(self, texts):
            raise EmbeddingError("down")

    monkeypatch.setattr(repository_routes, "get_embedding_provider", lambda: FailingProvider())
    assert ask(client, repository_id, headers, question=question).status_code == 200
    assert "quokka" not in caplog.text


def test_the_context_is_never_persisted(client, database, provider):
    repository_id, headers = indexed_repository(client, database, provider)
    add_analysis(database, repository_id)
    with database() as session:
        before = (row_counts(database), session.scalars(select(RepositoryAnalysis.updated_at)).one())

    for options in ({}, {"include_semantic": False}, {"include_analysis": False}):
        assert ask(client, repository_id, headers, **options).status_code == 200

    with database() as session:
        assert (row_counts(database), session.scalars(select(RepositoryAnalysis.updated_at)).one()) == before


def test_output_is_deterministic(client, database, provider):
    repository_id, headers = indexed_repository(client, database, provider)
    add_analysis(database, repository_id)

    first = ask(client, repository_id, headers)
    second = ask(client, repository_id, headers)

    assert first.status_code == 200 and first.json() == second.json()


# ------------------------------------------------------------------ endpoint: access control and validation


def test_unscanned_repository_returns_conflict(client, database, provider):
    repository_id, headers = create_repository(database, files={})

    response = ask(client, repository_id, headers)

    assert response.status_code == 409
    assert "scan" in response.json()["error"]["message"].lower()


def test_authentication_is_required(client, database, provider):
    repository_id, _ = indexed_repository(client, database, provider)

    assert ask(client, repository_id, {}).status_code == 401
    assert ask(client, repository_id, {"Authorization": "Bearer not-a-real-token"}).status_code == 401
    unauthenticated_invalid = client.post(f"/api/v1/repositories/{repository_id}/context", json={"question": ""})
    assert unauthenticated_invalid.status_code == 401


def test_unknown_repository_returns_not_found(client, database, provider):
    _, headers = create_repository(database, files=BASE_FILES)

    response = ask(client, uuid.uuid4(), headers)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "RESOURCE_NOT_FOUND"


def test_another_users_repository_is_never_readable(client, database, provider):
    other_id, _ = indexed_repository(client, database, provider, email="other@example.com")
    add_analysis(database, other_id)
    _, my_headers = create_repository(database, email="me@example.com", files={"mine.py": [(1, "nothing here\n")]})
    calls_before = len(provider.calls)

    response = ask(client, other_id, my_headers)

    assert response.status_code == 404
    for leaked in ("github_callback", "app/github_oauth.py", "web_application", "FastAPI"):
        assert leaked not in response.text
    assert len(provider.calls) == calls_before


def test_results_never_include_another_repositorys_code(client, database, provider):
    indexed_repository(client, database, provider, email="other@example.com", files={"theirs.py": [(1, "github token login\n")]})
    repository_id, headers = indexed_repository(client, database, provider, email="me@example.com", files={"mine.py": [(1, "github token login\n")]})

    body = ask(client, repository_id, headers, question="github token login").json()

    assert {c["file_path"] for c in body["relevant_chunks"]} == {"mine.py"}


@pytest.mark.parametrize(
    "body",
    [
        {}, {"question": ""}, {"question": "   "}, {"question": "\t\n "}, {"question": "x" * 1001}, {"question": 5},
        {"question": "ok", "max_chunks": 0}, {"question": "ok", "max_chunks": 21}, {"question": "ok", "max_chunks": "many"},
        {"question": "ok", "max_chars": 1_999}, {"question": "ok", "max_chars": 60_001},
        {"question": "ok", "include_exact": "maybe"}, {"question": "ok", "unexpected": 1},
    ],
)
def test_invalid_requests_are_rejected(client, database, provider, body):
    repository_id, headers = indexed_repository(client, database, provider)
    calls_before = len(provider.calls)

    response = client.post(f"/api/v1/repositories/{repository_id}/context", json=body, headers=headers)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert len(provider.calls) == calls_before


@pytest.mark.parametrize(
    "question",
    ["find foo_bar\u0000baz", "\u0000", "find `foo\u0000bar` please", "where is authentication\u0000flow"],
)
def test_nul_characters_in_the_question_are_rejected_with_a_clean_422(client, database, provider, question):
    repository_id, headers = indexed_repository(client, database, provider)
    calls_before = len(provider.calls)

    response = ask(client, repository_id, headers, question=question)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "foo_bar" not in response.text and "authentication" not in response.text
    assert len(provider.calls) == calls_before


def test_context_request_rejects_nul_but_accepts_normal_unicode():
    with pytest.raises(ValidationError):
        ContextRequest(question="a\u0000b")

    for question in ("naïve café 日本語 🚀", "Où est déjà_vu ?", "tab\tand\nnewline"):
        assert ContextRequest(question=question).question == question.strip()


def test_unicode_questions_and_code_are_still_answered(client, database, provider):
    files = {"app/x.py": [(1, "def déjà_vu():\n    return 1\n")]}
    repository_id, headers = indexed_repository(client, database, provider, files=files)
    question = "Où est `déjà_vu` défini ? 日本語 🚀"

    response = ask(client, repository_id, headers, question=question)

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == question
    assert body["relevant_chunks"][0]["file_path"] == "app/x.py"
    assert body["relevant_chunks"][0]["matched_terms"] == ["déjà_vu"]


def test_invalid_repository_id_and_malformed_body_are_rejected(client, database, provider):
    _, headers = indexed_repository(client, database, provider)

    bad_id = client.post("/api/v1/repositories/not-a-uuid/context", json={"question": "ok"}, headers=headers)
    not_json = client.post(f"/api/v1/repositories/{uuid.uuid4()}/context", content="not json",
                           headers={**headers, "Content-Type": "application/json"})

    assert bad_id.status_code == 422 and not_json.status_code == 422


@pytest.mark.parametrize("options", [{"max_chunks": 1, "max_chars": 2_000}, {"max_chunks": 20, "max_chars": 60_000}])
def test_documented_limits_are_accepted(client, database, provider, options):
    repository_id, headers = indexed_repository(client, database, provider)

    assert ask(client, repository_id, headers, question="a" * 1_000, **options).status_code == 200


def test_the_question_is_trimmed_and_echoed(client, database, provider):
    repository_id, headers = indexed_repository(client, database, provider)

    body = ask(client, repository_id, headers, question=f"   {QUESTION}  \n").json()

    assert body["question"] == QUESTION


def test_existing_endpoints_are_unaffected(client, database, provider):
    repository_id, headers = indexed_repository(client, database, provider)
    add_analysis(database, repository_id)

    search = client.get(f"/api/v1/repositories/{repository_id}/search", params={"query": "github"}, headers=headers)
    semantic = client.get(f"/api/v1/repositories/{repository_id}/semantic-search", params={"query": "github"}, headers=headers)
    analysis = client.get(f"/api/v1/repositories/{repository_id}/analysis", headers=headers)

    assert (search.status_code, semantic.status_code, analysis.status_code) == (200, 200, 200)
    assert json.dumps(search.json()) and json.dumps(analysis.json())
