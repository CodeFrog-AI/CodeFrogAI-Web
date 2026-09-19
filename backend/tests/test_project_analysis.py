"""Tests for static repository project analysis and its scan integration."""

import json
import logging

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.analyzer import service as analyzer_service
from app.analyzer.manifests import ManifestParseError, parse_manifest
from app.api.routes import repositories as repository_routes
from app.db.base import Base
from app.db.database import get_db
from app.db.models import RepositoryAnalysis, RepositoryChunk
from app.integrations.github.contents import GitHubContentError
from main import app
from tests.test_repository_scan import FakeGitHubClient, create_repository, scan, use_github


@pytest.fixture(autouse=True)
def no_real_embedding_provider(monkeypatch):
    """Where embeddings exist, keep these scans away from any real provider."""

    if hasattr(repository_routes, "get_embedding_provider"):
        from app.embeddings.provider import EmbeddingNotConfiguredError

        def unconfigured():
            raise EmbeddingNotConfiguredError("not configured")

        monkeypatch.setattr(repository_routes, "get_embedding_provider", unconfigured)


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


def package_json(**fields) -> str:
    return json.dumps(fields)


FULLSTACK = {
    "README.md": "# Demo\n",
    "Dockerfile": "FROM python:3.13\n",
    "docker-compose.yml": "services: {}\n",
    ".env": "SECRET=super-secret-env-value\n",
    ".env.example": "SECRET=changeme\n",
    ".github/workflows/ci.yml": "name: ci\n",
    "node_modules/left-pad/package.json": package_json(dependencies={"ignored-package": "1"}),
    "backend/main.py": "print('api')\n",
    "backend/app/routes.py": "x = 1\n",
    "backend/requirements.txt": (
        "fastapi>=0.115,<1.0\nuvicorn[standard]>=0.30\n# a comment\n-r other.txt\n"
        "SQLAlchemy==2.0.1 ; python_version > '3'\ngit+https://github.com/x/y.git#egg=y\n"
    ),
    "frontend/package.json": package_json(
        name="web",
        dependencies={"next": "^15", "react": "^19", "react-dom": "^19"},
        devDependencies={"typescript": "^5", "eslint": "^9"},
        scripts={"dev": "next dev", "build": "next build", "start": "next start",
                 "deploy": "curl -H 'Authorization: sk-secret-deploy-token' https://example.test"},
    ),
    "frontend/package-lock.json": "{}",
    "frontend/tsconfig.json": "{}",
    "frontend/next.config.ts": "export default {};\n",
    "frontend/eslint.config.mjs": "export default [];\n",
    "frontend/app/page.tsx": "export default function Page() { return null }\n",
}


def analyze(client, database, monkeypatch, files, **repository_kwargs):
    """Scan a repository made of `files` and return (repository_id, headers, analysis JSON)."""

    repository_id, headers = create_repository(database, **repository_kwargs)
    use_github(monkeypatch, FakeGitHubClient(files))
    assert scan(client, repository_id, headers).status_code == 200
    response = get_analysis(client, repository_id, headers)
    assert response.status_code == 200
    return repository_id, headers, response.json()


def get_analysis(client, repository_id, headers):
    return client.get(f"/api/v1/repositories/{repository_id}/analysis", headers=headers)


def analysis_rows(factory):
    with factory() as session:
        return session.scalars(select(RepositoryAnalysis)).all()


# ------------------------------------------------------------ full project analysis


def test_full_stack_project_is_analyzed(client, database, monkeypatch):
    fake = FakeGitHubClient(FULLSTACK)
    repository_id, headers = create_repository(database)
    use_github(monkeypatch, fake)
    assert scan(client, repository_id, headers).status_code == 200

    response = get_analysis(client, repository_id, headers)

    assert response.status_code == 200
    body = response.json()
    assert body["repository_id"] == str(repository_id)
    assert body["status"] == "completed"
    assert body["project_type"] == "web_application"
    assert body["languages"] == ["python", "typescript"]
    assert body["frameworks"] == ["FastAPI", "Next.js", "React"]
    assert body["package_managers"] == ["npm", "pip"]
    assert body["skipped_manifests"] == []
    assert {(d["ecosystem"], d["name"], d["dev"]) for d in body["dependencies"]} == {
        ("npm", "next", False), ("npm", "react", False), ("npm", "react-dom", False),
        ("npm", "typescript", True), ("npm", "eslint", True),
        ("pypi", "fastapi", False), ("pypi", "uvicorn", False), ("pypi", "sqlalchemy", False),
    }
    assert all(set(d) == {"name", "ecosystem", "dev"} for d in body["dependencies"])
    assert body["important_files"] == [
        ".env.example", "Dockerfile", "README.md", "docker-compose.yml",
        "backend/requirements.txt", "frontend/eslint.config.mjs", "frontend/next.config.ts",
        "frontend/package.json", "frontend/tsconfig.json", ".github/workflows/ci.yml",
    ]
    assert body["entry_points"] == [
        {"kind": "python_file", "name": "main.py", "path": "backend/main.py"},
        {"kind": "npm_script", "name": "dev", "path": "frontend/package.json"},
        {"kind": "npm_script", "name": "start", "path": "frontend/package.json"},
    ]


def test_analysis_never_exposes_contents_secrets_or_ignored_files(client, database, monkeypatch):
    _, _, body = analyze(client, database, monkeypatch, FULLSTACK)
    text = json.dumps(body)

    for secret in ("sk-secret-deploy-token", "super-secret-env-value", "curl -H", "next dev", "^15", "ignored-package"):
        assert secret not in text
    assert ".env" not in body["important_files"]
    assert "access_token" not in text and "embedding" not in text


def test_only_manifests_are_fetched_and_indexed_files_are_not_downloaded_again(client, database, monkeypatch):
    fake = FakeGitHubClient(FULLSTACK)
    repository_id, headers = create_repository(database)
    use_github(monkeypatch, fake)

    scan(client, repository_id, headers)

    assert fake.downloads.count("frontend/package.json") == 1
    assert fake.downloads.count("backend/requirements.txt") == 1
    for never_needed in ("README.md", "Dockerfile", "frontend/tsconfig.json", "node_modules/left-pad/package.json"):
        assert never_needed not in fake.downloads

    rescan = use_github(monkeypatch, FakeGitHubClient(FULLSTACK))
    scan(client, repository_id, headers)
    assert rescan.downloads == ["backend/requirements.txt"]


def test_analysis_is_deterministic_and_stored_once_per_repository(client, database, monkeypatch):
    repository_id, headers, first = analyze(client, database, monkeypatch, FULLSTACK)
    with database() as session:
        first_id = session.scalars(select(RepositoryAnalysis)).one().id

    use_github(monkeypatch, FakeGitHubClient(FULLSTACK))
    scan(client, repository_id, headers)
    second = get_analysis(client, repository_id, headers).json()

    assert {k: v for k, v in first.items() if k != "updated_at"} == {k: v for k, v in second.items() if k != "updated_at"}
    rows = analysis_rows(database)
    assert len(rows) == 1 and rows[0].id == first_id


def test_analysis_is_persisted_with_structured_columns(client, database, monkeypatch):
    repository_id, _, body = analyze(client, database, monkeypatch, FULLSTACK)

    with database() as session:
        row = session.scalars(select(RepositoryAnalysis)).one()
        assert row.repository_id == repository_id
        assert (row.status, row.project_type) == ("completed", "web_application")
        assert row.languages == body["languages"] and row.frameworks == body["frameworks"]
        assert row.package_managers == ["npm", "pip"]
        assert row.created_at is not None and row.updated_at is not None
        assert row.analysis_metadata["analyzer_version"] == 1
        assert row.analysis_metadata["manifests_analyzed"] == ["backend/requirements.txt", "frontend/package.json"]
        assert session.get(type(row.repository), repository_id).analysis.id == row.id


def test_rescan_updates_the_same_record_and_removes_stale_information(client, database, monkeypatch):
    repository_id, headers, before = analyze(client, database, monkeypatch, FULLSTACK)
    with database() as session:
        row_id = session.scalars(select(RepositoryAnalysis)).one().id
    assert "Next.js" in before["frameworks"]

    changed = {p: c for p, c in FULLSTACK.items() if not p.startswith("frontend/")}
    changed["backend/requirements.txt"] = "fastapi\ndjango>=5\n"
    use_github(monkeypatch, FakeGitHubClient(changed))
    assert scan(client, repository_id, headers).status_code == 200
    after = get_analysis(client, repository_id, headers).json()

    assert after["frameworks"] == ["Django", "FastAPI"]
    assert after["package_managers"] == ["pip"]
    assert after["languages"] == ["python"]
    assert after["project_type"] == "backend_service"
    assert {d["name"] for d in after["dependencies"]} == {"fastapi", "django"}
    assert all(e["kind"] == "python_file" for e in after["entry_points"])
    assert not any(path.startswith("frontend/") for path in after["important_files"])
    rows = analysis_rows(database)
    assert len(rows) == 1 and rows[0].id == row_id


# ------------------------------------------------------------ detection rules


@pytest.mark.parametrize(
    ("files", "languages"),
    [
        ({"a.py": "x=1\n", "b.py": "y=1\n", "c.js": "z\n", "s.css": "a{}\n", "d.json": "{}", "e.yaml": "a: 1\n"},
         ["python", "css", "javascript"]),
        ({"Main.java": "class A {}\n", "run.sh": "echo\n", "q.sql": "select 1;\n"}, ["java", "shell", "sql"]),
        ({"README.md": "docs"}, []),
    ],
)
def test_languages_come_from_indexed_files_and_exclude_config_formats(client, database, monkeypatch, files, languages):
    _, _, body = analyze(client, database, monkeypatch, files)

    assert body["languages"] == languages


PYPROJECT_POETRY = """
[project]
name = "x"
dependencies = ["FastAPI>=0.1", "uvicorn[standard]; python_version > '3.8'", "python_dotenv"]
[project.optional-dependencies]
test = ["pytest"]
[tool.poetry.dependencies]
python = "^3.11"
requests = "^2"
[tool.poetry.group.dev.dependencies]
black = "*"
"""


@pytest.mark.parametrize(
    ("files", "frameworks", "project_type", "managers"),
    [
        ({"package.json": package_json(dependencies={"express": "4"})}, ["Express"], "backend_service", ["npm"]),
        ({"package.json": package_json(dependencies={"react": "19"}), "yarn.lock": ""}, ["React"], "frontend_application", ["yarn"]),
        ({"package.json": package_json(dependencies={"@nestjs/core": "10"}), "pnpm-lock.yaml": ""}, ["NestJS"], "backend_service", ["pnpm"]),
        ({"package.json": package_json(dependencies={"next": "15", "react": "19"})}, ["Next.js", "React"], "web_application", ["npm"]),
        ({"package.json": package_json(dependencies={"vue": "3"}, packageManager="pnpm@9.1.0")}, ["Vue"], "frontend_application", ["pnpm"]),
        ({"requirements.txt": "Django==4.2\n"}, ["Django"], "backend_service", ["pip"]),
        ({"requirements.txt": "flask\n", "Pipfile": ""}, ["Flask"], "backend_service", ["pip", "pipenv"]),
        ({"pyproject.toml": PYPROJECT_POETRY, "poetry.lock": ""}, ["FastAPI"], "backend_service", ["poetry"]),
        ({"pyproject.toml": "[project]\ndependencies = ['fastapi']\n", "uv.lock": ""}, ["FastAPI"], "backend_service", ["pip", "uv"]),
        ({"pom.xml": "<dependency>spring-boot-starter-web</dependency>"}, ["Spring Boot"], "backend_service", ["maven"]),
        ({"build.gradle": "id 'org.springframework.boot' version '3'"}, ["Spring Boot"], "backend_service", ["gradle"]),
        ({"package.json": package_json(workspaces=["packages/*"], dependencies={"react": "19"})}, ["React"], "monorepo", ["npm"]),
        ({"pnpm-workspace.yaml": "packages: []\n", "package.json": package_json(name="root")}, [], "monorepo", ["pnpm"]),
        ({"package.json": package_json(name="tool", bin={"tool": "cli.js"})}, [], "cli", ["npm"]),
        ({"pyproject.toml": "[project]\nname='t'\n[project.scripts]\ntool = 't:main'\n"}, [], "cli", ["pip"]),
        ({"package.json": package_json(name="lib", main="dist/index.js")}, [], "library", ["npm"]),
        ({"pyproject.toml": "[build-system]\nrequires=['hatchling']\n[project]\nname='lib'\n"}, [], "library", ["pip"]),
        ({"README.md": "hello"}, [], "unknown", []),
    ],
)
def test_frameworks_package_managers_and_project_type(client, database, monkeypatch, files, frameworks, project_type, managers):
    _, _, body = analyze(client, database, monkeypatch, files)

    assert body["frameworks"] == frameworks
    assert body["project_type"] == project_type
    assert body["package_managers"] == managers
    assert body["status"] == "completed"


def test_pyproject_dependencies_are_extracted_with_dev_flags_and_no_versions(client, database, monkeypatch):
    _, _, body = analyze(client, database, monkeypatch, {"pyproject.toml": PYPROJECT_POETRY})

    assert {(d["name"], d["dev"]) for d in body["dependencies"]} == {
        ("fastapi", False), ("uvicorn", False), ("python-dotenv", False), ("requests", False),
        ("pytest", True), ("black", True),
    }
    assert all(set(d) == {"name", "ecosystem", "dev"} for d in body["dependencies"])


def test_entry_points_from_package_scripts_bin_main_pyproject_and_python_files(client, database, monkeypatch):
    files = {
        "package.json": package_json(name="tool", main="lib/index.js", bin={"tool": "cli.js"},
                                     scripts={"start": "node .", "test": "jest", "serve": "x"}),
        "pyproject.toml": "[project]\nname='t'\n[project.scripts]\nrun-t = 't:main'\n",
        "manage.py": "import django\n",
        "src/app.py": "app = 1\n",
        "a/b/c/d/deep.py": "x = 1\n",
        "a/b/c/d/main.py": "x = 1\n",
    }
    _, _, body = analyze(client, database, monkeypatch, files)

    assert {(e["kind"], e["name"], e["path"]) for e in body["entry_points"]} == {
        ("npm_script", "start", "package.json"), ("npm_script", "serve", "package.json"),
        ("npm_bin", "tool", "package.json"), ("npm_main", "lib/index.js", "package.json"),
        ("python_script", "run-t", "pyproject.toml"),
        ("python_file", "manage.py", "manage.py"), ("python_file", "app.py", "src/app.py"),
    }


def test_important_files_are_detected_by_name_and_pattern(client, database, monkeypatch):
    files = {name: "x\n" for name in (
        "README.md", "readme.rst", "package.json", "requirements.txt", "requirements-dev.txt", "pyproject.toml",
        "Dockerfile", "Dockerfile.prod", "docker-compose.yaml", "tsconfig.json", "tsconfig.build.json",
        "next.config.mjs", "vite.config.ts", "eslint.config.js", ".eslintrc.json", "Makefile",
        "src/notes.txt", "src/index.ts", "requirements.md",
    )}
    files["package.json"] = "{}"
    files["pyproject.toml"] = "[project]\nname='x'\n"
    _, _, body = analyze(client, database, monkeypatch, files)

    assert set(body["important_files"]) == {
        "README.md", "readme.rst", "package.json", "requirements.txt", "requirements-dev.txt", "pyproject.toml",
        "Dockerfile", "Dockerfile.prod", "docker-compose.yaml", "tsconfig.json", "tsconfig.build.json",
        "next.config.mjs", "vite.config.ts", "eslint.config.js", ".eslintrc.json", "Makefile",
    }


# ------------------------------------------------------------ missing / malformed manifests


def test_repository_without_manifests_is_analyzed_safely(client, database, monkeypatch):
    _, _, body = analyze(client, database, monkeypatch, {"README.md": "hi", "main.py": "print(1)\n"})

    assert body["status"] == "completed"
    assert (body["project_type"], body["frameworks"], body["dependencies"]) == ("unknown", [], [])
    assert body["languages"] == ["python"]
    assert body["entry_points"] == [{"kind": "python_file", "name": "main.py", "path": "main.py"}]


def test_empty_repository_is_analyzed_safely(client, database, monkeypatch):
    _, _, body = analyze(client, database, monkeypatch, {})

    assert body["status"] == "completed"
    assert body["project_type"] == "unknown" and body["languages"] == [] and body["important_files"] == []


def test_malformed_unreadable_and_oversized_manifests_are_skipped_without_failing_the_scan(client, database, monkeypatch):
    files = {
        "package.json": "{not valid json",
        "pyproject.toml": "[[[ definitely not toml",
        "requirements.txt": "fastapi\n",
        "services/requirements-extra.txt": "caf\xe9==1\n".encode("latin-1"),
        "big/requirements.txt": "a==1\n" * 120_000,
        "main.py": "print(1)\n",
    }
    repository_id, headers = create_repository(database)
    use_github(monkeypatch, FakeGitHubClient(files))

    scan_response = scan(client, repository_id, headers)
    body = get_analysis(client, repository_id, headers).json()

    assert scan_response.status_code == 200 and scan_response.json()["files_indexed"] >= 1
    assert body["status"] == "partial"
    assert sorted((m["path"], m["reason"]) for m in body["skipped_manifests"]) == [
        ("big/requirements.txt", "too_large"),
        ("package.json", "malformed"),
        ("pyproject.toml", "malformed"),
        ("services/requirements-extra.txt", "unreadable"),
    ]
    assert body["frameworks"] == ["FastAPI"] and body["package_managers"] == ["pip"]
    assert "definitely not toml" not in json.dumps(body)


def test_unavailable_manifest_blob_is_skipped(client, database, monkeypatch):
    repository_id, headers = create_repository(database)
    use_github(monkeypatch, FakeGitHubClient({"requirements.txt": "flask\n", "main.py": "x\n"}, unavailable={"requirements.txt"}))

    assert scan(client, repository_id, headers).status_code == 200
    body = get_analysis(client, repository_id, headers).json()

    assert body["status"] == "partial"
    assert body["skipped_manifests"] == [{"path": "requirements.txt", "reason": "unavailable"}]


def test_recovery_after_fixing_a_malformed_manifest(client, database, monkeypatch):
    repository_id, headers, broken = analyze(client, database, monkeypatch, {"package.json": "{oops"})
    assert broken["status"] == "partial"

    use_github(monkeypatch, FakeGitHubClient({"package.json": package_json(dependencies={"express": "4"})}))
    scan(client, repository_id, headers)
    fixed = get_analysis(client, repository_id, headers).json()

    assert fixed["status"] == "completed" and fixed["skipped_manifests"] == []
    assert fixed["frameworks"] == ["Express"]


# ------------------------------------------------------------ scan integration and failure isolation


def test_analysis_failure_never_fails_the_scan_and_is_stored_safely(client, database, monkeypatch, caplog):
    repository_id, headers = create_repository(database)
    use_github(monkeypatch, FakeGitHubClient(FULLSTACK))
    monkeypatch.setattr(analyzer_service, "analyze_project",
                        lambda *args: (_ for _ in ()).throw(RuntimeError("sk-leaky-manifest-content")))
    caplog.set_level(logging.DEBUG)

    response = scan(client, repository_id, headers)

    assert response.status_code == 200
    assert response.json()["status"] == "completed" and response.json()["chunks_created"] > 0
    body = get_analysis(client, repository_id, headers).json()
    assert body["status"] == "failed"
    assert (body["project_type"], body["languages"], body["dependencies"]) == ("unknown", [], [])
    assert "sk-leaky-manifest-content" not in caplog.text + json.dumps(body)
    with database() as session:
        assert session.scalar(select(func.count()).select_from(RepositoryChunk)) > 0
        row = session.scalars(select(RepositoryAnalysis)).one()
        assert row.analysis_metadata["error"] == analyzer_service.FAILURE_MESSAGE


def test_failed_reanalysis_keeps_previous_data_marked_failed_and_recovers(client, database, monkeypatch):
    repository_id, headers, good = analyze(client, database, monkeypatch, FULLSTACK)

    class TreeFailsOnSecondCall(FakeGitHubClient):
        calls = 0

        def get_tree(self, owner, name, ref):
            self.calls += 1
            if self.calls >= 2:
                raise GitHubContentError("boom raw github body")
            return super().get_tree(owner, name, ref)

    use_github(monkeypatch, TreeFailsOnSecondCall(FULLSTACK))
    response = scan(client, repository_id, headers)
    failed = get_analysis(client, repository_id, headers)

    assert response.status_code == 200
    assert failed.json()["status"] == "failed"
    assert failed.json()["project_type"] == good["project_type"]
    assert "boom" not in failed.text
    assert len(analysis_rows(database)) == 1

    use_github(monkeypatch, FakeGitHubClient(FULLSTACK))
    scan(client, repository_id, headers)
    assert get_analysis(client, repository_id, headers).json()["status"] == "completed"


def test_scan_failure_does_not_create_an_analysis(client, database, monkeypatch):
    repository_id, headers = create_repository(database)
    use_github(monkeypatch, FakeGitHubClient(tree_error=GitHubContentError("boom")))

    assert scan(client, repository_id, headers).status_code == 502
    assert analysis_rows(database) == []
    assert get_analysis(client, repository_id, headers).status_code == 409


def test_scan_still_reports_embeddings_when_the_embedding_feature_is_present(client, database, monkeypatch):
    """Runs only where the embedding integration exists: analysis must not disturb it."""

    pytest.importorskip("app.embeddings.indexing")

    class ConstantProvider:
        model = "constant"

        def embed(self, texts):
            return [[0.5] * 1536 for _ in texts]

    monkeypatch.setattr(repository_routes, "get_embedding_provider", lambda: ConstantProvider())
    repository_id, headers = create_repository(database)
    use_github(monkeypatch, FakeGitHubClient(FULLSTACK))

    response = scan(client, repository_id, headers)

    assert response.status_code == 200
    assert response.json()["embeddings"]["status"] == "completed"
    assert response.json()["embeddings"]["chunks_embedded"] == response.json()["chunks_created"]
    assert get_analysis(client, repository_id, headers).json()["status"] == "completed"


# ------------------------------------------------------------ access control


def test_analysis_requires_authentication(client, database, monkeypatch):
    repository_id, _, _ = analyze(client, database, monkeypatch, FULLSTACK)

    assert get_analysis(client, repository_id, {}).status_code == 401
    assert get_analysis(client, repository_id, {"Authorization": "Bearer nope"}).status_code == 401


def test_unknown_repository_and_other_users_repository_return_not_found(client, database, monkeypatch):
    other_id, _, _ = analyze(client, database, monkeypatch, FULLSTACK, email="other@example.com")
    _, my_headers = create_repository(database, email="me@example.com")

    for repository_id in (other_id, "00000000-0000-4000-8000-000000000000"):
        response = get_analysis(client, repository_id, my_headers)
        assert response.status_code == 404
        assert "web_application" not in response.text and "FastAPI" not in response.text


def test_invalid_repository_id_is_rejected(client, database):
    _, headers = create_repository(database)

    assert client.get("/api/v1/repositories/not-a-uuid/analysis", headers=headers).status_code == 422


def test_unscanned_repository_returns_a_clear_conflict(client, database):
    repository_id, headers = create_repository(database)

    response = get_analysis(client, repository_id, headers)

    assert response.status_code == 409
    assert "scan" in response.json()["error"]["message"].lower()


def test_scanning_another_users_repository_never_creates_analysis(client, database, monkeypatch):
    other_id, _ = create_repository(database, email="other@example.com")
    _, my_headers = create_repository(database, email="me@example.com")
    use_github(monkeypatch, FakeGitHubClient(FULLSTACK))

    assert scan(client, other_id, my_headers).status_code == 404
    assert analysis_rows(database) == []


# ------------------------------------------------------------ manifest parsers


def test_requirements_parser_handles_common_line_forms():
    info = parse_manifest("requirements.txt", (
        "# comment\n\nFastAPI[all]>=0.1 ; python_version>'3'\nPyJWT==2.9 # inline\nargon2_cffi\n"
        "-r base.txt\n--index-url https://x.test\n-e .\n./local\ngit+https://x.test/r.git#egg=r\nhttps://x.test/a.whl\n"
    ))

    assert [d.name for d in info.dependencies] == ["fastapi", "pyjwt", "argon2-cffi"]
    assert info.package_managers == {"pip"}


@pytest.mark.parametrize(
    ("path", "text"),
    [("package.json", "{bad"), ("package.json", "[]"), ("package.json", "42"), ("pyproject.toml", "= nope ="), ("unknown.lock", "{}")],
)
def test_parsers_raise_a_safe_error_for_invalid_input(path, text):
    with pytest.raises(ManifestParseError) as error:
        parse_manifest(path, text)

    assert text not in str(error.value) or text == "42"


def test_package_json_with_unexpected_field_types_is_tolerated():
    info = parse_manifest("package.json", json.dumps({
        "dependencies": ["not", "a", "dict"], "devDependencies": {"ok": "1", "1": 2},
        "scripts": "nope", "bin": 5, "main": 7, "packageManager": "unknown@1",
    }))

    assert [d.name for d in info.dependencies] == ["ok", "1"]
    assert info.entry_points == [] and info.package_managers == set()
