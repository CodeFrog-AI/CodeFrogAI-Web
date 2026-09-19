"""Deterministic project-structure rules applied to a file list and parsed manifests."""

from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

from app.analyzer.manifests import Dependency, EntryPoint, ManifestInfo
from app.scanner.filters import IGNORED_DIRECTORIES

MAX_DEPENDENCIES = 300
MAX_IMPORTANT_FILES = 60
MAX_ENTRY_POINTS = 40

# json/yaml are configuration formats, not the project's programming languages.
NON_PROGRAMMING_LANGUAGES = frozenset({"json", "yaml"})

FRAMEWORKS_BY_DEPENDENCY = {
    ("npm", "next"): "Next.js",
    ("npm", "react"): "React",
    ("npm", "vue"): "Vue",
    ("npm", "nuxt"): "Nuxt",
    ("npm", "@angular/core"): "Angular",
    ("npm", "svelte"): "Svelte",
    ("npm", "express"): "Express",
    ("npm", "@nestjs/core"): "NestJS",
    ("npm", "fastify"): "Fastify",
    ("pypi", "fastapi"): "FastAPI",
    ("pypi", "django"): "Django",
    ("pypi", "flask"): "Flask",
}
FRONTEND_FRAMEWORKS = frozenset({"React", "Vue", "Angular", "Svelte"})
BACKEND_FRAMEWORKS = frozenset({"FastAPI", "Django", "Flask", "Express", "NestJS", "Fastify", "Spring Boot"})
FULLSTACK_FRAMEWORKS = frozenset({"Next.js", "Nuxt"})

PACKAGE_MANAGER_FILES = {
    "package-lock.json": "npm",
    "yarn.lock": "yarn",
    "pnpm-workspace.yaml": "pnpm",
    "pnpm-lock.yaml": "pnpm",
    "bun.lockb": "bun",
    "bun.lock": "bun",
    "poetry.lock": "poetry",
    "uv.lock": "uv",
    "pipfile": "pipenv",
    "pipfile.lock": "pipenv",
    "setup.py": "pip",
    "cargo.toml": "cargo",
    "go.mod": "go",
}
JS_PACKAGE_MANAGERS = frozenset({"npm", "yarn", "pnpm", "bun"})
WORKSPACE_FILES = frozenset({"pnpm-workspace.yaml", "lerna.json", "nx.json", "turbo.json"})

IMPORTANT_FILENAMES = frozenset(
    {
        "package.json", "pyproject.toml", "setup.py", "setup.cfg", "pipfile", "dockerfile",
        "docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml", "tsconfig.json",
        "makefile", "alembic.ini", "pytest.ini", "tox.ini", "go.mod", "cargo.toml", "pom.xml",
        "build.gradle", "build.gradle.kts", ".env.example", *WORKSPACE_FILES,
    }
)
IMPORTANT_PREFIXES = (
    "readme", "next.config.", "vite.config.", "eslint.config.", ".eslintrc", "tailwind.config.",
    "webpack.config.", "dockerfile.", "tsconfig.",
)
PYTHON_ENTRY_FILENAMES = frozenset(
    {"main.py", "app.py", "manage.py", "wsgi.py", "asgi.py", "run.py", "server.py", "__main__.py"}
)
MAX_PYTHON_ENTRY_DEPTH = 3


@dataclass(frozen=True)
class ProjectAnalysis:
    project_type: str
    languages: list[str]
    frameworks: list[str]
    package_managers: list[str]
    dependencies: list[dict]
    important_files: list[str]
    entry_points: list[dict]
    dependencies_truncated: bool


def _depth_then_path(path: str) -> tuple[int, str]:
    return (path.count("/"), path)


def is_ignored_path(path: str) -> bool:
    return any(part in IGNORED_DIRECTORIES for part in PurePosixPath(path).parts[:-1])


def _is_important(path: str) -> bool:
    name = PurePosixPath(path).name.lower()
    if name in IMPORTANT_FILENAMES or name.startswith(IMPORTANT_PREFIXES):
        return True
    if name.startswith("requirements") and name.endswith(".txt"):
        return True
    return path.startswith(".github/workflows/") and name.endswith((".yml", ".yaml"))


def _merge_dependencies(manifests: list[ManifestInfo]) -> list[Dependency]:
    """One entry per package; it counts as dev only if every manifest lists it as dev."""

    merged: dict[tuple[str, str], bool] = {}
    for manifest in manifests:
        for dependency in manifest.dependencies:
            key = (dependency.ecosystem, dependency.name)
            merged[key] = merged.get(key, True) and dependency.dev
    return [Dependency(name, ecosystem, dev) for (ecosystem, name), dev in sorted(merged.items())]


def _project_type(frameworks: set[str], manifests: list[ManifestInfo], paths: list[str]) -> str:
    flags = set().union(*(manifest.flags for manifest in manifests)) if manifests else set()
    names = {PurePosixPath(path).name.lower() for path in paths}
    if "workspaces" in flags or names & WORKSPACE_FILES:
        return "monorepo"
    frontend = frameworks & FRONTEND_FRAMEWORKS
    backend = frameworks & BACKEND_FRAMEWORKS
    if frameworks & FULLSTACK_FRAMEWORKS or (frontend and backend):
        return "web_application"
    if frontend:
        return "frontend_application"
    if backend:
        return "backend_service"
    if "bin" in flags or "scripts" in flags:
        return "cli"
    if "library_fields" in flags or "build_system" in flags:
        return "library"
    return "unknown"


def analyze_project(
    paths: list[str], manifests: list[ManifestInfo], language_counts: dict[str, int]
) -> ProjectAnalysis:
    """Combine the file list, parsed manifests, and indexed language counts into one result."""

    paths = sorted(path for path in paths if not is_ignored_path(path))
    manifests = sorted(manifests, key=lambda manifest: manifest.path)

    languages = [
        language
        for language, _ in sorted(language_counts.items(), key=lambda item: (-item[1], item[0]))
        if language not in NON_PROGRAMMING_LANGUAGES
    ]

    dependencies = _merge_dependencies(manifests)
    frameworks = {
        FRAMEWORKS_BY_DEPENDENCY[(dependency.ecosystem, dependency.name)]
        for dependency in dependencies
        if (dependency.ecosystem, dependency.name) in FRAMEWORKS_BY_DEPENDENCY
    }
    frameworks.update(*(manifest.frameworks for manifest in manifests))

    package_managers = set().union(*(manifest.package_managers for manifest in manifests)) if manifests else set()
    package_managers.update(
        PACKAGE_MANAGER_FILES[name]
        for name in {PurePosixPath(path).name.lower() for path in paths}
        if name in PACKAGE_MANAGER_FILES
    )
    has_package_json = any(PurePosixPath(m.path).name.lower() == "package.json" for m in manifests)
    if has_package_json and not package_managers & JS_PACKAGE_MANAGERS:
        package_managers.add("npm")

    python_entries = [
        EntryPoint("python_file", PurePosixPath(path).name, path)
        for path in paths
        if PurePosixPath(path).name in PYTHON_ENTRY_FILENAMES
        and path.count("/") <= MAX_PYTHON_ENTRY_DEPTH
    ]
    entry_points = sorted(
        {entry for manifest in manifests for entry in manifest.entry_points} | set(python_entries),
        key=lambda entry: (entry.path, entry.kind, entry.name),
    )

    return ProjectAnalysis(
        project_type=_project_type(frameworks, manifests, paths),
        languages=languages,
        frameworks=sorted(frameworks),
        package_managers=sorted(package_managers),
        dependencies=[asdict(dependency) for dependency in dependencies[:MAX_DEPENDENCIES]],
        important_files=sorted(
            (path for path in paths if _is_important(path)), key=_depth_then_path
        )[:MAX_IMPORTANT_FILES],
        entry_points=[asdict(entry) for entry in entry_points[:MAX_ENTRY_POINTS]],
        dependencies_truncated=len(dependencies) > MAX_DEPENDENCIES,
    )
