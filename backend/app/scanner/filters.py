"""Decide which repository files are worth indexing, and their language."""

from pathlib import PurePosixPath

MAX_FILE_BYTES = 512 * 1024

IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".next",
        "dist",
        "build",
        "coverage",
        ".venv",
        "venv",
        ".pytest_cache",
        ".mypy_cache",
    }
)

# Machine-generated files that share an otherwise supported extension.
IGNORED_FILENAMES = frozenset({"package-lock.json", "pnpm-lock.yaml"})
GENERATED_SUFFIXES = (".min.js", ".min.css")

LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".sql": "sql",
    ".sh": "shell",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
}


def detect_language(path: str) -> str | None:
    """Return the indexable language for a repository path, or None to skip it."""

    posix_path = PurePosixPath(path)
    if any(part in IGNORED_DIRECTORIES for part in posix_path.parts[:-1]):
        return None
    name = posix_path.name.lower()
    if name in IGNORED_FILENAMES or name.endswith(GENERATED_SUFFIXES):
        return None
    return LANGUAGE_BY_EXTENSION.get(posix_path.suffix.lower())
