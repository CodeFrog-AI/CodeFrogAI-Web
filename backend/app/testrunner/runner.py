"""A controlled test runner: the only way CodeFrog runs a repository's tests.

This is validation, not a shell. The server decides the command from a small allowlist of
fixed templates chosen by looking at the project (pytest for Python, `npm test` for Node);
nothing a model or a client says can add an argument, an option, or a program. The only
variable part is a list of test file paths, which must be existing, safe, repository-relative
files that look like tests. Commands run without a shell, in the workspace, with a timeout,
a sanitized environment (no API keys, tokens, or database URLs), and bounded, redacted
output.

Limitation: the tests are the project's own code (plus the AI's edits) run on the server's
interpreter, without network or filesystem isolation. Treat it like any CI runner that
executes the code it is asked to test.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.context.redaction import is_sensitive_path, redact_secrets
from app.workspace.workspace import WorkspaceError, validate_path

logger = logging.getLogger(__name__)

TEST_TIMEOUT_SECONDS = 120
MAX_OUTPUT_CHARS = 4_000
MAX_CAPTURED_BYTES = 200_000
MAX_TEST_PATHS = 10
SCAN_DEPTH = 3

_PASSTHROUGH_ENVIRONMENT = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "TMPDIR", "PATHEXT", "COMSPEC")
_PYTHON_MARKERS = ("pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini", "conftest.py")
_PYTHON_TEST_FILE = re.compile(r"(^|/)(test_[^/]+|[^/]+_test)\.py$")
_SAFE_PATH = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]*$")
_COUNT = re.compile(r"(\d+) (passed|failed|error|errors)")

Status = Literal["passed", "failed", "not_run"]


@dataclass(frozen=True)
class TestResult:
    __test__ = False  # not a pytest test class

    status: Status
    command: str | None
    passed: int
    failed: int
    errors: int
    duration_ms: int
    timed_out: bool
    output: str
    reason: str | None = None


def is_test_file(path: str) -> bool:
    return bool(_PYTHON_TEST_FILE.search(path))


def detect_project(root: Path) -> Literal["python", "node"] | None:
    """Which allowlisted strategy fits this checkout, judged by files only."""

    if any((root / marker).is_file() for marker in _PYTHON_MARKERS) or _has_python_tests(root):
        return "python"
    package = root / "package.json"
    if package.is_file():
        try:
            scripts = json.loads(package.read_text(encoding="utf-8")).get("scripts", {})
        except (OSError, ValueError, AttributeError):
            return None
        if isinstance(scripts, dict) and isinstance(scripts.get("test"), str):
            return "node"
    return None


def _has_python_tests(root: Path) -> bool:
    for current, directories, files in os.walk(root):
        depth = len(Path(current).relative_to(root).parts)
        directories[:] = [d for d in directories if d not in (".git", "node_modules", "__pycache__", ".venv", "venv") and depth < SCAN_DEPTH]
        if any(_PYTHON_TEST_FILE.search(name) for name in files):
            return True
    return False


def select_test_paths(root: Path, candidates: list[str]) -> list[str]:
    """The candidates that are safe, existing test files (at most MAX_TEST_PATHS); everything else is dropped."""

    selected: list[str] = []
    for candidate in candidates:
        if len(selected) >= MAX_TEST_PATHS:
            break
        try:
            relative = validate_path(candidate)
        except WorkspaceError:
            continue
        if not _SAFE_PATH.match(relative) or is_sensitive_path(relative) or not is_test_file(relative):
            continue
        target = root / relative
        if target.is_symlink() or not target.is_file() or relative in selected:
            continue
        selected.append(relative)
    return selected


def run_tests(root: Path, paths: list[str] | None = None) -> TestResult:
    """Run the project's tests from the allowlist. Never raises for a failing or missing test setup."""

    project = detect_project(root)
    if project is None:
        return TestResult("not_run", None, 0, 0, 0, 0, False, "", "No supported test setup was found (pytest or npm test).")
    if project == "python":
        chosen = select_test_paths(root, paths or [])
        argv = [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", "--maxfail=25", *chosen]
        display = " ".join(["pytest", "-q", *chosen])
    else:
        npm = shutil.which("npm")
        if npm is None:
            return TestResult("not_run", "npm test", 0, 0, 0, 0, False, "", "npm is not available on the server.")
        argv = [npm, "test", "--silent"]
        display = "npm test"
    return _execute(root, argv, display, project)


def _environment() -> dict[str, str]:
    environment = {name: os.environ[name] for name in _PASSTHROUGH_ENVIRONMENT if name in os.environ}
    environment.update({"CI": "1", "PYTHONDONTWRITEBYTECODE": "1", "NODE_ENV": "test", "PYTHONUTF8": "1", "NO_COLOR": "1"})
    return environment


def _execute(root: Path, argv: list[str], display: str, project: str) -> TestResult:
    started = time.perf_counter()
    timed_out = False
    with tempfile.TemporaryFile() as capture:
        try:
            completed = subprocess.run(  # noqa: S603 - fixed allowlisted argv, no shell
                argv, cwd=root, env=_environment(), stdin=subprocess.DEVNULL, stdout=capture, stderr=subprocess.STDOUT,
                timeout=TEST_TIMEOUT_SECONDS, shell=False, check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out, returncode = True, -1
        except OSError as error:
            logger.warning("Test runner could not start (exception type=%s)", type(error).__name__)
            return TestResult("not_run", display, 0, 0, 0, 0, False, "", "The test command could not be started.")
        size = capture.seek(0, os.SEEK_END)
        capture.seek(max(size - MAX_CAPTURED_BYTES, 0))
        raw = capture.read().decode("utf-8", errors="replace")
    duration_ms = round((time.perf_counter() - started) * 1000)
    output = _tail(redact_secrets(raw)[0])
    counts = {"passed": 0, "failed": 0, "error": 0}
    for number, word in _COUNT.findall(raw[-2_000:]):
        counts["error" if word.startswith("error") else word] = int(number)
    logger.info("Tests finished project=%s returncode=%d timed_out=%s", project, returncode, timed_out)
    if timed_out:
        return TestResult("failed", display, counts["passed"], counts["failed"], counts["error"], duration_ms, True, output, f"The tests did not finish within {TEST_TIMEOUT_SECONDS} seconds.")
    if project == "python" and returncode == 5:
        return TestResult("not_run", display, 0, 0, 0, duration_ms, False, output, "No tests were collected.")
    status: Status = "passed" if returncode == 0 else "failed"
    return TestResult(status, display, counts["passed"], counts["failed"], counts["error"], duration_ms, False, output)


def _tail(text: str) -> str:
    text = text.strip()
    return text if len(text) <= MAX_OUTPUT_CHARS else "..." + text[-MAX_OUTPUT_CHARS:]
