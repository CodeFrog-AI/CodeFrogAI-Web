"""The only place CodeFrog starts a Git process.

Every call is an argument list (never a shell string), runs with the repository as its
working directory, has a timeout, and gets a minimal, sanitized environment: no user or
system Git configuration, no credential helpers, no prompts, no hooks. Credentials are
passed through the environment, never on the command line or in a remote URL, and process
output is never returned to callers, so tokens and filesystem paths cannot leak from here.
"""

import base64
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import ApplicationError

logger = logging.getLogger(__name__)

LOCAL_TIMEOUT_SECONDS = 30
NETWORK_TIMEOUT_SECONDS = 180
MAX_OUTPUT_BYTES = 8_000_000

_PASSTHROUGH_ENVIRONMENT = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "TMPDIR", "PATHEXT")

# Applied to every command, before the subcommand: neutralize repository-level surprises.
_SAFE_CONFIG = (
    "core.hooksPath=" + os.devnull,
    "core.fsmonitor=false",
    "core.symlinks=false",
    "core.autocrlf=false",
    "core.quotePath=false",
    "credential.helper=",
    "protocol.ext.allow=never",
    "commit.gpgsign=false",
    "gc.auto=0",
)


class GitError(ApplicationError):
    """A failed or refused Git operation. Messages are client-safe: no paths, output, or URLs."""

    status_code = 500
    code = "GIT_ERROR"
    message = "The Git operation failed."

    def __init__(self, code: str, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True)
class GitResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _environment(token: str | None) -> dict[str, str]:
    environment = {name: os.environ[name] for name in _PASSTHROUGH_ENVIRONMENT if name in os.environ}
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_ASKPASS": "",
            "LC_ALL": "C",
        }
    )
    if token:
        credentials = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        environment.update(
            {
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "http.https://github.com/.extraHeader",
                "GIT_CONFIG_VALUE_0": f"Authorization: Basic {credentials}",
            }
        )
    return environment


def run_git(
    directory: Path,
    arguments: list[str],
    *,
    token: str | None = None,
    network: bool = False,
    config: tuple[str, ...] = (),
) -> GitResult:
    """Run `git <arguments>` inside `directory` and return its result (never raises on a non-zero exit)."""

    executable = shutil.which("git")
    if executable is None:
        raise GitError("GIT_UNAVAILABLE", "Git is not available on the server.", 503)
    command = [executable, *(item for setting in (*_SAFE_CONFIG, *config) for item in ("-c", setting)), *arguments]
    try:
        completed = subprocess.run(  # noqa: S603 - argument list, no shell, fixed executable
            command,
            cwd=directory,
            env=_environment(token),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=NETWORK_TIMEOUT_SECONDS if network else LOCAL_TIMEOUT_SECONDS,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Git timed out subcommand=%s", arguments[0])
        raise GitError("GIT_TIMEOUT", "The Git operation timed out.", 504) from None
    except OSError as error:
        logger.warning("Git could not start (exception type=%s)", type(error).__name__)
        raise GitError("GIT_UNAVAILABLE", "Git is not available on the server.", 503) from None
    logger.info("Git finished subcommand=%s returncode=%d", arguments[0], completed.returncode)
    return GitResult(
        completed.returncode,
        completed.stdout[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace"),
        completed.stderr[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace"),
    )
