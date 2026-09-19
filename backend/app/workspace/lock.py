"""An inter-process lock per repository workspace.

An advisory OS file lock (not a threading lock), so two server processes or threads cannot
initialize, edit, branch, commit, or push in the same checkout at once. It is non-blocking:
a busy workspace is reported as a conflict rather than queued. The lock file lives outside
the checkout so it never shows up in Git status. The OS releases it if the process dies.
"""

import logging
import os
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.core.exceptions import ConflictError

logger = logging.getLogger(__name__)

BUSY_MESSAGE = "Another operation is already running for this repository."

if sys.platform == "win32":
    import msvcrt

    def _try_lock(descriptor: int) -> bool:
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock(descriptor: int) -> None:
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(descriptor: int) -> bool:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _unlock(descriptor: int) -> None:
        fcntl.flock(descriptor, fcntl.LOCK_UN)


@contextmanager
def exclusive_workspace(base_directory: Path, repository_id: uuid.UUID) -> Iterator[None]:
    """Hold the repository's workspace lock, or raise ConflictError immediately if it is busy."""

    locks = Path(base_directory) / "locks"
    try:
        locks.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(locks / f"{repository_id}.lock", os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as error:
        logger.warning("Workspace lock unavailable (exception type=%s)", type(error).__name__)
        raise ConflictError("The workspace could not be locked.") from None
    try:
        if not _try_lock(descriptor):
            raise ConflictError(BUSY_MESSAGE)
        try:
            yield
        finally:
            try:
                _unlock(descriptor)
            except OSError:
                pass
    finally:
        os.close(descriptor)
