"""Test doubles for GitHub: local bare repositories that stand in for the remote.

Nothing here reaches the network. A "remote" is a bare Git repository in a temporary
directory, served to the code under test through a file:// URL, and the "stored GitHub
token" is a recognizable fake so tests can prove it never leaks.
"""

import os
import subprocess
from pathlib import Path

import pytest
from sqlalchemy.orm import object_session

from app.db.models import RepositoryChunk, RepositoryFile
from app.workspace import service as workspace_service
from app.workspace.service import GitRemote

FAKE_TOKEN = "gho_faketokenthatmustneverleak0123456789"
_ENVIRONMENT = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
_IDENTITY = ["-c", "user.name=Remote Dev", "-c", "user.email=dev@example.com", "-c", "commit.gpgsign=false"]


def git(cwd, *arguments, input_text=None):
    """Run real Git for test setup and for inspecting results independently of the code under test."""

    completed = subprocess.run(
        ["git", *_IDENTITY, *arguments], cwd=cwd, env=_ENVIRONMENT, capture_output=True, text=True, check=True, input=input_text
    )
    return completed.stdout


def make_remote(base: Path, name: str, files: dict[str, str], branch: str = "main", symlinks: dict[str, str] | None = None) -> Path:
    """A bare repository containing one commit with `files` (and symlink entries, if any)."""

    work = base / f"{name}-work"
    work.mkdir(parents=True)
    git(work, "init", "-q", "-b", branch)
    for path, text in files.items():
        target = work / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode())
    git(work, "add", "--all")
    for link, target in (symlinks or {}).items():  # a real symlink entry (mode 120000), without needing symlink privileges
        blob = git(work, "hash-object", "-w", "--stdin", input_text=target).strip()
        git(work, "update-index", "--add", "--cacheinfo", f"120000,{blob},{link}")
    git(work, "commit", "-q", "-m", "initial commit")
    bare = base / f"{name}.git"
    git(base, "clone", "-q", "--bare", str(work), str(bare))
    return bare


def indexed_files(repository) -> dict[str, str]:
    """The repository's indexed content, as the fake GitHub's initial commit."""

    session = object_session(repository)
    files = {}
    for file in session.query(RepositoryFile).filter(RepositoryFile.repository_id == repository.id):
        chunks = (
            session.query(RepositoryChunk.content)
            .filter(RepositoryChunk.repository_file_id == file.id)
            .order_by(RepositoryChunk.chunk_index)
        )
        files[file.path] = "".join(content for (content,) in chunks)
    return files


class FakeGitHub:
    """Serves each repository from a bare repository built from its indexed files (once)."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.remotes: dict = {}

    def remote_path(self, repository) -> Path:
        if repository.id not in self.remotes:
            self.remotes[repository.id] = make_remote(
                self.base, str(repository.id), indexed_files(repository), repository.default_branch
            )
        return self.remotes[repository.id]

    def url(self, repository) -> str:
        return self.remote_path(repository).as_uri()


@pytest.fixture
def fake_github(monkeypatch, tmp_path):
    fake = FakeGitHub(tmp_path / "remotes")
    monkeypatch.setattr(workspace_service, "remote_url", fake.url)
    monkeypatch.setattr(workspace_service, "resolve_remote", lambda repository: GitRemote(fake.url(repository), FAKE_TOKEN))
    return fake


def add_remote_commit(bare: Path, scratch: Path, files: dict[str, str], branch: str = "main") -> None:
    """Someone else pushes a commit to `branch` of the remote."""

    work = scratch / f"push-{len(list(scratch.glob('push-*')))}"
    git(scratch, "clone", "-q", str(bare), str(work))
    git(work, "checkout", "-q", "-B", branch)
    for path, text in files.items():
        (work / path).parent.mkdir(parents=True, exist_ok=True)
        (work / path).write_bytes(text.encode())
    git(work, "add", "--all")
    git(work, "commit", "-q", "-m", "remote change")
    git(work, "push", "-q", "origin", f"HEAD:refs/heads/{branch}", "--force")
