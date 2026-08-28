"""Core persistent entities and their relationships.

These models intentionally describe storage only. Integrations, agent execution,
and vector embeddings are introduced by later roadmap features.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class TimestampMixin:
    """Provide consistent, timezone-aware audit timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), default="active", nullable=False)

    github_accounts: Mapped[list["GitHubAccount"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    agent_tasks: Mapped[list["AgentTask"]] = relationship(back_populates="user")


class GitHubAccount(TimestampMixin, Base):
    __tablename__ = "github_accounts"
    __table_args__ = (UniqueConstraint("github_user_id", name="uq_github_accounts_github_user_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    github_user_id: Mapped[int] = mapped_column(nullable=False)
    login: Mapped[str] = mapped_column(String(255), nullable=False)
    access_token_encrypted: Mapped[str | None] = mapped_column(Text)

    user: Mapped[User] = relationship(back_populates="github_accounts")
    repositories: Mapped[list["Repository"]] = relationship(back_populates="github_account")


class Repository(TimestampMixin, Base):
    __tablename__ = "repositories"
    __table_args__ = (
        UniqueConstraint("github_repository_id", name="uq_repositories_github_repository_id"),
        UniqueConstraint("owner", "name", name="uq_repositories_owner_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    github_account_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("github_accounts.id", ondelete="CASCADE"), index=True
    )
    github_repository_id: Mapped[int] = mapped_column(nullable=False)
    owner: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    default_branch: Mapped[str] = mapped_column(String(255), default="main", nullable=False)
    connection_status: Mapped[str] = mapped_column(String(32), default="connected", nullable=False)
    connection_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    github_account: Mapped[GitHubAccount] = relationship(back_populates="repositories")
    files: Mapped[list["RepositoryFile"]] = relationship(
        back_populates="repository", cascade="all, delete-orphan"
    )
    agent_tasks: Mapped[list["AgentTask"]] = relationship(back_populates="repository")
    pull_requests: Mapped[list["PullRequest"]] = relationship(back_populates="repository")


class RepositoryFile(TimestampMixin, Base):
    __tablename__ = "repository_files"
    __table_args__ = (UniqueConstraint("repository_id", "path", name="uq_repository_files_path"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repository_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"), index=True
    )
    path: Mapped[str] = mapped_column(String(2048), nullable=False)
    language: Mapped[str | None] = mapped_column(String(64))
    sha: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    size_bytes: Mapped[int | None] = mapped_column(Integer)

    repository: Mapped[Repository] = relationship(back_populates="files")
    chunks: Mapped[list["RepositoryChunk"]] = relationship(
        back_populates="repository_file", cascade="all, delete-orphan"
    )


class RepositoryChunk(TimestampMixin, Base):
    __tablename__ = "repository_chunks"
    __table_args__ = (
        UniqueConstraint("repository_file_id", "chunk_index", name="uq_repository_chunks_position"),
        CheckConstraint("end_line >= start_line", name="ck_repository_chunks_line_range"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repository_file_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("repository_files.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    context_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    repository_file: Mapped[RepositoryFile] = relationship(back_populates="chunks")


class AgentTask(TimestampMixin, Base):
    __tablename__ = "agent_tasks"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'planning', 'running', 'waiting_for_approval', 'completed', 'failed', 'cancelled')",
            name="ck_agent_tasks_status",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"), index=True)
    repository_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id", ondelete="RESTRICT"), index=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True, nullable=False)

    user: Mapped[User] = relationship(back_populates="agent_tasks")
    repository: Mapped[Repository] = relationship(back_populates="agent_tasks")
    runs: Mapped[list["AgentRun"]] = relationship(back_populates="task", cascade="all, delete-orphan")
    code_changes: Mapped[list["CodeChange"]] = relationship(back_populates="task")
    pull_requests: Mapped[list["PullRequest"]] = relationship(back_populates="task")


class AgentRun(TimestampMixin, Base):
    __tablename__ = "agent_runs"
    __table_args__ = (UniqueConstraint("task_id", "run_number", name="uq_agent_runs_task_number"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agent_tasks.id", ondelete="CASCADE"), index=True)
    run_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)

    task: Mapped[AgentTask] = relationship(back_populates="runs")
    messages: Mapped[list["AgentMessage"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    tool_calls: Mapped[list["ToolCall"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    code_changes: Mapped[list["CodeChange"]] = relationship(back_populates="run")
    test_runs: Mapped[list["TestRun"]] = relationship(back_populates="run")
    security_scans: Mapped[list["SecurityScan"]] = relationship(back_populates="run")


class AgentMessage(TimestampMixin, Base):
    __tablename__ = "agent_messages"
    __table_args__ = (UniqueConstraint("run_id", "sequence", name="uq_agent_messages_sequence"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    message_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    run: Mapped[AgentRun] = relationship(back_populates="messages")


class ToolCall(TimestampMixin, Base):
    __tablename__ = "tool_calls"
    __table_args__ = (UniqueConstraint("run_id", "sequence", name="uq_tool_calls_sequence"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    arguments: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)

    run: Mapped[AgentRun] = relationship(back_populates="tool_calls")


class CodeChange(TimestampMixin, Base):
    __tablename__ = "code_changes"
    __table_args__ = (UniqueConstraint("agent_run_id", "file_path", name="uq_code_changes_run_file"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agent_tasks.id", ondelete="RESTRICT"), index=True)
    agent_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agent_runs.id", ondelete="RESTRICT"), index=True)
    file_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    patch: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="generated", nullable=False)

    task: Mapped[AgentTask] = relationship(back_populates="code_changes")
    run: Mapped[AgentRun] = relationship(back_populates="code_changes")


class TestRun(TimestampMixin, Base):
    __tablename__ = "test_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agent_runs.id", ondelete="RESTRICT"), index=True)
    command: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True, nullable=False)
    exit_code: Mapped[int | None] = mapped_column(Integer)
    output: Mapped[str | None] = mapped_column(Text)

    run: Mapped[AgentRun] = relationship(back_populates="test_runs")


class SecurityScan(TimestampMixin, Base):
    __tablename__ = "security_scans"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agent_runs.id", ondelete="RESTRICT"), index=True)
    scanner: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True, nullable=False)
    findings: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    summary: Mapped[str | None] = mapped_column(Text)

    run: Mapped[AgentRun] = relationship(back_populates="security_scans")


class PullRequest(TimestampMixin, Base):
    __tablename__ = "pull_requests"
    __table_args__ = (UniqueConstraint("repository_id", "github_pull_request_number", name="uq_pull_requests_number"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repository_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id", ondelete="RESTRICT"), index=True)
    task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agent_tasks.id", ondelete="RESTRICT"), index=True)
    github_pull_request_number: Mapped[int] = mapped_column(nullable=False)
    branch_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="open", index=True, nullable=False)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)

    repository: Mapped[Repository] = relationship(back_populates="pull_requests")
    task: Mapped[AgentTask] = relationship(back_populates="pull_requests")


Index("ix_repository_files_repository_sha", RepositoryFile.repository_id, RepositoryFile.sha)
