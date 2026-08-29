"""Tests for the SQLAlchemy domain model metadata."""

from sqlalchemy import CheckConstraint, UniqueConstraint
from sqlalchemy.orm import configure_mappers

from app.db.base import Base
from app.db.models import AgentRun, AgentTask, Repository, RepositoryChunk, RepositoryFile, User


def test_all_roadmap_models_are_registered():
    expected_tables = {
        "users", "github_accounts", "repositories", "repository_files",
        "repository_chunks", "agent_tasks", "agent_runs", "agent_messages",
        "tool_calls", "code_changes", "test_runs", "security_scans", "pull_requests",
    }

    assert expected_tables <= set(Base.metadata.tables)


def test_model_relationships_are_configured():
    configure_mappers()

    assert User.__mapper__.relationships["github_accounts"].back_populates == "user"
    assert Repository.__mapper__.relationships["files"].cascade.delete_orphan
    assert AgentTask.__mapper__.relationships["runs"].cascade.delete_orphan
    assert AgentRun.__mapper__.relationships["messages"].back_populates == "run"


def test_key_integrity_constraints_are_declared():
    file_constraints = {constraint.name for constraint in RepositoryFile.__table__.constraints if isinstance(constraint, UniqueConstraint)}
    chunk_checks = {constraint.name for constraint in RepositoryChunk.__table__.constraints if isinstance(constraint, CheckConstraint)}
    task_checks = {constraint.name for constraint in AgentTask.__table__.constraints if isinstance(constraint, CheckConstraint)}

    assert "uq_repository_files_path" in file_constraints
    assert "ck_repository_chunks_line_range" in chunk_checks
    assert "ck_agent_tasks_status" in task_checks
    assert User.__table__.c.password_hash.nullable
