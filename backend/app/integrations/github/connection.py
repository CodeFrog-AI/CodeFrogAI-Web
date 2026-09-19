"""Verify GitHub repository access and connect repositories to CodeFrog."""

from typing import Protocol

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.exceptions import ConflictError, ForbiddenError, NotFoundError
from app.db.models import GitHubAccount, Repository, User
from app.integrations.github.contents import GitHubNotFoundError, GitHubRepository


class RepositoryClient(Protocol):
    def list_repositories(self) -> list[GitHubRepository]: ...

    def get_repository_by_id(self, github_repository_id: int) -> GitHubRepository: ...


def get_user_github_account(session: Session, user: User) -> GitHubAccount:
    """Return the user's linked GitHub account or raise a safe conflict error."""

    account = (
        session.query(GitHubAccount)
        .filter(GitHubAccount.user_id == user.id)
        .order_by(GitHubAccount.created_at)
        .first()
    )
    if account is None:
        raise ConflictError("No GitHub account is connected. Sign in with GitHub first.")
    return account


def connected_repository_ids(session: Session, account: GitHubAccount) -> set[int]:
    rows = session.query(Repository.github_repository_id).filter(
        Repository.github_account_id == account.id
    )
    return {github_id for (github_id,) in rows}


def connect_repository(
    session: Session, account: GitHubAccount, client: RepositoryClient, github_repository_id: int
) -> tuple[Repository, bool]:
    """Connect a repository the account can access; returns (repository, created).

    Access is proven by the repository appearing in the account's own GitHub
    repository list, never by trusting the supplied ID.
    """

    github_repository = next(
        (
            item
            for item in client.list_repositories()
            if item.github_repository_id == github_repository_id
        ),
        None,
    )
    if github_repository is None:
        try:
            client.get_repository_by_id(github_repository_id)
        except GitHubNotFoundError:
            raise NotFoundError("Repository was not found on GitHub") from None
        raise ForbiddenError("This repository is not accessible with your GitHub account")

    repository = _find_repository(session, github_repository_id)
    created = repository is None
    if repository is None:
        repository = Repository(github_account_id=account.id, github_repository_id=github_repository_id)
        session.add(repository)
    elif repository.github_account.user_id != account.user_id:
        raise ConflictError("This repository is already connected to another account")

    repository.owner = github_repository.owner
    repository.name = github_repository.name
    repository.default_branch = github_repository.default_branch
    repository.connection_metadata = {"private": github_repository.private}
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise ConflictError("This repository could not be connected") from None
    session.refresh(repository)
    return repository, created


def _find_repository(session: Session, github_repository_id: int) -> Repository | None:
    return (
        session.query(Repository)
        .filter(Repository.github_repository_id == github_repository_id)
        .first()
    )
