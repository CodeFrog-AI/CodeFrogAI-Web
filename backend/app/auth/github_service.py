"""Safe database account-linking logic for GitHub OAuth identities."""

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth.service import find_user_by_email
from app.core.exceptions import BadRequestError
from app.db.models import GitHubAccount, User
from app.integrations.github.oauth import GitHubIdentity


def find_github_account(session: Session, github_user_id: int) -> GitHubAccount | None:
    """Find the unique local link for a GitHub user ID."""

    return (
        session.query(GitHubAccount)
        .filter(GitHubAccount.github_user_id == github_user_id)
        .first()
    )


def resolve_github_identity(session: Session, identity: GitHubIdentity) -> User:
    """Authenticate an existing link or safely link/create the matching local user."""

    account = find_github_account(session, identity.github_user_id)
    if account is not None:
        user = session.get(User, account.user_id)
        if user is None:
            raise BadRequestError("GitHub authentication could not be completed")
        return user

    user = find_user_by_email(session, identity.email)
    if user is None:
        user = User(email=identity.email, name=identity.name, status="active")
        session.add(user)
        session.flush()

    account = GitHubAccount(
        user_id=user.id,
        github_user_id=identity.github_user_id,
        login=identity.login,
    )
    session.add(account)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise BadRequestError("GitHub authentication could not be completed") from None
    session.refresh(user)
    return user
