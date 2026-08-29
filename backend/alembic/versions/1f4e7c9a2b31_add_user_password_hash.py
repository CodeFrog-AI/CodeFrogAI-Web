"""add user password hash

Revision ID: 1f4e7c9a2b31
Revises: 8a38dcec7099
Create Date: 2026-08-30
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "1f4e7c9a2b31"
down_revision: Union[str, Sequence[str], None] = "8a38dcec7099"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add a nullable hash so existing users are preserved safely."""

    op.add_column("users", sa.Column("password_hash", sa.String(length=512), nullable=True))


def downgrade() -> None:
    """Remove the local-password hash column."""

    op.drop_column("users", "password_hash")
