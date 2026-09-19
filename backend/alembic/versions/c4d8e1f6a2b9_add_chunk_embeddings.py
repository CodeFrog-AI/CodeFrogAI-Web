"""add chunk embeddings

Revision ID: c4d8e1f6a2b9
Revises: 1f4e7c9a2b31
Create Date: 2026-09-19 18:00:00.000000

"""
from typing import Sequence, Union

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c4d8e1f6a2b9'
down_revision: Union[str, Sequence[str], None] = '1f4e7c9a2b31'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add nullable embedding columns; existing chunks simply have no embedding yet."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column(
        'repository_chunks',
        sa.Column('embedding', pgvector.sqlalchemy.Vector(dim=1536), nullable=True),
    )
    op.add_column(
        'repository_chunks',
        sa.Column('embedding_content_hash', sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    """Drop the embedding columns (the vector extension is left installed)."""
    op.drop_column('repository_chunks', 'embedding_content_hash')
    op.drop_column('repository_chunks', 'embedding')
