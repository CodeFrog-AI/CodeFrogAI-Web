"""add repository analyses

Revision ID: d5e9f2a7b3c1
Revises: c4d8e1f6a2b9
Create Date: 2026-09-19 20:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd5e9f2a7b3c1'
down_revision: Union[str, Sequence[str], None] = 'c4d8e1f6a2b9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the one-to-one repository analysis table."""
    op.create_table(
        'repository_analyses',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('repository_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('project_type', sa.String(length=64), nullable=False),
        sa.Column('languages', sa.JSON(), nullable=False),
        sa.Column('frameworks', sa.JSON(), nullable=False),
        sa.Column('package_managers', sa.JSON(), nullable=False),
        sa.Column('dependencies', sa.JSON(), nullable=False),
        sa.Column('important_files', sa.JSON(), nullable=False),
        sa.Column('entry_points', sa.JSON(), nullable=False),
        sa.Column('analysis_metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint(
            "status IN ('completed', 'partial', 'failed')", name='ck_repository_analyses_status'
        ),
        sa.ForeignKeyConstraint(['repository_id'], ['repositories.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('repository_id', name='uq_repository_analyses_repository_id'),
    )


def downgrade() -> None:
    """Drop the repository analysis table."""
    op.drop_table('repository_analyses')
