"""add user ai settings

Revision ID: e6f0a3b8c4d2
Revises: d5e9f2a7b3c1
Create Date: 2026-09-21 20:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'e6f0a3b8c4d2'
down_revision: Union[str, Sequence[str], None] = 'd5e9f2a7b3c1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the per-user AI provider settings table (API keys are stored encrypted)."""
    op.create_table(
        'user_ai_settings',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('llm_api_key_encrypted', sa.Text(), nullable=True),
        sa.Column('llm_api_key_hint', sa.String(length=8), nullable=True),
        sa.Column('llm_model', sa.String(length=128), nullable=True),
        sa.Column('embedding_api_key_encrypted', sa.Text(), nullable=True),
        sa.Column('embedding_api_key_hint', sa.String(length=8), nullable=True),
        sa.Column('embedding_model', sa.String(length=128), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', name='uq_user_ai_settings_user_id'),
    )


def downgrade() -> None:
    """Drop the per-user AI provider settings table."""
    op.drop_table('user_ai_settings')
