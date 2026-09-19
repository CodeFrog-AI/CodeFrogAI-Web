"""The local working tree that approved agent edits are written to."""

from app.workspace.workspace import (
    FileChange,
    FileOperation,
    Workspace,
    WorkspaceError,
    WriteScope,
    exclusive_workspace,
    get_workspace_root,
    validate_path,
)

__all__ = [
    "FileChange",
    "FileOperation",
    "Workspace",
    "WorkspaceError",
    "WriteScope",
    "exclusive_workspace",
    "get_workspace_root",
    "validate_path",
]
