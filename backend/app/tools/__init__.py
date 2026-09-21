"""Agent tools and the registry an agent uses to discover and run them.

Read tools are always available. Write tools are only visible and runnable when the server
supplies an approved workspace."""

from typing import Any

from app.tools.analyze_project import ANALYZE_PROJECT
from app.tools.base import Tool, ToolContext, ToolError, ToolResult, failure
from app.tools.read_file import READ_FILE
from app.tools.search_code import SEARCH_CODE
from app.tools.write_tools import CREATE_FILE, DELETE_FILE, EDIT_FILE

TOOLS: tuple[Tool[Any], ...] = (SEARCH_CODE, READ_FILE, ANALYZE_PROJECT)
WRITE_TOOLS: tuple[Tool[Any], ...] = (EDIT_FILE, CREATE_FILE, DELETE_FILE)
_TOOLS_BY_NAME: dict[str, Tool[Any]] = {tool.name: tool for tool in (*TOOLS, *WRITE_TOOLS)}

__all__ = [
    "TOOLS",
    "WRITE_TOOLS",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolResult",
    "execute_tool",
    "get_tool",
    "tool_definitions",
]


def get_tool(name: str) -> Tool[Any] | None:
    return _TOOLS_BY_NAME.get(name)


def tool_definitions(include_write: bool = False) -> list[dict[str, Any]]:
    """Name, description, and JSON input schema of the tools, for model discovery."""

    return [tool.definition() for tool in (*TOOLS, *(WRITE_TOOLS if include_write else ()))]


def execute_tool(name: str, arguments: Any, context: ToolContext) -> ToolResult:
    """Run a tool by name; an unknown name is reported as an error result."""

    tool = get_tool(name)
    # Without an approved workspace, write tools are not merely refused: they do not exist.
    if tool is None or (tool.writes and context.workspace is None):
        available = [tool.name for tool in (*TOOLS, *(WRITE_TOOLS if context.workspace else ()))]
        return failure("UNKNOWN_TOOL", f"Unknown tool. Available tools: {', '.join(available)}")
    return tool.execute(context, arguments)
