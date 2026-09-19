"""Read-only agent tools and the registry an agent uses to discover and run them."""

from typing import Any

from app.tools.analyze_project import ANALYZE_PROJECT
from app.tools.base import Tool, ToolContext, ToolError, ToolResult, failure
from app.tools.read_file import READ_FILE
from app.tools.search_code import SEARCH_CODE

TOOLS: tuple[Tool[Any], ...] = (SEARCH_CODE, READ_FILE, ANALYZE_PROJECT)
_TOOLS_BY_NAME: dict[str, Tool[Any]] = {tool.name: tool for tool in TOOLS}

__all__ = [
    "TOOLS",
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


def tool_definitions() -> list[dict[str, Any]]:
    """Name, description, and JSON input schema of every tool, for model discovery."""

    return [tool.definition() for tool in TOOLS]


def execute_tool(name: str, arguments: Any, context: ToolContext) -> ToolResult:
    """Run a tool by name; an unknown name is reported as an error result."""

    tool = get_tool(name)
    if tool is None:
        return failure("UNKNOWN_TOOL", f"Unknown tool. Available tools: {', '.join(_TOOLS_BY_NAME)}")
    return tool.execute(context, arguments)
