"""Helper functions for tests."""
from unittest.mock import MagicMock


def create_text_response(text: str):
    """Create mock Anthropic response with text content."""
    response = MagicMock()
    response.stop_reason = "end_turn"
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    response.content = [text_block]
    return response


def create_tool_use_response(tool_name: str, tool_input: dict, tool_id: str = "tool_123"):
    """Create mock Anthropic response with tool_use content."""
    response = MagicMock()
    response.stop_reason = "tool_use"
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = tool_name
    tool_block.input = tool_input
    tool_block.id = tool_id
    response.content = [tool_block]
    return response
