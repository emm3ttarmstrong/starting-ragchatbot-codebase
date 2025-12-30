"""Tests for AIGenerator class."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add backend and tests to path
backend_path = Path(__file__).parent.parent
tests_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))
sys.path.insert(0, str(tests_path))

from helpers import create_text_response, create_tool_use_response


class TestAIGeneratorGenerateResponse:
    """Tests for AIGenerator.generate_response method."""

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_direct_answer(self, mock_anthropic_class):
        """Test response with no tool_use returns text directly."""
        from ai_generator import AIGenerator

        # Setup mock
        mock_client = mock_anthropic_class.return_value
        mock_client.messages.create.return_value = create_text_response("Direct answer")

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = generator.generate_response("What is 2+2?")

        assert result == "Direct answer"
        mock_client.messages.create.assert_called_once()

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_with_tools_no_use(self, mock_anthropic_class):
        """Test response with tools available but not used."""
        from ai_generator import AIGenerator

        mock_client = mock_anthropic_class.return_value
        mock_client.messages.create.return_value = create_text_response(
            "Answer without tools"
        )

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        tools = [
            {
                "name": "search",
                "description": "Search",
                "input_schema": {"type": "object"},
            }
        ]
        result = generator.generate_response("Hello", tools=tools)

        assert result == "Answer without tools"

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_triggers_tool_use(
        self, mock_anthropic_class, mock_tool_manager
    ):
        """Test that tool_use response triggers tool execution."""
        from ai_generator import AIGenerator

        mock_client = mock_anthropic_class.return_value
        # First call returns tool_use, second returns final answer
        mock_client.messages.create.side_effect = [
            create_tool_use_response(
                "search_course_content", {"query": "machine learning"}
            ),
            create_text_response("Final answer after tool use"),
        ]

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        tools = [
            {
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {"type": "object"},
            }
        ]
        result = generator.generate_response(
            "What is ML?", tools=tools, tool_manager=mock_tool_manager
        )

        assert result == "Final answer after tool use"
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="machine learning"
        )

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_with_conversation_history(self, mock_anthropic_class):
        """Test that conversation history is included in system prompt."""
        from ai_generator import AIGenerator

        mock_client = mock_anthropic_class.return_value
        mock_client.messages.create.return_value = create_text_response(
            "Response with context"
        )

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "Follow-up question", conversation_history="User: Hi\nAssistant: Hello!"
        )

        # Verify system prompt includes history
        call_args = mock_client.messages.create.call_args
        system_content = call_args.kwargs.get("system", "")
        assert "Previous conversation" in system_content
        assert "User: Hi" in system_content

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_api_error(self, mock_anthropic_class):
        """Test that API errors propagate correctly."""
        from ai_generator import AIGenerator

        mock_client = mock_anthropic_class.return_value
        mock_client.messages.create.side_effect = Exception("API rate limit exceeded")

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")

        with pytest.raises(Exception) as exc_info:
            generator.generate_response("Test query")

        assert "API rate limit exceeded" in str(exc_info.value)

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_missing_api_key(self, mock_anthropic_class):
        """Test behavior with empty API key."""
        from ai_generator import AIGenerator

        mock_client = mock_anthropic_class.return_value
        mock_client.messages.create.side_effect = Exception(
            "Authentication failed: API key required"
        )

        generator = AIGenerator("", "claude-sonnet-4-20250514")

        with pytest.raises(Exception) as exc_info:
            generator.generate_response("Test")

        assert "Authentication" in str(exc_info.value) or "API key" in str(
            exc_info.value
        )


class TestAIGeneratorToolExecution:
    """Tests for tool execution behavior in AIGenerator."""

    @patch("ai_generator.anthropic.Anthropic")
    def test_handle_tool_execution_single_tool(
        self, mock_anthropic_class, mock_tool_manager
    ):
        """Test execution of a single tool call."""
        from ai_generator import AIGenerator

        mock_client = mock_anthropic_class.return_value
        mock_client.messages.create.side_effect = [
            create_tool_use_response("search_course_content", {"query": "test"}),
            create_text_response("Tool result incorporated"),
        ]

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        tools = [
            {
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {"type": "object"},
            }
        ]
        result = generator.generate_response(
            "Query", tools=tools, tool_manager=mock_tool_manager
        )

        assert result == "Tool result incorporated"
        assert mock_client.messages.create.call_count == 2

    @patch("ai_generator.anthropic.Anthropic")
    def test_handle_tool_execution_tool_returns_error(
        self, mock_anthropic_class, mock_tool_manager
    ):
        """Test handling when tool returns an error string."""
        from ai_generator import AIGenerator

        mock_tool_manager.execute_tool.return_value = "Error: No courses found"

        mock_client = mock_anthropic_class.return_value
        mock_client.messages.create.side_effect = [
            create_tool_use_response("search_course_content", {"query": "test"}),
            create_text_response("No results found based on search"),
        ]

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        tools = [
            {
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {"type": "object"},
            }
        ]
        result = generator.generate_response(
            "Query", tools=tools, tool_manager=mock_tool_manager
        )

        # Should still get a response, even with error from tool
        assert result == "No results found based on search"

    @patch("ai_generator.anthropic.Anthropic")
    def test_handle_tool_execution_tool_raises_exception(
        self, mock_anthropic_class, mock_tool_manager
    ):
        """
        Test that tool exceptions are caught and handled gracefully.
        The error message should be passed to Claude as a tool result.
        """
        from ai_generator import AIGenerator

        # Make tool raise an exception
        mock_tool_manager.execute_tool.side_effect = Exception(
            "Tool crashed unexpectedly"
        )

        mock_client = mock_anthropic_class.return_value
        mock_client.messages.create.side_effect = [
            create_tool_use_response("search_course_content", {"query": "test"}),
            create_text_response("I encountered an error while searching"),
        ]

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        tools = [
            {
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {"type": "object"},
            }
        ]

        # Should NOT raise - exception is caught and passed to Claude
        result = generator.generate_response(
            "Query", tools=tools, tool_manager=mock_tool_manager
        )

        assert result == "I encountered an error while searching"

        # Verify tool result contains error message
        second_call = mock_client.messages.create.call_args_list[1]
        messages = second_call.kwargs.get("messages", [])
        tool_result_content = messages[2]["content"][0]["content"]
        assert "Tool execution error" in tool_result_content
        assert "Tool crashed unexpectedly" in tool_result_content

    @patch("ai_generator.anthropic.Anthropic")
    def test_handle_tool_execution_message_format(
        self, mock_anthropic_class, mock_tool_manager
    ):
        """Test that tool results are formatted correctly for the API."""
        from ai_generator import AIGenerator

        mock_client = mock_anthropic_class.return_value
        mock_client.messages.create.side_effect = [
            create_tool_use_response(
                "search_course_content", {"query": "test"}, tool_id="tool_abc"
            ),
            create_text_response("Final response"),
        ]
        mock_tool_manager.execute_tool.return_value = "Tool found: relevant content"

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        tools = [
            {
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {"type": "object"},
            }
        ]
        generator.generate_response(
            "Query", tools=tools, tool_manager=mock_tool_manager
        )

        # Check the second API call includes tool results
        second_call = mock_client.messages.create.call_args_list[1]
        messages = second_call.kwargs.get("messages", [])

        # Should have: user message, assistant tool_use, user tool_result
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

        # Verify tool result format
        tool_results = messages[2]["content"]
        assert len(tool_results) == 1
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "tool_abc"
        assert tool_results[0]["content"] == "Tool found: relevant content"


class TestAIGeneratorConfiguration:
    """Tests for AIGenerator initialization and configuration."""

    @patch("ai_generator.anthropic.Anthropic")
    def test_initialization(self, mock_anthropic_class):
        """Test AIGenerator initializes with correct parameters."""
        from ai_generator import AIGenerator

        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")

        assert generator.model == "claude-sonnet-4-20250514"
        mock_anthropic_class.assert_called_once_with(api_key="test-api-key")

    @patch("ai_generator.anthropic.Anthropic")
    def test_base_params(self, mock_anthropic_class):
        """Test base API parameters are set correctly."""
        from ai_generator import AIGenerator

        generator = AIGenerator("key", "claude-sonnet-4-20250514")

        assert generator.base_params["model"] == "claude-sonnet-4-20250514"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800


class TestAIGeneratorSequentialToolCalling:
    """Tests for sequential tool calling (max 2 rounds)."""

    @patch("ai_generator.anthropic.Anthropic")
    def test_two_sequential_tool_calls(self, mock_anthropic_class, mock_tool_manager):
        """Two tool calls execute, then final response returned."""
        from ai_generator import AIGenerator

        mock_client = mock_anthropic_class.return_value
        # Round 1: tool_use, Round 2: tool_use, Final: text
        mock_client.messages.create.side_effect = [
            create_tool_use_response(
                "get_course_outline", {"course_title": "AI Course"}, tool_id="t1"
            ),
            create_tool_use_response(
                "search_course_content", {"query": "machine learning"}, tool_id="t2"
            ),
            create_text_response("Final answer after two tool calls"),
        ]

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        tools = [
            {
                "name": "get_course_outline",
                "description": "Get outline",
                "input_schema": {"type": "object"},
            },
            {
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {"type": "object"},
            },
        ]
        result = generator.generate_response(
            "Query", tools=tools, tool_manager=mock_tool_manager
        )

        assert result == "Final answer after two tool calls"
        assert mock_tool_manager.execute_tool.call_count == 2
        assert mock_client.messages.create.call_count == 3

    @patch("ai_generator.anthropic.Anthropic")
    def test_max_rounds_enforced(self, mock_anthropic_class, mock_tool_manager):
        """Loop exits after 2 rounds even if Claude wants more tools."""
        from ai_generator import AIGenerator

        mock_client = mock_anthropic_class.return_value
        # Claude keeps requesting tools, but should be forced to respond after 2 rounds
        mock_client.messages.create.side_effect = [
            create_tool_use_response("search", {"q": "1"}, tool_id="t1"),
            create_tool_use_response("search", {"q": "2"}, tool_id="t2"),
            create_text_response("Forced final response"),
        ]

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        tools = [
            {
                "name": "search",
                "description": "Search",
                "input_schema": {"type": "object"},
            }
        ]
        result = generator.generate_response(
            "Query", tools=tools, tool_manager=mock_tool_manager
        )

        assert result == "Forced final response"
        # Only 2 tool executions despite wanting more
        assert mock_tool_manager.execute_tool.call_count == 2

        # Verify final call has no tools
        final_call = mock_client.messages.create.call_args_list[2]
        assert "tools" not in final_call.kwargs

    @patch("ai_generator.anthropic.Anthropic")
    def test_early_exit_no_tool_use(self, mock_anthropic_class, mock_tool_manager):
        """Immediate return when Claude doesn't request more tools after first."""
        from ai_generator import AIGenerator

        mock_client = mock_anthropic_class.return_value
        # First call uses tool, second doesn't
        mock_client.messages.create.side_effect = [
            create_tool_use_response("search", {"query": "test"}, tool_id="t1"),
            create_text_response("Done after one tool"),
        ]

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        tools = [
            {
                "name": "search",
                "description": "Search",
                "input_schema": {"type": "object"},
            }
        ]
        result = generator.generate_response(
            "Query", tools=tools, tool_manager=mock_tool_manager
        )

        assert result == "Done after one tool"
        assert mock_tool_manager.execute_tool.call_count == 1
        assert mock_client.messages.create.call_count == 2

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_error_passed_to_claude(self, mock_anthropic_class, mock_tool_manager):
        """Tool errors included in results, loop continues."""
        from ai_generator import AIGenerator

        # First tool fails, but loop continues
        mock_tool_manager.execute_tool.side_effect = [
            Exception("Tool crashed"),
            "Success result",
        ]

        mock_client = mock_anthropic_class.return_value
        mock_client.messages.create.side_effect = [
            create_tool_use_response("tool1", {"query": "q1"}, tool_id="t1"),
            create_tool_use_response("tool2", {"query": "q2"}, tool_id="t2"),
            create_text_response("Handled error and got result"),
        ]

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        tools = [{"name": "tool1"}, {"name": "tool2"}]
        result = generator.generate_response(
            "Query", tools=tools, tool_manager=mock_tool_manager
        )

        assert result == "Handled error and got result"

        # Verify error was passed in first tool result
        second_call = mock_client.messages.create.call_args_list[1]
        messages = second_call.kwargs.get("messages", [])
        tool_result_content = messages[2]["content"][0]["content"]
        assert "Tool execution error" in tool_result_content
        assert "Tool crashed" in tool_result_content

    @patch("ai_generator.anthropic.Anthropic")
    def test_message_accumulation(self, mock_anthropic_class, mock_tool_manager):
        """Messages accumulate correctly across rounds."""
        from ai_generator import AIGenerator

        mock_client = mock_anthropic_class.return_value
        mock_client.messages.create.side_effect = [
            create_tool_use_response("search", {"q": "1"}, tool_id="t1"),
            create_tool_use_response("search", {"q": "2"}, tool_id="t2"),
            create_text_response("Final"),
        ]

        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        tools = [{"name": "search"}]
        generator.generate_response(
            "Query", tools=tools, tool_manager=mock_tool_manager
        )

        # Check final call has accumulated messages
        final_call = mock_client.messages.create.call_args_list[2]
        messages = final_call.kwargs["messages"]

        # Should have: user, assistant, user (tool result), assistant, user (tool result)
        assert len(messages) == 5
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"
