"""Tests for CourseSearchTool and ToolManager."""
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute method."""

    def test_execute_successful_search(self, mock_vector_store, sample_search_results):
        """Test execute returns formatted results on successful search."""
        mock_vector_store.search.return_value = sample_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="machine learning")

        assert "AI Course" in result
        assert "Lesson 1" in result
        mock_vector_store.search.assert_called_once_with(
            query="machine learning",
            course_name=None,
            lesson_number=None
        )

    def test_execute_empty_results(self, mock_vector_store, empty_search_results):
        """Test execute returns appropriate message for empty results."""
        mock_vector_store.search.return_value = empty_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_execute_empty_results_with_course_filter(self, mock_vector_store, empty_search_results):
        """Test execute message includes course filter info for empty results."""
        mock_vector_store.search.return_value = empty_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test", course_name="AI Course")

        assert "No relevant content found" in result
        assert "AI Course" in result

    def test_execute_empty_results_with_lesson_filter(self, mock_vector_store, empty_search_results):
        """Test execute message includes lesson filter info for empty results."""
        mock_vector_store.search.return_value = empty_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test", lesson_number=5)

        assert "No relevant content found" in result
        assert "lesson 5" in result

    def test_execute_with_error(self, mock_vector_store, error_search_results):
        """Test execute returns error message from VectorStore."""
        mock_vector_store.search.return_value = error_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test query")

        assert result == "Search error: connection failed"

    def test_execute_with_course_filter(self, mock_vector_store, sample_search_results):
        """Test execute passes course_name filter to VectorStore."""
        mock_vector_store.search.return_value = sample_search_results

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test", course_name="AI Course")

        mock_vector_store.search.assert_called_once_with(
            query="test",
            course_name="AI Course",
            lesson_number=None
        )

    def test_execute_with_lesson_filter(self, mock_vector_store, sample_search_results):
        """Test execute passes lesson_number filter to VectorStore."""
        mock_vector_store.search.return_value = sample_search_results

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test", lesson_number=3)

        mock_vector_store.search.assert_called_once_with(
            query="test",
            course_name=None,
            lesson_number=3
        )

    def test_execute_with_both_filters(self, mock_vector_store, sample_search_results):
        """Test execute passes both filters to VectorStore."""
        mock_vector_store.search.return_value = sample_search_results

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test", course_name="AI Course", lesson_number=2)

        mock_vector_store.search.assert_called_once_with(
            query="test",
            course_name="AI Course",
            lesson_number=2
        )


class TestCourseSearchToolFormatResults:
    """Tests for CourseSearchTool._format_results and source tracking."""

    def test_format_results_tracks_sources(self, mock_vector_store, sample_search_results):
        """Test that last_sources is populated after formatting results."""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test")

        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "AI Course - Lesson 1"
        assert tool.last_sources[0]["url"] == "https://example.com/lesson1"

    def test_format_results_multiple_chunks(self, mock_vector_store):
        """Test formatting with multiple result chunks."""
        multi_results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course A", "lesson_number": 2}
            ],
            distances=[0.3, 0.4]
        )
        mock_vector_store.search.return_value = multi_results
        mock_vector_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test")

        assert "[Course A - Lesson 1]" in result
        assert "[Course A - Lesson 2]" in result
        assert "Content 1" in result
        assert "Content 2" in result


class TestCourseSearchToolDefinition:
    """Tests for CourseSearchTool.get_tool_definition."""

    def test_get_tool_definition_schema(self, mock_vector_store):
        """Test tool definition matches expected schema for Anthropic."""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]


class TestToolManager:
    """Tests for ToolManager class."""

    def test_register_tool(self, mock_vector_store):
        """Test ToolManager registers tools correctly."""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)

        assert "search_course_content" in manager.tools

    def test_register_multiple_tools(self, mock_vector_store):
        """Test ToolManager can register multiple tools."""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)

        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        assert len(manager.tools) == 2
        assert "search_course_content" in manager.tools
        assert "get_course_outline" in manager.tools

    def test_get_tool_definitions(self, mock_vector_store):
        """Test ToolManager returns all tool definitions."""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_execute_tool_success(self, mock_vector_store, sample_search_results):
        """Test ToolManager executes registered tool."""
        mock_vector_store.search.return_value = sample_search_results
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="test")

        assert "AI Course" in result

    def test_execute_unknown_tool(self):
        """Test ToolManager returns error for unknown tool name."""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", query="test")

        assert "not found" in result.lower()

    def test_get_last_sources(self, mock_vector_store, sample_search_results):
        """Test ToolManager retrieves sources from tools."""
        mock_vector_store.search.return_value = sample_search_results
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute to populate sources
        manager.execute_tool("search_course_content", query="test")
        sources = manager.get_last_sources()

        assert len(sources) == 1

    def test_reset_sources(self, mock_vector_store, sample_search_results):
        """Test ToolManager clears sources from all tools."""
        mock_vector_store.search.return_value = sample_search_results
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute then reset
        manager.execute_tool("search_course_content", query="test")
        manager.reset_sources()

        sources = manager.get_last_sources()
        assert sources == []
