"""Shared fixtures and mocks for RAG chatbot tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from vector_store import SearchResults

# --- Helper functions for creating mock responses ---


def create_text_response(text: str):
    """Create mock Anthropic response with text content."""
    response = MagicMock()
    response.stop_reason = "end_turn"
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    response.content = [text_block]
    return response


def create_tool_use_response(
    tool_name: str, tool_input: dict, tool_id: str = "tool_123"
):
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


# --- Fixtures ---


@pytest.fixture
def sample_search_results():
    """Pre-built SearchResults for various test scenarios."""
    return SearchResults(
        documents=["This is lesson content about machine learning."],
        metadata=[{"course_title": "AI Course", "lesson_number": 1, "chunk_index": 0}],
        distances=[0.5],
    )


@pytest.fixture
def empty_search_results():
    """Empty SearchResults for no matches."""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    """SearchResults with error."""
    return SearchResults.empty("Search error: connection failed")


@pytest.fixture
def mock_vector_store(sample_search_results):
    """Mock VectorStore with controllable search results."""
    store = MagicMock()
    store.search.return_value = sample_search_results
    store.get_lesson_link.return_value = "https://example.com/lesson1"
    store._resolve_course_name.return_value = "AI Course"
    return store


@pytest.fixture
def mock_chroma_collection():
    """Mock ChromaDB collection for VectorStore tests."""
    collection = MagicMock()
    collection.query.return_value = {
        "documents": [["Content chunk 1"]],
        "metadatas": [[{"course_title": "Test Course", "lesson_number": 1}]],
        "distances": [[0.5]],
    }
    return collection


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client with controllable responses."""
    client = MagicMock()
    # Default to returning a text response
    client.messages.create.return_value = create_text_response("Test response")
    return client


@pytest.fixture
def temp_chroma_path(tmp_path):
    """Temporary ChromaDB path for integration tests."""
    return str(tmp_path / "test_chroma_db")


@pytest.fixture
def mock_tool_manager():
    """Mock ToolManager for AIGenerator tests."""
    manager = MagicMock()
    manager.execute_tool.return_value = "Tool result: Found relevant content"
    manager.get_last_sources.return_value = [
        {"text": "AI Course - Lesson 1", "url": "https://example.com"}
    ]
    return manager
