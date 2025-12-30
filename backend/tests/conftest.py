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


# --- API Testing Fixtures ---

@pytest.fixture
def mock_rag_system():
    """Mock RAGSystem for API tests."""
    rag = MagicMock()
    rag.query.return_value = (
        "This is a test answer about the course.",
        [{"text": "Test Course - Lesson 1", "url": "https://example.com/lesson1"}]
    )
    rag.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Python Basics", "Machine Learning", "Web Development"]
    }
    rag.session_manager.create_session.return_value = "test-session-123"
    return rag


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app without static file mounting."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional

    # Create a minimal test app with just the API endpoints
    app = FastAPI(title="Course Materials RAG System - Test")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Pydantic models (matching app.py)
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class Source(BaseModel):
        text: str
        url: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Source]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    # Store the mock so tests can access it
    app.state.rag_system = mock_rag_system

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        return {"status": "ok", "message": "Course Materials RAG System"}

    return app


@pytest.fixture
def client(test_app):
    """FastAPI TestClient for API testing."""
    from fastapi.testclient import TestClient
    return TestClient(test_app)


@pytest.fixture
def sample_query_request():
    """Sample query request data."""
    return {"query": "What is machine learning?"}


@pytest.fixture
def sample_query_request_with_session():
    """Sample query request with session ID."""
    return {
        "query": "Tell me more about neural networks",
        "session_id": "existing-session-456"
    }


# --- Fixtures ---

@pytest.fixture
def sample_search_results():
    """Pre-built SearchResults for various test scenarios."""
    return SearchResults(
        documents=["This is lesson content about machine learning."],
        metadata=[{"course_title": "AI Course", "lesson_number": 1, "chunk_index": 0}],
        distances=[0.5]
    )


@pytest.fixture
def empty_search_results():
    """Empty SearchResults for no matches."""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


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
        'documents': [['Content chunk 1']],
        'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 1}]],
        'distances': [[0.5]]
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
    manager.get_last_sources.return_value = [{"text": "AI Course - Lesson 1", "url": "https://example.com"}]
    return manager
