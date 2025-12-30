"""Integration tests for RAGSystem."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
import pytest

# Add backend and tests to path
backend_path = Path(__file__).parent.parent
tests_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))
sys.path.insert(0, str(tests_path))

from helpers import create_text_response, create_tool_use_response


@dataclass
class MockConfig:
    """Mock configuration for testing."""
    ANTHROPIC_API_KEY: str = "test-api-key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"


class TestRAGSystemQuery:
    """Tests for RAGSystem.query method."""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_success_with_response(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vs, temp_chroma_path):
        """Test successful query returns response and sources."""
        from rag_system import RAGSystem

        config = MockConfig(CHROMA_PATH=temp_chroma_path)

        # Setup mocks
        mock_ai_gen.return_value.generate_response.return_value = "This is the answer"
        mock_vs.return_value.get_lesson_link.return_value = "https://example.com"

        rag = RAGSystem(config)

        # Manually set sources on the search tool
        rag.search_tool.last_sources = [{"text": "Course A - Lesson 1", "url": "https://example.com"}]

        response, sources = rag.query("What is machine learning?")

        assert response == "This is the answer"
        assert len(sources) == 1
        mock_ai_gen.return_value.generate_response.assert_called_once()

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_with_session_id(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vs, temp_chroma_path):
        """Test query with session_id updates conversation history."""
        from rag_system import RAGSystem

        config = MockConfig(CHROMA_PATH=temp_chroma_path)
        mock_ai_gen.return_value.generate_response.return_value = "Answer"
        mock_session.return_value.get_conversation_history.return_value = "Previous context"

        rag = RAGSystem(config)
        response, _ = rag.query("Follow-up question", session_id="session_123")

        # Verify history was retrieved
        mock_session.return_value.get_conversation_history.assert_called_with("session_123")

        # Verify exchange was added
        mock_session.return_value.add_exchange.assert_called_once()

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_without_session_id(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vs, temp_chroma_path):
        """Test query without session_id still works."""
        from rag_system import RAGSystem

        config = MockConfig(CHROMA_PATH=temp_chroma_path)
        mock_ai_gen.return_value.generate_response.return_value = "Answer"

        rag = RAGSystem(config)
        response, _ = rag.query("Question without session")

        assert response == "Answer"
        # History should not be retrieved
        mock_session.return_value.get_conversation_history.assert_not_called()

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_sources_reset_after_retrieval(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vs, temp_chroma_path):
        """Test that sources are cleared after query returns."""
        from rag_system import RAGSystem

        config = MockConfig(CHROMA_PATH=temp_chroma_path)
        mock_ai_gen.return_value.generate_response.return_value = "Answer"

        rag = RAGSystem(config)
        rag.search_tool.last_sources = [{"text": "Source", "url": "https://example.com"}]

        _, sources = rag.query("Question")

        # Sources should be returned
        assert len(sources) == 1

        # After query, sources should be reset
        assert rag.search_tool.last_sources == []

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_ai_generator_exception(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vs, temp_chroma_path):
        """Test that AI generator exceptions propagate."""
        from rag_system import RAGSystem

        config = MockConfig(CHROMA_PATH=temp_chroma_path)
        mock_ai_gen.return_value.generate_response.side_effect = Exception("API Error")

        rag = RAGSystem(config)

        with pytest.raises(Exception) as exc_info:
            rag.query("Question")

        assert "API Error" in str(exc_info.value)


class TestRAGSystemInitialization:
    """Tests for RAGSystem initialization."""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_initialization_creates_components(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vs, temp_chroma_path):
        """Test that RAGSystem creates all required components."""
        from rag_system import RAGSystem

        config = MockConfig(CHROMA_PATH=temp_chroma_path)
        rag = RAGSystem(config)

        # Verify all components are initialized
        mock_doc_proc.assert_called_once()
        mock_vs.assert_called_once()
        mock_ai_gen.assert_called_once()
        mock_session.assert_called_once()

        # Verify tools are registered
        assert len(rag.tool_manager.tools) == 2
        assert "search_course_content" in rag.tool_manager.tools
        assert "get_course_outline" in rag.tool_manager.tools

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_initialization_with_empty_api_key(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vs, temp_chroma_path):
        """Test initialization with empty API key (deferred failure)."""
        from rag_system import RAGSystem

        config = MockConfig(ANTHROPIC_API_KEY="", CHROMA_PATH=temp_chroma_path)

        # Initialization should succeed (API key not validated until query)
        rag = RAGSystem(config)

        # Verify AIGenerator was called with empty key
        mock_ai_gen.assert_called_with("", "claude-sonnet-4-20250514")


class TestDiagnoseQueryFailure:
    """Diagnostic tests to identify which component causes query failures."""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_step1_tool_manager_setup(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vs, temp_chroma_path):
        """Step 1: Verify ToolManager registers tools correctly."""
        from rag_system import RAGSystem

        config = MockConfig(CHROMA_PATH=temp_chroma_path)
        rag = RAGSystem(config)

        defs = rag.tool_manager.get_tool_definitions()
        assert len(defs) == 2

        tool_names = [d["name"] for d in defs]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_step2_ai_generator_receives_tools(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vs, temp_chroma_path):
        """Step 2: Verify AIGenerator receives tool definitions."""
        from rag_system import RAGSystem

        config = MockConfig(CHROMA_PATH=temp_chroma_path)
        mock_ai_gen.return_value.generate_response.return_value = "Test response"

        rag = RAGSystem(config)
        rag.query("Test query")

        # Verify generate_response was called with tools
        call_kwargs = mock_ai_gen.return_value.generate_response.call_args.kwargs
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 2

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_step3_tool_manager_passed_to_generator(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vs, temp_chroma_path):
        """Step 3: Verify tool_manager is passed for tool execution."""
        from rag_system import RAGSystem

        config = MockConfig(CHROMA_PATH=temp_chroma_path)
        mock_ai_gen.return_value.generate_response.return_value = "Test response"

        rag = RAGSystem(config)
        rag.query("Test query")

        call_kwargs = mock_ai_gen.return_value.generate_response.call_args.kwargs
        assert "tool_manager" in call_kwargs
        assert call_kwargs["tool_manager"] is rag.tool_manager


class TestRAGSystemWithRealComponents:
    """Integration tests with minimal mocking to test component interaction."""

    @patch('ai_generator.anthropic.Anthropic')
    def test_full_query_flow_with_real_tool_manager(self, mock_anthropic_class, temp_chroma_path):
        """Test query flow with real ToolManager but mocked API."""
        from rag_system import RAGSystem
        from vector_store import SearchResults

        config = MockConfig(CHROMA_PATH=temp_chroma_path)

        # Setup mock Anthropic client
        mock_client = mock_anthropic_class.return_value
        mock_client.messages.create.return_value = create_text_response("Test answer")

        with patch('rag_system.VectorStore') as mock_vs_class:
            # Setup VectorStore mock
            mock_vs = mock_vs_class.return_value
            mock_vs.search.return_value = SearchResults(
                documents=["Content"],
                metadata=[{"course_title": "Test", "lesson_number": 1}],
                distances=[0.5]
            )
            mock_vs.get_lesson_link.return_value = None

            rag = RAGSystem(config)
            response, sources = rag.query("What is AI?")

            assert response == "Test answer"

    @patch('ai_generator.anthropic.Anthropic')
    def test_full_query_flow_with_tool_use(self, mock_anthropic_class, temp_chroma_path):
        """Test query flow when Claude uses a tool."""
        from rag_system import RAGSystem
        from vector_store import SearchResults

        config = MockConfig(CHROMA_PATH=temp_chroma_path)

        # Setup mock - first returns tool_use, second returns final answer
        mock_client = mock_anthropic_class.return_value
        mock_client.messages.create.side_effect = [
            create_tool_use_response("search_course_content", {"query": "AI concepts"}),
            create_text_response("AI is about machine learning...")
        ]

        with patch('rag_system.VectorStore') as mock_vs_class:
            mock_vs = mock_vs_class.return_value
            mock_vs.search.return_value = SearchResults(
                documents=["AI fundamentals content"],
                metadata=[{"course_title": "AI Course", "lesson_number": 1}],
                distances=[0.3]
            )
            mock_vs.get_lesson_link.return_value = "https://example.com/ai-course/1"

            rag = RAGSystem(config)
            response, sources = rag.query("What is AI?")

            assert response == "AI is about machine learning..."
            # Verify search was called (tool was executed)
            mock_vs.search.assert_called_once()
