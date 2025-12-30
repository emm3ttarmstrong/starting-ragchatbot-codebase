"""Tests for VectorStore class."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from vector_store import VectorStore, SearchResults


class TestSearchResults:
    """Tests for SearchResults dataclass."""

    def test_from_chroma_with_results(self):
        """Test creating SearchResults from valid ChromaDB results."""
        chroma_results = {
            'documents': [['Doc 1', 'Doc 2']],
            'metadatas': [[{'course_title': 'Course A'}, {'course_title': 'Course B'}]],
            'distances': [[0.1, 0.2]]
        }
        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == ['Doc 1', 'Doc 2']
        assert len(results.metadata) == 2
        assert results.distances == [0.1, 0.2]
        assert results.error is None

    def test_from_chroma_empty_results(self):
        """Test creating SearchResults from empty ChromaDB results."""
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == []
        assert results.metadata == []
        assert results.is_empty()

    def test_from_chroma_none_fields(self):
        """Test creating SearchResults when ChromaDB returns empty outer lists."""
        chroma_results = {
            'documents': [],
            'metadatas': [],
            'distances': []
        }
        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == []
        assert results.is_empty()

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message."""
        results = SearchResults.empty("Test error")

        assert results.is_empty()
        assert results.error == "Test error"


class TestVectorStoreResolveCourseName:
    """Tests for VectorStore._resolve_course_name method."""

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_resolve_course_name_success(self, mock_embedding, mock_client, temp_chroma_path):
        """Test successful course name resolution."""
        # Setup mock collection
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'documents': [['AI Fundamentals Course']],
            'metadatas': [[{'title': 'AI Fundamentals Course', 'instructor': 'John'}]]
        }
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)
        result = store._resolve_course_name("AI course")

        assert result == "AI Fundamentals Course"
        mock_collection.query.assert_called_once()

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_resolve_course_name_empty_catalog(self, mock_embedding, mock_client, temp_chroma_path):
        """
        BUG TEST: Empty catalog should return None, not raise IndexError.
        Tests lines 110-112 in vector_store.py.
        """
        # Setup mock to return empty results
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)
        result = store._resolve_course_name("nonexistent")

        # Should return None gracefully, not raise error
        assert result is None

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_resolve_course_name_completely_empty_results(self, mock_embedding, mock_client, temp_chroma_path):
        """
        BUG TEST: Completely empty results (empty outer lists) should not raise IndexError.
        """
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'documents': [],
            'metadatas': []
        }
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)

        # This should not raise an IndexError
        try:
            result = store._resolve_course_name("test")
            # If no exception, result should be None
            assert result is None
        except IndexError as e:
            pytest.fail(f"IndexError raised when accessing empty results: {e}")

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_resolve_course_name_exception_handling(self, mock_embedding, mock_client, temp_chroma_path):
        """
        Test that exceptions in _resolve_course_name are caught and return None.
        Tests lines 113-114 in vector_store.py.
        """
        mock_collection = MagicMock()
        mock_collection.query.side_effect = Exception("Database connection failed")
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)
        result = store._resolve_course_name("test")

        # Should return None, not raise exception
        assert result is None


class TestVectorStoreSearch:
    """Tests for VectorStore.search method."""

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_returns_results(self, mock_embedding, mock_client, temp_chroma_path):
        """Test basic search returns results."""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'documents': [['Machine learning is...']],
            'metadatas': [[{'course_title': 'AI Course', 'lesson_number': 1}]],
            'distances': [[0.3]]
        }
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)
        results = store.search("what is machine learning")

        assert not results.is_empty()
        assert "Machine learning" in results.documents[0]
        assert results.error is None

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_empty_collection(self, mock_embedding, mock_client, temp_chroma_path):
        """Test search on empty collection returns empty results."""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)
        results = store.search("anything")

        assert results.is_empty()
        assert results.error is None  # Empty is not an error

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_course_filter_not_found(self, mock_embedding, mock_client, temp_chroma_path):
        """Test search with course filter that doesn't match returns error."""
        mock_catalog = MagicMock()
        mock_catalog.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }

        mock_content = MagicMock()
        mock_content.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }

        def get_collection(name, **kwargs):
            if name == "course_catalog":
                return mock_catalog
            return mock_content

        mock_client.return_value.get_or_create_collection.side_effect = get_collection

        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)
        results = store.search("test", course_name="Nonexistent Course")

        assert results.is_empty()
        assert results.error is not None
        assert "No course found" in results.error

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_chroma_exception(self, mock_embedding, mock_client, temp_chroma_path):
        """
        Test that ChromaDB exceptions are caught and return error SearchResults.
        Tests lines 99-100 in vector_store.py.
        """
        mock_collection = MagicMock()
        mock_collection.query.side_effect = Exception("ChromaDB connection error")
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)
        results = store.search("test query")

        assert results.is_empty()
        assert results.error is not None
        assert "Search error" in results.error


class TestVectorStoreBuildFilter:
    """Tests for VectorStore._build_filter method."""

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_build_filter_no_params(self, mock_embedding, mock_client, temp_chroma_path):
        """Test filter with no parameters returns None."""
        mock_client.return_value.get_or_create_collection.return_value = MagicMock()

        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)
        result = store._build_filter(None, None)

        assert result is None

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_build_filter_course_only(self, mock_embedding, mock_client, temp_chroma_path):
        """Test filter with only course_title."""
        mock_client.return_value.get_or_create_collection.return_value = MagicMock()

        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)
        result = store._build_filter("AI Course", None)

        assert result == {"course_title": "AI Course"}

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_build_filter_lesson_only(self, mock_embedding, mock_client, temp_chroma_path):
        """Test filter with only lesson_number."""
        mock_client.return_value.get_or_create_collection.return_value = MagicMock()

        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)
        result = store._build_filter(None, 1)

        assert result == {"lesson_number": 1}

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_build_filter_both_params(self, mock_embedding, mock_client, temp_chroma_path):
        """Test filter with both course_title and lesson_number."""
        mock_client.return_value.get_or_create_collection.return_value = MagicMock()

        store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", 5)
        result = store._build_filter("AI Course", 2)

        assert result == {
            "$and": [
                {"course_title": "AI Course"},
                {"lesson_number": 2}
            ]
        }
