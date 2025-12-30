"""Tests for FastAPI endpoints."""
import pytest


class TestRootEndpoint:
    """Tests for the root endpoint."""

    @pytest.mark.api
    def test_root_returns_ok_status(self, client):
        """GET / should return status ok."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "message" in data


class TestQueryEndpoint:
    """Tests for the /api/query endpoint."""

    @pytest.mark.api
    def test_query_with_valid_request(self, client, sample_query_request, mock_rag_system):
        """POST /api/query with valid query returns answer and sources."""
        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "This is a test answer about the course."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "Test Course - Lesson 1"

    @pytest.mark.api
    def test_query_creates_session_when_not_provided(self, client, sample_query_request, mock_rag_system):
        """POST /api/query without session_id creates a new session."""
        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"
        mock_rag_system.session_manager.create_session.assert_called_once()

    @pytest.mark.api
    def test_query_uses_provided_session(self, client, sample_query_request_with_session, mock_rag_system):
        """POST /api/query with session_id uses the provided session."""
        response = client.post("/api/query", json=sample_query_request_with_session)

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing-session-456"
        mock_rag_system.query.assert_called_once_with(
            "Tell me more about neural networks",
            "existing-session-456"
        )

    @pytest.mark.api
    def test_query_missing_query_field(self, client):
        """POST /api/query without query field returns 422."""
        response = client.post("/api/query", json={})

        assert response.status_code == 422

    @pytest.mark.api
    def test_query_empty_query_string(self, client, mock_rag_system):
        """POST /api/query with empty query string is accepted."""
        response = client.post("/api/query", json={"query": ""})

        assert response.status_code == 200

    @pytest.mark.api
    def test_query_handles_rag_system_error(self, client, mock_rag_system):
        """POST /api/query returns 500 when RAG system raises exception."""
        mock_rag_system.query.side_effect = Exception("RAG system error")

        response = client.post("/api/query", json={"query": "test"})

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "RAG system error" in data["detail"]

    @pytest.mark.api
    def test_query_with_long_query(self, client, mock_rag_system):
        """POST /api/query handles long query strings."""
        long_query = "What is machine learning? " * 100
        response = client.post("/api/query", json={"query": long_query})

        assert response.status_code == 200


class TestCoursesEndpoint:
    """Tests for the /api/courses endpoint."""

    @pytest.mark.api
    def test_courses_returns_stats(self, client, mock_rag_system):
        """GET /api/courses returns course statistics."""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Python Basics" in data["course_titles"]
        assert "Machine Learning" in data["course_titles"]
        assert "Web Development" in data["course_titles"]

    @pytest.mark.api
    def test_courses_calls_get_course_analytics(self, client, mock_rag_system):
        """GET /api/courses calls get_course_analytics on RAG system."""
        client.get("/api/courses")

        mock_rag_system.get_course_analytics.assert_called_once()

    @pytest.mark.api
    def test_courses_handles_error(self, client, mock_rag_system):
        """GET /api/courses returns 500 when RAG system raises exception."""
        mock_rag_system.get_course_analytics.side_effect = Exception("Database error")

        response = client.get("/api/courses")

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Database error" in data["detail"]

    @pytest.mark.api
    def test_courses_empty_catalog(self, client, mock_rag_system):
        """GET /api/courses handles empty course catalog."""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []


class TestAPIResponseFormat:
    """Tests for API response format consistency."""

    @pytest.mark.api
    def test_query_response_has_correct_structure(self, client, sample_query_request):
        """Query response matches QueryResponse model."""
        response = client.post("/api/query", json=sample_query_request)
        data = response.json()

        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

        for source in data["sources"]:
            assert "text" in source
            assert isinstance(source["text"], str)

    @pytest.mark.api
    def test_courses_response_has_correct_structure(self, client):
        """Courses response matches CourseStats model."""
        response = client.get("/api/courses")
        data = response.json()

        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

        for title in data["course_titles"]:
            assert isinstance(title, str)

    @pytest.mark.api
    def test_error_response_format(self, client, mock_rag_system):
        """Error responses have consistent format."""
        mock_rag_system.query.side_effect = Exception("Test error")

        response = client.post("/api/query", json={"query": "test"})

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
