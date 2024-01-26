"""
Tests for the FastAPI application endpoints.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from generation.rag_pipeline import RAGResponse, SourceReference


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_response():
    """Construct a fake RAGResponse for mocking the pipeline."""
    return RAGResponse(
        answer="RAG improves LLM reliability by grounding responses in retrieved documents.",
        sources=[
            SourceReference(
                document="architecture.md",
                chunk_id="abc123",
                text="RAG (Retrieval-Augmented Generation) combines retrieval with generation...",
                chunk_index=14,
                page_number=None,
            )
        ],
        query="What is RAG?",
        retrieval_strategy="hybrid",
        latency_ms=450.0,
    )


@pytest.fixture
def client(mock_response):
    """Create a test client with mocked pipeline."""
    # We need to mock the lifespan to avoid real model loading
    with patch("api.app._build_pipeline") as mock_build:
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = mock_response
        mock_build.return_value = (mock_pipeline, MagicMock(), MagicMock(), MagicMock())

        # Import and patch app_state
        from api.app import app, app_state
        app_state.pipeline = mock_pipeline
        app_state.vector_store = MagicMock()
        app_state.vector_store.__len__ = lambda self: 42

        yield TestClient(app)


# ---------------------------------------------------------------------------
# Health Endpoint Tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "indexed_chunks" in data

    def test_health_status_healthy(self, client):
        response = client.get("/health")
        assert response.json()["status"] == "healthy"


# ---------------------------------------------------------------------------
# Query Endpoint Tests
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    def test_query_returns_200(self, client):
        response = client.post("/query", json={"question": "What is RAG?"})
        assert response.status_code == 200

    def test_query_response_has_answer(self, client):
        response = client.post("/query", json={"question": "What is RAG?"})
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 0

    def test_query_response_has_sources(self, client):
        response = client.post("/query", json={"question": "What is RAG?"})
        data = response.json()
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_source_structure(self, client):
        response = client.post("/query", json={"question": "What is RAG?"})
        source = response.json()["sources"][0]
        assert "document" in source
        assert "chunk_id" in source
        assert "text" in source
        assert "chunk_index" in source

    def test_query_response_has_metadata(self, client):
        response = client.post("/query", json={"question": "What is RAG?"})
        data = response.json()
        assert "retrieval_strategy" in data
        assert "latency_ms" in data

    def test_empty_question_rejected(self, client):
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 422  # Pydantic validation error

    def test_too_short_question_rejected(self, client):
        response = client.post("/query", json={"question": "hi"})
        assert response.status_code == 422

    def test_missing_question_rejected(self, client):
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_pipeline_not_initialized_returns_503(self):
        """Test that 503 is returned when pipeline is not initialized."""
        from api.app import app, app_state
        app_state.pipeline = None

        test_client = TestClient(app)
        response = test_client.post("/query", json={"question": "What is RAG?"})
        assert response.status_code == 503


# ---------------------------------------------------------------------------
# Ingest Endpoint Tests
# ---------------------------------------------------------------------------

class TestIngestEndpoint:
    def test_ingest_starts_background_task(self, client):
        from api.app import app_state
        app_state.embedding_pipeline = MagicMock()

        response = client.post("/ingest", json={"source": "/tmp/test.txt"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ingestion_started"
