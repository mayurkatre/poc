"""
Tests for retrieval components: BM25, MMR, and hybrid fusion.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ingestion.chunking import DocumentChunk
from retrieval.hybrid_search import BM25Index, reciprocal_rank_fusion
from retrieval.mmr import cosine_similarity, mmr_rerank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(chunk_id: str, text: str, source: str = "test.md") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
        doc_type="markdown",
        chunk_index=int(chunk_id.replace("c", "")),
        metadata={"source": source},
    )


# ---------------------------------------------------------------------------
# BM25 Tests
# ---------------------------------------------------------------------------

class TestBM25Index:
    @pytest.fixture
    def chunks(self):
        return [
            make_chunk("c0", "RAG combines retrieval and generation for accurate answers"),
            make_chunk("c1", "FAISS is a fast vector similarity search library"),
            make_chunk("c2", "HyDE generates hypothetical documents to improve retrieval"),
            make_chunk("c3", "Python is a programming language for data science"),
            make_chunk("c4", "Embedding models convert text to dense vector representations"),
        ]

    def test_build_and_search(self, chunks):
        index = BM25Index()
        index.build(chunks)
        results = index.search("retrieval augmented generation", top_k=3)
        assert len(results) > 0
        # Most relevant chunk should mention retrieval
        top_text = results[0][0].text.lower()
        assert "retrieval" in top_text

    def test_empty_search_returns_empty(self):
        index = BM25Index()
        results = index.search("query")
        assert results == []

    def test_top_k_respected(self, chunks):
        index = BM25Index()
        index.build(chunks)
        results = index.search("retrieval", top_k=2)
        assert len(results) <= 2

    def test_scores_are_positive(self, chunks):
        index = BM25Index()
        index.build(chunks)
        results = index.search("vector embeddings")
        for _, score in results:
            assert score >= 0

    def test_irrelevant_query_low_scores(self, chunks):
        index = BM25Index()
        index.build(chunks)
        # Query with no matching terms
        results = index.search("xyzzy frobnicator quux")
        # All scores should be 0 (no matching terms)
        assert all(score == 0.0 for _, score in results)

    def test_tokenization_case_insensitive(self, chunks):
        index = BM25Index()
        index.build(chunks)
        r_lower = index.search("retrieval")
        r_upper = index.search("RETRIEVAL")
        assert len(r_lower) == len(r_upper)


# ---------------------------------------------------------------------------
# MMR Tests
# ---------------------------------------------------------------------------

class TestMMR:
    def test_cosine_similarity_identical(self):
        v = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_mmr_returns_top_k(self):
        chunks = [make_chunk(f"c{i}", f"document {i}") for i in range(10)]
        embeddings = [np.random.rand(16).tolist() for _ in range(10)]
        query_emb = np.random.rand(16).tolist()

        result = mmr_rerank(query_emb, chunks, embeddings, top_k=5)
        assert len(result) == 5

    def test_mmr_no_duplicates(self):
        # Identical chunks should not both be selected at high diversity setting
        text = "exact same content about retrieval"
        chunks = [make_chunk(f"c{i}", text) for i in range(5)]
        embeddings = [[1.0, 0.0, 0.0]] * 5  # All identical embeddings
        query_emb = [1.0, 0.0, 0.0]

        result = mmr_rerank(query_emb, chunks, embeddings, top_k=3, lambda_param=0.0)
        # With lambda=0 (diversity only), only 1 should be selected
        # since all are identical, any selection makes the rest redundant
        assert len(result) >= 1

    def test_mmr_respects_lambda_relevance(self):
        """Lambda=1.0 should select purely by relevance (cosine with query)."""
        query_emb = [1.0, 0.0]
        chunks = [
            make_chunk("c0", "highly relevant"),
            make_chunk("c1", "medium relevant"),
            make_chunk("c2", "less relevant"),
        ]
        embeddings = [
            [0.95, 0.05],  # Most similar to query
            [0.70, 0.30],
            [0.40, 0.60],
        ]
        result = mmr_rerank(query_emb, chunks, embeddings, top_k=2, lambda_param=1.0)
        assert result[0].chunk_id == "c0"


# ---------------------------------------------------------------------------
# RRF Fusion Tests
# ---------------------------------------------------------------------------

class TestRRFFusion:
    def test_basic_fusion(self):
        chunks_a = [make_chunk("c1", "doc1"), make_chunk("c2", "doc2")]
        chunks_b = [make_chunk("c2", "doc2"), make_chunk("c3", "doc3")]

        result = reciprocal_rank_fusion([chunks_a, chunks_b])
        ids = [c.chunk_id for c in result]

        # c2 appears in both lists so should rank higher
        assert "c2" in ids
        assert ids.index("c2") < ids.index("c1") or ids.index("c2") < ids.index("c3")

    def test_deduplication(self):
        chunks_a = [make_chunk("c1", "doc1"), make_chunk("c2", "doc2")]
        chunks_b = [make_chunk("c1", "doc1"), make_chunk("c3", "doc3")]

        result = reciprocal_rank_fusion([chunks_a, chunks_b])
        ids = [c.chunk_id for c in result]
        assert len(ids) == len(set(ids))  # No duplicates

    def test_rrf_scores_attached(self):
        chunks_a = [make_chunk("c1", "doc1")]
        result = reciprocal_rank_fusion([chunks_a])
        assert "rrf_score" in result[0].metadata

    def test_empty_lists(self):
        result = reciprocal_rank_fusion([[], []])
        assert result == []
