"""
Hybrid Search Module
====================
Combines dense vector retrieval with BM25 keyword search using
Reciprocal Rank Fusion (RRF) for score normalization.

Why hybrid?
  - Dense retrieval: great for semantic similarity, struggles with exact terms
  - BM25: great for exact keyword matching, struggles with paraphrasing
  - Combined: gets the best of both worlds

Architecture:
  Query → [Dense Search || BM25 Search] → RRF Fusion → Ranked Chunks
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import numpy as np
from loguru import logger

from ingestion.chunking import DocumentChunk
from ingestion.embedding_pipeline import BaseVectorStore, EmbeddingPipeline
from retrieval.base_retriever import BaseRetriever, RetrievalResult
from retrieval.mmr import mmr_rerank


class BM25Index:
    """
    In-memory BM25 index built from DocumentChunks.

    Uses standard BM25+ parameters (k1=1.5, b=0.75).
    Rebuilt on each load since it's fast and doesn't require persistence.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunks: list[DocumentChunk] = []
        self._tf: list[dict[str, float]] = []
        self._df: dict[str, int] = defaultdict(int)
        self._avg_dl: float = 0.0
        self._n: int = 0

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + lowercase tokenizer."""
        return text.lower().split()

    def build(self, chunks: list[DocumentChunk]) -> None:
        """
        Build the BM25 index from a list of chunks.

        Args:
            chunks: All indexed document chunks.
        """
        self._chunks = chunks
        self._n = len(chunks)
        total_dl = 0

        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            total_dl += len(tokens)
            tf: dict[str, float] = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            self._tf.append(dict(tf))
            for token in set(tokens):
                self._df[token] += 1

        self._avg_dl = total_dl / max(self._n, 1)
        logger.debug(f"BM25 index built: {self._n} documents, avg_dl={self._avg_dl:.1f}")

    def search(self, query: str, top_k: int = 20) -> list[tuple[DocumentChunk, float]]:
        """
        BM25 search for a query string.

        Args:
            query: Query text.
            top_k: Number of top results to return.

        Returns:
            List of (chunk, score) tuples sorted by relevance.
        """
        if not self._chunks:
            return []

        query_tokens = self._tokenize(query)
        scores = np.zeros(self._n)

        for token in query_tokens:
            if token not in self._df:
                continue

            idf = math.log(
                (self._n - self._df[token] + 0.5) / (self._df[token] + 0.5) + 1
            )

            for i, tf_dict in enumerate(self._tf):
                tf = tf_dict.get(token, 0)
                if tf == 0:
                    continue
                dl = sum(tf_dict.values())
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * dl / self._avg_dl
                )
                scores[i] += idf * (numerator / denominator)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (self._chunks[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]


def reciprocal_rank_fusion(
    ranked_lists: list[list[DocumentChunk]],
    k: int = 60,
) -> list[DocumentChunk]:
    """
    Reciprocal Rank Fusion (RRF) for combining multiple ranked lists.

    Score(d) = Σ 1 / (k + rank(d, list))

    Args:
        ranked_lists: Multiple ordered lists of chunks.
        k: RRF constant (default 60, from original paper).

    Returns:
        Fused and re-ranked list of unique chunks.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    chunk_map: dict[str, DocumentChunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            rrf_scores[chunk.chunk_id] += 1.0 / (k + rank)
            chunk_map[chunk.chunk_id] = chunk

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    result = []
    for chunk_id in sorted_ids:
        chunk = chunk_map[chunk_id]
        chunk.metadata["rrf_score"] = rrf_scores[chunk_id]
        result.append(chunk)

    return result


class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval combining dense vector search and BM25.

    Uses Reciprocal Rank Fusion to merge results, then optionally
    applies MMR to diversify the final selection.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_pipeline: EmbeddingPipeline,
        bm25_index: BM25Index,
        top_k: int = 20,
        final_top_k: int = 5,
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7,
        use_mmr: bool = True,
        mmr_lambda: float = 0.5,
    ):
        """
        Args:
            vector_store: Dense vector store.
            embedding_pipeline: For query embedding.
            bm25_index: Pre-built BM25 index.
            top_k: Initial candidates per retriever.
            final_top_k: Final number of results after fusion.
            bm25_weight: Weight applied to BM25 rankings.
            dense_weight: Weight applied to dense rankings.
            use_mmr: Whether to apply MMR for diversity.
            mmr_lambda: MMR lambda parameter.
        """
        super().__init__(vector_store, embedding_pipeline, top_k)
        self.bm25_index = bm25_index
        self.final_top_k = final_top_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Hybrid retrieval with RRF fusion and optional MMR.

        Args:
            query: User query string.

        Returns:
            RetrievalResult with fused and diversified chunks.
        """
        logger.info(f"Hybrid retrieval for: '{query[:80]}'")

        # --- Dense search ---
        query_embedding = self._embed_query(query)
        dense_results = self.vector_store.search(query_embedding, top_k=self.top_k)
        logger.debug(f"Dense: {len(dense_results)} results")

        # --- BM25 search ---
        bm25_results_raw = self.bm25_index.search(query, top_k=self.top_k)
        bm25_results = [chunk for chunk, _ in bm25_results_raw]
        logger.debug(f"BM25: {len(bm25_results)} results")

        # --- RRF Fusion ---
        fused = reciprocal_rank_fusion([dense_results, bm25_results])
        fused = fused[: self.top_k]
        logger.debug(f"After RRF: {len(fused)} unique chunks")

        # --- MMR for diversity ---
        if self.use_mmr and len(fused) > self.final_top_k:
            # Re-embed fused chunks for MMR
            fused_texts = [c.text for c in fused]
            fused_embeddings = self.embedding_pipeline.embedder.embed(fused_texts)
            fused = mmr_rerank(
                query_embedding=query_embedding,
                chunks=fused,
                chunk_embeddings=fused_embeddings,
                top_k=self.final_top_k,
                lambda_param=self.mmr_lambda,
            )
        else:
            fused = fused[: self.final_top_k]

        return RetrievalResult(
            chunks=fused,
            query=query,
            strategy="hybrid",
            metadata={
                "dense_count": len(dense_results),
                "bm25_count": len(bm25_results),
                "use_mmr": self.use_mmr,
            },
        )
