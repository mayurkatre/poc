"""
Cross-Encoder Reranker
======================
Uses a cross-encoder model to score (query, document) pairs jointly.

Why cross-encoders?
  Bi-encoders (used for initial retrieval) encode query and document
  independently — fast but less precise. Cross-encoders attend over
  BOTH query and document simultaneously — slow but much more accurate.

  Strategy: Retrieve Top-20 fast, Rerank Top-5 accurate.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  Fine-tuned on MS MARCO passage ranking task.
  Outputs a relevance score (higher = more relevant).
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from ingestion.chunking import DocumentChunk


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder model.

    Takes the top-k candidates from initial retrieval and produces
    a more accurate relevance ranking for the final generation context.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: HuggingFace cross-encoder model identifier.
            top_n: Number of documents to keep after reranking.
            device: Compute device ('cpu', 'cuda', or None for auto-detect).
        """
        self.model_name = model_name
        self.top_n = top_n
        self._model = None
        self._device = device

    def _get_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(
                    self.model_name,
                    device=self._device,
                    max_length=512,
                )
                logger.info(f"Loaded cross-encoder: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "Install sentence-transformers: pip install sentence-transformers"
                )
        return self._model

    def rerank(
        self, query: str, chunks: list[DocumentChunk]
    ) -> list[DocumentChunk]:
        """
        Rerank a list of chunks by their relevance to the query.

        Args:
            query: The user query.
            chunks: Retrieved chunks to rerank.

        Returns:
            Top-N chunks sorted by cross-encoder relevance score (descending).
        """
        if not chunks:
            return []

        if len(chunks) <= self.top_n:
            logger.debug(f"Reranker: {len(chunks)} ≤ top_n={self.top_n}, skipping.")
            return chunks

        model = self._get_model()

        # Build (query, chunk_text) pairs
        pairs = [(query, chunk.text) for chunk in chunks]

        # Score all pairs
        scores = model.predict(pairs, show_progress_bar=False)

        # Attach scores and sort
        scored = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        top_chunks = []
        for score, chunk in scored[: self.top_n]:
            chunk.metadata["rerank_score"] = float(score)
            top_chunks.append(chunk)

        logger.info(
            f"Reranker: {len(chunks)} → {len(top_chunks)} chunks "
            f"(top score: {scored[0][0]:.4f})"
        )
        return top_chunks


class PassThroughReranker:
    """
    No-op reranker for testing or when reranking is disabled.

    Returns the top-N chunks without any reranking.
    """

    def __init__(self, top_n: int = 5):
        self.top_n = top_n

    def rerank(self, query: str, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Return top-N chunks without reranking."""
        return chunks[: self.top_n]


def create_reranker(
    enabled: bool = True,
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: int = 5,
) -> CrossEncoderReranker | PassThroughReranker:
    """
    Factory function for creating rerankers.

    Args:
        enabled: Whether to use cross-encoder reranking.
        model: Cross-encoder model name.
        top_n: Number of chunks to keep.

    Returns:
        CrossEncoderReranker or PassThroughReranker.
    """
    if enabled:
        return CrossEncoderReranker(model_name=model, top_n=top_n)
    return PassThroughReranker(top_n=top_n)
