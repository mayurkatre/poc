"""
Base Retriever Interface
========================
Defines the common contract for all retrieval strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ingestion.chunking import DocumentChunk
from ingestion.embedding_pipeline import BaseVectorStore, EmbeddingPipeline


@dataclass
class RetrievalResult:
    """Container for retrieval results with scoring metadata."""

    chunks: list[DocumentChunk]
    query: str
    strategy: str
    scores: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def top_chunk(self) -> Optional[DocumentChunk]:
        return self.chunks[0] if self.chunks else None


class BaseRetriever(ABC):
    """Abstract base class for all retrieval strategies."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_pipeline: EmbeddingPipeline,
        top_k: int = 20,
    ):
        """
        Args:
            vector_store: Indexed vector store.
            embedding_pipeline: For query embedding.
            top_k: Number of candidates to retrieve initially.
        """
        self.vector_store = vector_store
        self.embedding_pipeline = embedding_pipeline
        self.top_k = top_k

    @abstractmethod
    def retrieve(self, query: str) -> RetrievalResult:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query: User query string.

        Returns:
            RetrievalResult containing ranked chunks.
        """
        ...

    def _embed_query(self, query: str) -> list[float]:
        return self.embedding_pipeline.embed_query(query)
