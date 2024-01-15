"""Retrieval package: dense, BM25, hybrid, and HyDE retrievers."""

from retrieval.base_retriever import BaseRetriever, RetrievalResult
from retrieval.hybrid_search import BM25Index, HybridRetriever, reciprocal_rank_fusion
from retrieval.hyde_retriever import HyDERetriever
from retrieval.mmr import mmr_rerank

__all__ = [
    "BaseRetriever",
    "RetrievalResult",
    "HybridRetriever",
    "HyDERetriever",
    "BM25Index",
    "mmr_rerank",
    "reciprocal_rank_fusion",
]
