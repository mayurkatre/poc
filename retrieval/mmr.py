"""
Maximum Marginal Relevance (MMR)
=================================
Reference: Carbonell & Goldstein 1998 - "The use of MMR, diversity-based reranking
for reordering documents and producing summaries"

Key Idea:
  Balances relevance to the query with diversity among selected documents.
  Avoids returning 5 near-duplicate chunks on the same sub-topic.

  Score(d) = λ * sim(d, query) - (1 - λ) * max_sim(d, selected)
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from ingestion.chunking import DocumentChunk


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def mmr_rerank(
    query_embedding: list[float],
    chunks: list[DocumentChunk],
    chunk_embeddings: list[list[float]],
    top_k: int = 5,
    lambda_param: float = 0.5,
) -> list[DocumentChunk]:
    """
    Apply Maximum Marginal Relevance to select diverse, relevant chunks.

    Args:
        query_embedding: Embedding of the query.
        chunks: Candidate chunks to select from.
        chunk_embeddings: Pre-computed embeddings for each chunk.
        top_k: Number of chunks to select.
        lambda_param: Balance between relevance (1.0) and diversity (0.0).
                      Typical values: 0.5 (balanced), 0.7 (relevance-favoring).

    Returns:
        Selected chunks in MMR-ranked order.
    """
    if not chunks:
        return []

    top_k = min(top_k, len(chunks))
    remaining_indices = list(range(len(chunks)))
    selected_indices: list[int] = []
    selected_embeddings: list[list[float]] = []

    # Pre-compute query similarities
    query_sims = [
        cosine_similarity(query_embedding, emb) for emb in chunk_embeddings
    ]

    for _ in range(top_k):
        best_idx = -1
        best_score = float("-inf")

        for idx in remaining_indices:
            relevance = query_sims[idx]

            # Redundancy: max similarity to already-selected docs
            if selected_embeddings:
                redundancy = max(
                    cosine_similarity(chunk_embeddings[idx], sel_emb)
                    for sel_emb in selected_embeddings
                )
            else:
                redundancy = 0.0

            score = lambda_param * relevance - (1 - lambda_param) * redundancy

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx == -1:
            break

        selected_indices.append(best_idx)
        selected_embeddings.append(chunk_embeddings[best_idx])
        remaining_indices.remove(best_idx)

    result = [chunks[i] for i in selected_indices]
    logger.debug(
        f"MMR: selected {len(result)}/{len(chunks)} chunks "
        f"(λ={lambda_param})"
    )
    return result
