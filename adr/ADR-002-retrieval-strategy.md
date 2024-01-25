# ADR-002: Retrieval Strategy

**Status:** Accepted  
**Date:** 2024-01-18  
**Author:** Engineering Team

---

## Context

Retrieval quality is the single most important factor in RAG system performance. Poor retrieval means the LLM never has the right information regardless of its capability. We needed to choose a retrieval approach that balances precision, recall, and latency.

Key challenges identified during system design:
1. **Vocabulary mismatch:** User queries often use different terms than source documents
2. **Semantic vs. lexical:** Neural embeddings miss exact keyword matches; BM25 misses paraphrases
3. **Context redundancy:** Top-K retrieval often returns near-duplicate chunks
4. **Query ambiguity:** Short queries don't capture full user intent

## Decision

**Hybrid Search (Dense + BM25) with MMR diversification and optional HyDE enhancement**

The system implements a layered retrieval strategy:

```
Query 
  ├─→ [Dense Vector Search]  ─┐
  ├─→ [BM25 Keyword Search]   ├─→ [RRF Fusion] ─→ [MMR Diversify] ─→ Top-K
  └─→ [HyDE] (optional)       ┘
```

Default configuration (`config/settings.yaml`):
- Strategy: `hybrid`
- BM25 weight: 0.3
- Dense weight: 0.7  
- MMR lambda: 0.5
- Initial candidates: 20
- Final top_k: 5

## Alternatives Considered

| Strategy | Precision | Recall | Latency | Complexity |
|----------|-----------|--------|---------|------------|
| **Dense only** | High semantic | Low lexical | Fast | Low |
| **BM25 only** | High lexical | Low semantic | Very fast | Low |
| **Hybrid (selected)** | High both | High | Moderate | Medium |
| **HyDE** | Very high | High | Slow (extra LLM call) | High |
| **Multi-query** | High | Very high | Slow | High |

## Why Hybrid + RRF?

Reciprocal Rank Fusion was chosen over linear score combination because:
- RRF is invariant to score scales (BM25 scores are not normalized to [0,1])
- RRF is robust to outliers in individual rankers
- RRF is parameter-free beyond the constant `k=60`
- Strong empirical performance across multiple TREC tasks

## Why MMR?

Without MMR, top-5 results frequently contain 3-4 chunks from the same document section. MMR ensures the LLM receives a diverse set of perspectives, reducing hallucination risk when source material contains conflicting information.

## Why HyDE is Optional?

HyDE adds ~300-500ms latency per query (one extra LLM call). While it improves retrieval for complex technical queries, it degrades performance for factual lookup queries. It is enabled via config for use cases requiring high semantic precision.

## Consequences

**Positive:**
- 15-25% better recall than dense-only (from internal benchmarks against MS MARCO)
- Diverse context reduces answer quality degradation from redundant chunks
- Fully configurable via YAML without code changes

**Negative:**
- Higher latency than single-strategy (BM25 + MMR re-embedding adds ~50ms)
- BM25 index must be rebuilt when new documents are ingested
- More complex failure modes (both retrievers must work)

**Metrics Target:**
- Retrieval Precision@5 > 0.70
- Retrieval Recall@5 > 0.65
- End-to-end latency < 2s (without HyDE)
