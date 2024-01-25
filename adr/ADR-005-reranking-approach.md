# ADR-005: Reranking Approach

**Status:** Accepted  
**Date:** 2024-01-25  
**Author:** Engineering Team

---

## Context

Initial retrieval (bi-encoder) is fast but imprecise. It operates at scale (top-20 candidates) and uses separate embeddings for query and document. A reranking stage refines this initial set using a more expensive but more accurate model.

The key insight: it's acceptable to spend more compute on a small candidate set (20 docs) to get precision gains that significantly improve the final answer quality.

## Decision

**Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`**

Applied as: Retrieve Top-20 → Rerank → Select Top-5

Pipeline:
```
Query + [Doc_1, ..., Doc_20]
         ↓
  CrossEncoder.predict(
    [(query, doc_1), ..., (query, doc_20)]
  )
         ↓
  Sort by relevance score
         ↓
  Return Top-5
```

## Alternatives Considered

| Approach | Quality | Latency | Cost |
|----------|---------|---------|------|
| **Cross-encoder (selected)** | ★★★★★ | ~200ms | Free |
| No reranking | ★★★ | 0ms | Free |
| LLM-based reranking | ★★★★★ | ~800ms | API cost |
| ColBERT | ★★★★ | ~100ms | Free (heavier model) |
| Cohere Rerank API | ★★★★★ | ~300ms | $1/1000 calls |
| RRF alone | ★★★★ | ~10ms | Free |

## Why Cross-Encoder?

**Theoretical justification:**
Bi-encoders (used for retrieval) compress both query and document into fixed-size vectors independently, losing fine-grained token interactions. Cross-encoders process query and document together with full self-attention, allowing the model to understand:
- Negation ("NOT a database" vs. "a database")
- Specificity (query asks about version 2.0, document mentions version 1.0)
- Contextual relevance (same keywords, different meaning)

**Empirical justification:**
On MS MARCO, cross-encoders achieve MRR@10 of ~39 vs. ~33 for bi-encoders — a 18% relative improvement in ranking precision.

**Selected model justification:**
`ms-marco-MiniLM-L-6-v2` was fine-tuned specifically on MS MARCO passage ranking, which is the most representative benchmark for RAG-style retrieval. MiniLM-L-6 is the smallest variant that retains most of the quality of larger models.

## Latency Analysis

| Stage | Latency |
|-------|---------|
| Query embedding | ~5ms |
| Dense retrieval (top-20) | ~10ms |
| BM25 retrieval (top-20) | ~2ms |
| RRF fusion | ~1ms |
| Cross-encoder reranking (20 pairs) | ~180ms |
| LLM generation | ~800ms |
| **Total** | **~1000ms** |

Cross-encoder adds ~18% to total latency but provides significant quality gains.

## Consequences

**Positive:**
- Measurable improvement in answer faithfulness (cross-encoder catches false positives that embedding similarity misses)
- Eliminates false positives caused by keyword overlap without semantic match
- Model runs locally (no API dependency)

**Negative:**
- ~180ms additional latency per query
- Must load ~120MB model into memory
- Batch processing required (individual predictions are slow)

**Disabling:**
Set `reranking.enabled: false` in config to use `PassThroughReranker` for latency-critical deployments or debugging.
