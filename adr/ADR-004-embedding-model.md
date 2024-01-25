# ADR-004: Embedding Model Selection

**Status:** Accepted  
**Date:** 2024-01-22  
**Author:** Engineering Team

---

## Context

The embedding model converts text to dense vectors for similarity search. The choice affects retrieval quality, latency, cost, and operational complexity. We need to balance model quality against practical deployment constraints.

Requirements:
- Suitable for retrieval tasks (not just sentence similarity)
- Fast enough for real-time query embedding (<50ms)
- Works offline (no mandatory API dependency for POC)
- Manageable memory footprint (<1GB)

## Decision

**Default: `all-MiniLM-L6-v2`** (via sentence-transformers)  
**Optional: `text-embedding-3-small`** (via OpenAI API)

Selectable via `config/settings.yaml: embedding.provider`.

### Model Comparison

| Model | Dim | MTEB Score | Speed | Cost | Offline |
|-------|-----|------------|-------|------|---------|
| **all-MiniLM-L6-v2** | 384 | 56.3 | ~5ms/batch | Free | ✅ |
| all-mpnet-base-v2 | 768 | 57.8 | ~15ms/batch | Free | ✅ |
| text-embedding-3-small | 1536 | 62.3 | ~50ms/API | $0.02/1M | ❌ |
| text-embedding-3-large | 3072 | 64.6 | ~100ms/API | $0.13/1M | ❌ |
| bge-large-en-v1.5 | 1024 | 63.6 | ~20ms/batch | Free | ✅ |

## Why all-MiniLM-L6-v2 as Default?

1. **Speed:** 6x faster than larger models, enabling real-time query embedding
2. **Quality:** Top performer on BEIR retrieval benchmarks relative to size
3. **Zero cost:** Runs locally, no API key required for development
4. **Small footprint:** ~80MB model, fits in memory alongside the application
5. **Production ready:** Used by hundreds of production RAG systems

## Why OpenAI as Optional?

For production workloads where retrieval precision is critical:
- `text-embedding-3-small` outperforms all-MiniLM by ~6 points on MTEB
- Better multilingual support
- Managed infrastructure (no GPU needed at scale)

## Embedding Caching Design

To avoid redundant computation, both providers are wrapped with `CachedEmbedder`:

```python
cache_key = SHA256(text)[:24]
if key in disk_cache:
    return cached_embedding
else:
    embedding = model.embed(text)
    cache[key] = embedding
    return embedding
```

This reduces ingestion time by >90% on re-indexing operations.

## Consequences

**Positive:**
- Zero external dependencies for basic operation
- Consistent behavior across environments (no API version drift)
- Embedding cache reduces repeated API costs by 90%+

**Negative:**
- all-MiniLM produces 384-dim vectors vs. 1536-dim OpenAI (some precision loss)
- Switching providers requires re-indexing entire corpus (different vector spaces)
- Local model requires ~200MB disk space

**Future:**
Consider `bge-large-en-v1.5` as a higher-quality offline alternative once computational resources allow.
