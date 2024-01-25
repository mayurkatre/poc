# ADR-003: Chunking Strategy

**Status:** Accepted  
**Date:** 2024-01-20  
**Author:** Engineering Team

---

## Context

Document chunking determines what units of information are indexed and retrieved. Chunking strategy profoundly impacts both retrieval quality and answer quality. Chunks that are too large dilute relevance signals; chunks too small lose context.

Problem space:
- PDFs: Page-level chunks too large; sentence-level too small
- Markdown: Natural heading boundaries don't align with fixed character counts
- Technical docs: Code blocks should not be split mid-function
- Narrative text: Semantic paragraphs are more meaningful than arbitrary splits

## Decision

**Default: Semantic Chunking** (embedding-based boundary detection)  
**Fallback: Sentence Chunking**  
**Alternative: Fixed-size with overlap**

All strategies available via `config/settings.yaml: ingestion.chunking_strategy`.

### Semantic Chunking Algorithm

```
1. Split document into sentences
2. Embed consecutive sentence pairs
3. Compute cosine similarity between adjacent sentences  
4. Identify "semantic breaks" where similarity < threshold (0.75)
5. Group sentences between breaks into coherent chunks
6. Apply max-size constraint with sentence-boundary fallback
```

### Default Parameters
- `chunk_size: 512` characters
- `chunk_overlap: 64` characters  
- `similarity_threshold: 0.75`

## Alternatives Considered

| Strategy | Pros | Cons |
|----------|------|------|
| **Fixed-size** | Simple, predictable, fast | Cuts semantic units arbitrarily |
| **Sentence-based** | Preserves sentence integrity | Ignores topic boundaries |
| **Semantic (selected)** | Preserves topic coherence | Requires embedding model at ingest time |
| **Recursive text splitter** | Tries multiple separators | Still character-based |
| **Document-level** | No context loss | Too large for precision retrieval |
| **Proposition chunking** | Very high precision | Requires LLM at ingest (expensive) |

## Chunk Metadata

Every chunk carries:
```python
{
    "source": "rag_paper.pdf",          # Origin document
    "chunk_id": "sha256_hash[:16]",     # Stable unique identifier
    "chunk_index": 14,                   # Position in document
    "page_number": 3,                    # PDF page (if applicable)
    "chunking_strategy": "semantic",     # For debugging
    "sentence_count": 4,                 # Sentences in this chunk
    "char_count": 487                    # Character length
}
```

## Consequences

**Positive:**
- ~20% improvement in retrieval precision vs. fixed-size in our internal tests
- Chunks align with human-readable paragraphs
- Metadata enables precise source attribution (page, chunk ID)

**Negative:**
- Slower ingestion (requires embedding model during chunking)
- Non-deterministic chunk boundaries across model versions
- Cannot be used without sentence-transformers installed

**Mitigation:**
Semantic chunker gracefully falls back to sentence chunker if the embedding model is unavailable.
