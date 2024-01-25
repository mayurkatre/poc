# ADR-001: Vector Database Selection

**Status:** Accepted  
**Date:** 2024-01-15  
**Author:** Engineering Team

---

## Context

The RAG system requires a vector store for persisting and querying document embeddings. The choice of vector database significantly impacts retrieval latency, scalability, operational complexity, and cost. We needed to evaluate options suitable for a production-inspired POC that could scale to a real deployment.

Key requirements:
- Sub-100ms query latency for top-20 retrieval
- Support for L2 and inner product similarity
- Persistence across restarts
- Minimal operational overhead for initial deployment
- Clear migration path to a managed cloud solution

## Decision

**Primary:** FAISS (Facebook AI Similarity Search) with file-based persistence  
**Secondary/Alternative:** ChromaDB as a pluggable alternative

The system supports both via an abstract `BaseVectorStore` interface, selectable via `config/settings.yaml`.

## Alternatives Considered

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **FAISS** | Fastest query time (microseconds), battle-tested at Meta scale, CPU and GPU support | In-memory (requires loading), no built-in metadata filtering | ✅ Selected as default |
| **ChromaDB** | Developer-friendly API, built-in metadata storage and filtering, REST API option | Slightly higher latency, heavier dependencies | ✅ Supported as alternative |
| **Pinecone** | Fully managed, scales to billions of vectors, built-in metadata | Paid service, external dependency, adds network latency | ❌ Rejected for POC (cost) |
| **Weaviate** | Rich schema support, multimodal, GraphQL interface | Complex setup, overkill for POC | ❌ Rejected (complexity) |
| **pgvector** | Leverages existing PostgreSQL, ACID compliance | Slower vector search than purpose-built FAISS | ❌ Rejected (latency) |
| **Qdrant** | Rust-based, fast, rich filtering, on-premise or cloud | Less established ecosystem | ❌ Rejected (maturity risk) |

## Rationale

FAISS was chosen as the default because:

1. **Performance:** Uses HNSW or IVF indexes for approximate nearest neighbor search with microsecond latency
2. **Maturity:** Used in production at Meta, billions of vectors
3. **Portability:** Runs entirely on-disk, no server process needed
4. **Simplicity:** Single pip install, zero configuration
5. **Research alignment:** Most RAG research papers use FAISS as reference implementation

ChromaDB is supported as an alternative because:
- Provides richer metadata filtering capabilities
- Better developer experience for exploration
- Easier to add new document types with schema evolution

## Consequences

**Positive:**
- No network round-trips for retrieval (FAISS is in-process)
- Full control over index lifecycle and persistence
- Easy to swap backends via config without code changes

**Negative:**
- FAISS index must be loaded into memory at startup (RAM constraint for large corpora)
- No real-time updates without index rebuild
- FAISS lacks built-in metadata filtering (must filter post-retrieval)

**Future Migration Path:**
For production at scale (>10M documents), migrate to Pinecone or Qdrant by implementing a new `BaseVectorStore` subclass without changing retrieval or generation logic.
