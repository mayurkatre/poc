"""
FastAPI Application
====================
REST API interface for the RAG system.

Endpoints:
  POST /query          - Standard RAG query
  POST /query/stream   - Streaming RAG query (SSE)
  GET  /health         - Health check
  GET  /stats          - Index statistics
  POST /ingest         - Ingest a document at runtime
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

import config
from ingestion.chunking import ChunkerFactory
from ingestion.document_loader import DocumentLoaderFactory
from config.openrouter import get_client
from ingestion.embedding_pipeline import (
    EmbeddingPipeline,
    create_embedder,
    create_vector_store,
)
from reranking.cross_encoder import create_reranker
from retrieval.hybrid_search import BM25Index, HybridRetriever
from generation.rag_pipeline import RAGPipeline, RAGResponse


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Incoming query request."""

    question: str = Field(..., min_length=3, max_length=1000, description="User question")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Override final top_k")
    strategy: Optional[str] = Field(None, description="Override retrieval strategy")


class SourceModel(BaseModel):
    """A single cited source."""

    document: str
    chunk_id: str
    text: str
    chunk_index: int
    page_number: Optional[int] = None


class QueryResponse(BaseModel):
    """Standard query response."""

    answer: str
    sources: list[SourceModel]
    query: str
    retrieval_strategy: str
    latency_ms: float


class IngestRequest(BaseModel):
    """Request to ingest a document."""

    source: str = Field(..., description="File path or URL to ingest")


class HealthResponse(BaseModel):
    status: str
    version: str
    indexed_chunks: int


class StatsResponse(BaseModel):
    indexed_chunks: int
    embedding_provider: str
    vector_store_provider: str
    retrieval_strategy: str


# ---------------------------------------------------------------------------
# Application State
# ---------------------------------------------------------------------------

class AppState:
    """Holds shared pipeline components initialized at startup."""

    pipeline: Optional[RAGPipeline] = None
    vector_store = None
    embedding_pipeline: Optional[EmbeddingPipeline] = None
    bm25_index: Optional[BM25Index] = None
    settings: dict = {}


app_state = AppState()


def _build_pipeline() -> tuple[RAGPipeline, EmbeddingPipeline, any, BM25Index]:
    """Initialize all pipeline components from config."""
    cfg = config.get_settings()

    # Embedder
    emb_cfg = cfg.get("embedding", {})
    embedder = create_embedder(
        provider=emb_cfg.get("provider", "sentence_transformers"),
        model_name=emb_cfg.get("model_name", "all-MiniLM-L6-v2"),
        cache_enabled=emb_cfg.get("cache_enabled", True),
        cache_dir=emb_cfg.get("cache_dir", ".cache/embeddings"),
    )

    # Vector Store
    vs_cfg = cfg.get("vector_store", {})
    vector_store = create_vector_store(
        provider=vs_cfg.get("provider", "faiss"),
        index_path=vs_cfg.get("index_path", ".cache/faiss_index"),
        chroma_persist_dir=vs_cfg.get("chroma_persist_dir", ".cache/chroma"),
        collection_name=vs_cfg.get("collection_name", "rag_documents"),
        dimension=embedder.dimension,
    )

    # Try loading existing index
    try:
        vector_store.load()
        logger.info(f"Loaded existing vector store: {len(vector_store)} chunks")
    except FileNotFoundError:
        logger.warning("No existing index found. Run `python ingest.py` first.")

    embedding_pipeline = EmbeddingPipeline(embedder, vector_store)

    # BM25 Index (populated from vector store chunks if available)
    bm25_index = BM25Index()
    if hasattr(vector_store, "_chunks") and vector_store._chunks:
        bm25_index.build(vector_store._chunks)

    # Retriever
    ret_cfg = cfg.get("retrieval", {})
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_pipeline=embedding_pipeline,
        bm25_index=bm25_index,
        top_k=vs_cfg.get("top_k", 20),
        final_top_k=ret_cfg.get("final_top_k", 5),
        bm25_weight=ret_cfg.get("bm25_weight", 0.3),
        dense_weight=ret_cfg.get("dense_weight", 0.7),
        use_mmr=ret_cfg.get("mmr_enabled", True),
        mmr_lambda=ret_cfg.get("mmr_lambda", 0.5),
    )

    # Reranker
    rr_cfg = cfg.get("reranking", {})
    reranker = create_reranker(
        enabled=rr_cfg.get("enabled", True),
        model=rr_cfg.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        top_n=rr_cfg.get("top_n", 5),
    )

    # Generation
    gen_cfg = cfg.get("generation", {})
    qr_cfg = cfg.get("query_rewriting", {})
    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        llm_model=gen_cfg.get("model", "openai/gpt-4o-mini"),
        temperature=gen_cfg.get("temperature", 0.0),
        max_tokens=gen_cfg.get("max_tokens", 1024),
        streaming=gen_cfg.get("streaming", True),
        query_rewriting_enabled=qr_cfg.get("enabled", True),
        cache_enabled=True,
    )

    return pipeline, embedding_pipeline, vector_store, bm25_index


# ---------------------------------------------------------------------------
# Application Lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup."""
    logger.info("Starting RAG API server...")
    try:
        (
            app_state.pipeline,
            app_state.embedding_pipeline,
            app_state.vector_store,
            app_state.bm25_index,
        ) = _build_pipeline()
        app_state.settings = config.get_settings()
        logger.success("RAG pipeline ready.")
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {e}")

    yield

    logger.info("Shutting down RAG API server.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG System API",
    description="Production-inspired Retrieval-Augmented Generation system",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
cfg = config.get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.get("api", {}).get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    indexed = (
        len(app_state.vector_store)
        if app_state.vector_store
        else 0
    )
    return HealthResponse(
        status="healthy" if app_state.pipeline else "degraded",
        version="1.0.0",
        indexed_chunks=indexed,
    )


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """Return pipeline statistics."""
    if not app_state.pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    cfg = config.get_settings()
    return StatsResponse(
        indexed_chunks=len(app_state.vector_store) if app_state.vector_store else 0,
        embedding_provider=cfg.get("embedding", {}).get("provider", "unknown"),
        vector_store_provider=cfg.get("vector_store", {}).get("provider", "unknown"),
        retrieval_strategy=cfg.get("retrieval", {}).get("strategy", "hybrid"),
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest):
    """
    Query the RAG system.

    Retrieves relevant documents and generates an answer with citations.
    """
    if not app_state.pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized. Run ingest first.")

    try:
        response: RAGResponse = app_state.pipeline.query(request.question)
        return QueryResponse(
            answer=response.answer,
            sources=[SourceModel(**s.to_dict()) for s in response.sources],
            query=response.query,
            retrieval_strategy=response.retrieval_strategy,
            latency_ms=response.latency_ms,
        )
    except Exception as e:
        logger.exception(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream", tags=["RAG"])
async def query_stream(request: QueryRequest):
    """
    Stream a RAG response using Server-Sent Events.

    First event contains sources, subsequent events are answer tokens.
    """
    if not app_state.pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    import json as _json

    async def event_stream():
        try:
            token_iter, sources = app_state.pipeline.query_stream(request.question)

            # Send sources first
            sources_data = _json.dumps(
                {"type": "sources", "sources": [s.to_dict() for s in sources]}
            )
            yield f"data: {sources_data}\n\n"

            # Stream answer tokens
            for token in token_iter:
                token_data = _json.dumps({"type": "token", "content": token})
                yield f"data: {token_data}\n\n"

            yield 'data: {"type": "done"}\n\n'
        except Exception as e:
            error_data = _json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/ingest", tags=["Ingestion"])
async def ingest_document(
    request: IngestRequest, background_tasks: BackgroundTasks
):
    """
    Ingest a new document into the RAG system.

    Runs in the background so the API remains responsive.
    """
    if not app_state.embedding_pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    def _run_ingest(source: str):
        try:
            cfg = config.get_settings()
            ing_cfg = cfg.get("ingestion", {})

            docs = DocumentLoaderFactory.load(source)
            chunker = ChunkerFactory.create(
                strategy=ing_cfg.get("chunking_strategy", "semantic"),
                chunk_size=ing_cfg.get("chunk_size", 512),
                overlap=ing_cfg.get("chunk_overlap", 64),
            )
            all_chunks = []
            for doc in docs:
                all_chunks.extend(chunker.chunk(doc))

            app_state.embedding_pipeline.index(all_chunks)

            # Rebuild BM25
            if hasattr(app_state.vector_store, "_chunks"):
                app_state.bm25_index.build(app_state.vector_store._chunks)

            logger.success(f"Ingested {source}: {len(all_chunks)} chunks")
        except Exception as e:
            logger.error(f"Ingest failed for {source}: {e}")

    background_tasks.add_task(_run_ingest, request.source)
    return {"status": "ingestion_started", "source": request.source}
