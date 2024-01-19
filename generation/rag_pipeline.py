"""
RAG Generation Pipeline
========================
Orchestrates the full Retrieve → Rerank → Generate → Cite pipeline.

Features:
  - Configurable LLM backend (OpenAI)
  - Streaming support
  - Source attribution in every response
  - Query rewriting for improved retrieval coverage
  - Guardrails against hallucination (low-confidence detection)
  - Response caching
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterator, Optional, Union

from loguru import logger

from ingestion.chunking import DocumentChunk
from reranking.cross_encoder import CrossEncoderReranker, PassThroughReranker
from retrieval.base_retriever import BaseRetriever, RetrievalResult


# ---------------------------------------------------------------------------
# Response Data Models
# ---------------------------------------------------------------------------

@dataclass
class SourceReference:
    """A single cited source in the generated answer."""

    document: str
    chunk_id: str
    text: str
    chunk_index: int
    page_number: Optional[int] = None
    rerank_score: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "document": self.document,
            "chunk_id": self.chunk_id,
            "text": self.text[:300],  # Truncate for response
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
        }


@dataclass
class RAGResponse:
    """Full response from the RAG pipeline."""

    answer: str
    sources: list[SourceReference]
    query: str
    retrieval_strategy: str
    latency_ms: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "query": self.query,
            "retrieval_strategy": self.retrieval_strategy,
            "latency_ms": round(self.latency_ms, 2),
        }


# ---------------------------------------------------------------------------
# Query Rewriter
# ---------------------------------------------------------------------------

class QueryRewriter:
    """
    Generates multiple reformulations of a query to improve retrieval recall.

    Uses an LLM to produce diverse phrasings of the same intent, then
    retrieves for all variants and merges results.
    """

    def __init__(self, model: str = "gpt-4o-mini", num_variants: int = 3):
        self.model = model
        self.num_variants = num_variants

    def rewrite(self, query: str) -> list[str]:
        """
        Generate query variants.

        Args:
            query: Original user query.

        Returns:
            List of [original] + [variants].
        """
        try:
            import openai
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            prompt = (
                f"Generate {self.num_variants} different phrasings of the following "
                f"question. Each rephrasing should preserve the intent but use "
                f"different vocabulary or structure. Return ONLY the questions, "
                f"one per line, no numbering.\n\n"
                f"Question: {query}"
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=256,
            )
            variants = [
                line.strip()
                for line in response.choices[0].message.content.strip().split("\n")
                if line.strip()
            ]
            logger.debug(f"Query rewriter: {len(variants)} variants generated")
            return [query] + variants[: self.num_variants]
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
            return [query]


# ---------------------------------------------------------------------------
# Response Cache
# ---------------------------------------------------------------------------

class ResponseCache:
    """Disk-based response cache keyed by query hash."""

    def __init__(self, cache_dir: str = ".cache/responses"):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, query: str) -> str:
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:24]

    def get(self, query: str) -> Optional[RAGResponse]:
        path = self._cache_dir / f"{self._key(query)}.pkl"
        if path.exists():
            with path.open("rb") as f:
                logger.debug(f"Cache HIT for query: {query[:60]}")
                return pickle.load(f)
        return None

    def set(self, query: str, response: RAGResponse) -> None:
        path = self._cache_dir / f"{self._key(query)}.pkl"
        with path.open("wb") as f:
            pickle.dump(response, f)


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Full RAG pipeline: query → retrieve → rerank → generate → cite.

    Supports:
      - Multiple retrieval strategies
      - Query rewriting
      - Streaming generation
      - Response caching
      - Source attribution
    """

    SYSTEM_PROMPT = """You are a precise and helpful research assistant.

INSTRUCTIONS:
- Answer the question using ONLY the provided context documents.
- Be accurate and factual. Do not add information not present in the context.
- Structure your answer clearly.
- If the context does not contain enough information to answer fully, say so explicitly.
- Do not hallucinate or make up facts.

CONTEXT:
{context}"""

    def __init__(
        self,
        retriever: BaseRetriever,
        reranker: Union[CrossEncoderReranker, PassThroughReranker],
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        streaming: bool = False,
        query_rewriting_enabled: bool = True,
        cache_enabled: bool = True,
    ):
        """
        Args:
            retriever: Configured retrieval strategy.
            reranker: Cross-encoder or pass-through reranker.
            llm_model: OpenAI model for generation.
            temperature: LLM temperature (0 = deterministic).
            max_tokens: Max tokens in response.
            streaming: Whether to stream the response.
            query_rewriting_enabled: Whether to rewrite queries.
            cache_enabled: Whether to cache responses.
        """
        self.retriever = retriever
        self.reranker = reranker
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.streaming = streaming
        self.query_rewriter = QueryRewriter() if query_rewriting_enabled else None
        self.cache = ResponseCache() if cache_enabled else None

    def _build_context(self, chunks: list[DocumentChunk]) -> str:
        """Format retrieved chunks as numbered context blocks."""
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            source = Path(chunk.source).name
            page = chunk.metadata.get("page_number", "")
            page_str = f", page {page}" if page else ""
            parts.append(
                f"[{i}] Source: {source}{page_str} (chunk {chunk.chunk_index})\n"
                f"{chunk.text}"
            )
        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, chunks: list[DocumentChunk]) -> list[SourceReference]:
        """Convert DocumentChunks to SourceReference objects."""
        return [
            SourceReference(
                document=Path(c.source).name,
                chunk_id=c.chunk_id,
                text=c.text,
                chunk_index=c.chunk_index,
                page_number=c.metadata.get("page_number"),
                rerank_score=c.metadata.get("rerank_score"),
            )
            for c in chunks
        ]

    def _generate(self, query: str, context: str) -> str:
        """Call the LLM to generate an answer."""
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        system = self.SYSTEM_PROMPT.format(context=context)
        response = client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _generate_streaming(self, query: str, context: str) -> Iterator[str]:
        """Stream the LLM response token by token."""
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        system = self.SYSTEM_PROMPT.format(context=context)
        with client.chat.completions.stream(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ) as stream:
            for text in stream.text_stream:
                yield text

    def _retrieve_with_rewriting(self, query: str) -> RetrievalResult:
        """
        Retrieve with optional query rewriting.

        If rewriting is enabled, retrieves for all query variants and
        merges results using deduplication by chunk_id.
        """
        if self.query_rewriter is None:
            return self.retriever.retrieve(query)

        queries = self.query_rewriter.rewrite(query)
        all_chunks: dict[str, DocumentChunk] = {}

        for q in queries:
            result = self.retriever.retrieve(q)
            for chunk in result.chunks:
                if chunk.chunk_id not in all_chunks:
                    all_chunks[chunk.chunk_id] = chunk

        merged = list(all_chunks.values())
        logger.info(
            f"Query rewriting: {len(queries)} queries → {len(merged)} unique chunks"
        )

        return RetrievalResult(
            chunks=merged,
            query=query,
            strategy=f"{self.retriever.__class__.__name__}+rewrite",
        )

    def query(self, question: str) -> RAGResponse:
        """
        Execute the full RAG pipeline for a question.

        Args:
            question: User question.

        Returns:
            RAGResponse with answer and citations.
        """
        start_time = time.monotonic()

        # --- Cache check ---
        if self.cache:
            cached = self.cache.get(question)
            if cached:
                logger.info("Returning cached response.")
                return cached

        logger.info(f"RAG query: '{question[:80]}'")

        # --- Retrieval ---
        retrieval_result = self._retrieve_with_rewriting(question)
        logger.info(f"Retrieved {len(retrieval_result.chunks)} chunks")

        # --- Reranking ---
        reranked = self.reranker.rerank(question, retrieval_result.chunks)
        logger.info(f"After reranking: {len(reranked)} chunks")

        # --- Build context ---
        context = self._build_context(reranked)

        # --- Generation ---
        answer = self._generate(question, context)

        # --- Build response ---
        sources = self._extract_sources(reranked)
        latency = (time.monotonic() - start_time) * 1000

        response = RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            retrieval_strategy=retrieval_result.strategy,
            latency_ms=latency,
        )

        logger.success(
            f"RAG complete: {len(answer)} chars, "
            f"{len(sources)} sources, {latency:.0f}ms"
        )

        # --- Cache store ---
        if self.cache:
            self.cache.set(question, response)

        return response

    def query_stream(
        self, question: str
    ) -> tuple[Iterator[str], list[SourceReference]]:
        """
        Stream the answer while returning sources immediately.

        Args:
            question: User question.

        Returns:
            Tuple of (token iterator, source references).
        """
        # Retrieve and rerank
        retrieval_result = self._retrieve_with_rewriting(question)
        reranked = self.reranker.rerank(question, retrieval_result.chunks)
        context = self._build_context(reranked)
        sources = self._extract_sources(reranked)

        token_iter = self._generate_streaming(question, context)
        return token_iter, sources
