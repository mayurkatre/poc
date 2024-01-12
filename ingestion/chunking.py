"""
Document Chunking Module
========================
Implements multiple chunking strategies:
  - Fixed-size with overlap
  - Sentence-boundary aware
  - Semantic chunking (embedding-based coherence)

Each strategy preserves rich metadata for source attribution.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from ingestion.document_loader import RawDocument


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """A single chunk of a document with full metadata."""

    chunk_id: str
    text: str
    source: str
    doc_type: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        text: str,
        source: str,
        doc_type: str,
        chunk_index: int,
        extra_metadata: dict | None = None,
    ) -> "DocumentChunk":
        """
        Factory method that auto-generates a stable chunk ID.

        Args:
            text: Chunk text content.
            source: Origin document path or URL.
            doc_type: Type of source document.
            chunk_index: Position of this chunk within the document.
            extra_metadata: Additional metadata from the parent document.

        Returns:
            DocumentChunk instance.
        """
        chunk_id = hashlib.sha256(
            f"{source}:{chunk_index}:{text[:64]}".encode()
        ).hexdigest()[:16]

        metadata = {
            "source": source,
            "chunk_index": chunk_index,
            "char_count": len(text),
            "word_count": len(text.split()),
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return cls(
            chunk_id=chunk_id,
            text=text,
            source=source,
            doc_type=doc_type,
            chunk_index=chunk_index,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Base Chunker
# ---------------------------------------------------------------------------

class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, document: RawDocument) -> list[DocumentChunk]:
        """
        Split a RawDocument into DocumentChunks.

        Args:
            document: Loaded document to chunk.

        Returns:
            Ordered list of chunks.
        """
        ...


# ---------------------------------------------------------------------------
# Fixed-Size Chunker
# ---------------------------------------------------------------------------

class FixedSizeChunker(BaseChunker):
    """
    Splits text into fixed-size windows with configurable overlap.

    Uses character counts. Overlap helps preserve context across boundaries.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        """
        Args:
            chunk_size: Target character length per chunk.
            overlap: Number of characters to repeat between adjacent chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: RawDocument) -> list[DocumentChunk]:
        """Split document using fixed-size sliding window."""
        text = document.content
        chunks: list[DocumentChunk] = []
        start = 0
        index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if len(chunk_text) >= 20:  # Skip trivially small chunks
                chunks.append(
                    DocumentChunk.create(
                        text=chunk_text,
                        source=document.source,
                        doc_type=document.doc_type,
                        chunk_index=index,
                        extra_metadata={
                            **document.metadata,
                            "chunking_strategy": "fixed",
                            "start_char": start,
                            "end_char": end,
                        },
                    )
                )
                index += 1

            start += self.chunk_size - self.overlap

        logger.debug(f"Fixed chunker: {len(chunks)} chunks from {document.source}")
        return chunks


# ---------------------------------------------------------------------------
# Sentence-Boundary Chunker
# ---------------------------------------------------------------------------

class SentenceChunker(BaseChunker):
    """
    Splits text at sentence boundaries, grouping sentences to fill chunks.

    Avoids cutting sentences mid-way, which helps preserve semantic meaning.
    """

    def __init__(self, chunk_size: int = 512, overlap_sentences: int = 1):
        """
        Args:
            chunk_size: Max character length per chunk.
            overlap_sentences: Number of sentences to repeat for context.
        """
        self.chunk_size = chunk_size
        self.overlap_sentences = overlap_sentences
        self._sent_pattern = re.compile(r"(?<=[.!?])\s+")

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into individual sentences."""
        return [s.strip() for s in self._sent_pattern.split(text) if s.strip()]

    def chunk(self, document: RawDocument) -> list[DocumentChunk]:
        """Group sentences into chunks, respecting size limits."""
        sentences = self._split_sentences(document.content)
        chunks: list[DocumentChunk] = []
        current: list[str] = []
        current_len = 0
        index = 0

        for sent in sentences:
            if current_len + len(sent) > self.chunk_size and current:
                chunk_text = " ".join(current)
                chunks.append(
                    DocumentChunk.create(
                        text=chunk_text,
                        source=document.source,
                        doc_type=document.doc_type,
                        chunk_index=index,
                        extra_metadata={
                            **document.metadata,
                            "chunking_strategy": "sentence",
                            "sentence_count": len(current),
                        },
                    )
                )
                index += 1
                # Overlap: keep last N sentences
                current = current[-self.overlap_sentences :] if self.overlap_sentences else []
                current_len = sum(len(s) for s in current)

            current.append(sent)
            current_len += len(sent)

        # Flush remaining
        if current:
            chunk_text = " ".join(current)
            chunks.append(
                DocumentChunk.create(
                    text=chunk_text,
                    source=document.source,
                    doc_type=document.doc_type,
                    chunk_index=index,
                    extra_metadata={
                        **document.metadata,
                        "chunking_strategy": "sentence",
                        "sentence_count": len(current),
                    },
                )
            )

        logger.debug(f"Sentence chunker: {len(chunks)} chunks from {document.source}")
        return chunks


# ---------------------------------------------------------------------------
# Semantic Chunker
# ---------------------------------------------------------------------------

class SemanticChunker(BaseChunker):
    """
    Embedding-based semantic chunking.

    Groups sentences together as long as cosine similarity between consecutive
    sentence embeddings stays above a threshold. When similarity drops
    (topic shift), a new chunk begins.

    This produces chunks with high internal coherence.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        similarity_threshold: float = 0.75,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Args:
            chunk_size: Soft max characters per chunk.
            overlap: Overlap characters on chunk boundary.
            similarity_threshold: Minimum cosine similarity to merge sentences.
            embedding_model: Sentence-transformers model name.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self._model = None
        self._sent_chunker = SentenceChunker(chunk_size=chunk_size, overlap_sentences=1)

    def _get_model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.embedding_model)
                logger.debug(f"Loaded semantic chunking model: {self.embedding_model}")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Falling back to sentence chunker."
                )
                return None
        return self._model

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np
        a, b = np.array(a), np.array(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    def chunk(self, document: RawDocument) -> list[DocumentChunk]:
        """
        Chunk document by detecting semantic breaks in sentence embeddings.
        Falls back to SentenceChunker if model is unavailable.
        """
        model = self._get_model()
        if model is None:
            logger.warning("Using sentence chunker fallback for semantic chunking.")
            return self._sent_chunker.chunk(document)

        # Split into sentences
        sent_pattern = re.compile(r"(?<=[.!?])\s+")
        sentences = [s.strip() for s in sent_pattern.split(document.content) if s.strip()]

        if len(sentences) < 2:
            return self._sent_chunker.chunk(document)

        # Embed all sentences in batch
        embeddings = model.encode(sentences, batch_size=32, show_progress_bar=False)

        # Compute consecutive similarities
        breaks = [0]
        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < self.similarity_threshold:
                breaks.append(i)
        breaks.append(len(sentences))

        # Build chunks from semantic groups
        chunks: list[DocumentChunk] = []
        index = 0
        for start, end in zip(breaks, breaks[1:]):
            group = sentences[start:end]
            chunk_text = " ".join(group).strip()

            # Hard split if still too large
            if len(chunk_text) > self.chunk_size * 2:
                sub_doc = RawDocument(
                    content=chunk_text,
                    source=document.source,
                    doc_type=document.doc_type,
                    metadata=document.metadata,
                )
                sub_chunks = self._sent_chunker.chunk(sub_doc)
                for sc in sub_chunks:
                    sc.chunk_index = index
                    sc.metadata["chunking_strategy"] = "semantic"
                    chunks.append(sc)
                    index += 1
                continue

            if chunk_text:
                chunks.append(
                    DocumentChunk.create(
                        text=chunk_text,
                        source=document.source,
                        doc_type=document.doc_type,
                        chunk_index=index,
                        extra_metadata={
                            **document.metadata,
                            "chunking_strategy": "semantic",
                            "sentence_count": len(group),
                        },
                    )
                )
                index += 1

        logger.info(
            f"Semantic chunker: {len(chunks)} chunks from {document.source}"
        )
        return chunks


# ---------------------------------------------------------------------------
# Chunker Factory
# ---------------------------------------------------------------------------

class ChunkerFactory:
    """Creates the appropriate chunker based on strategy name."""

    _strategies: dict[str, type[BaseChunker]] = {
        "fixed": FixedSizeChunker,
        "sentence": SentenceChunker,
        "semantic": SemanticChunker,
    }

    @classmethod
    def create(
        cls,
        strategy: str = "semantic",
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> BaseChunker:
        """
        Instantiate a chunker by strategy name.

        Args:
            strategy: One of 'fixed', 'sentence', 'semantic'.
            chunk_size: Target chunk size in characters.
            overlap: Overlap in characters (or sentences for sentence strategy).

        Returns:
            Instantiated BaseChunker.
        """
        if strategy not in cls._strategies:
            raise ValueError(
                f"Unknown chunking strategy: '{strategy}'. "
                f"Choose from: {list(cls._strategies.keys())}"
            )

        chunker_cls = cls._strategies[strategy]
        if strategy == "semantic":
            return SemanticChunker(chunk_size=chunk_size, overlap=overlap)
        elif strategy == "sentence":
            return SentenceChunker(chunk_size=chunk_size)
        else:
            return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
