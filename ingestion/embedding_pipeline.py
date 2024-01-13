"""
Embedding Pipeline Module
=========================
Handles embedding generation and vector store management.

Supports:
  - OpenAI embeddings (text-embedding-3-small)
  - Sentence Transformers (all-MiniLM-L6-v2)
  - FAISS and Chroma vector stores
  - Disk-based embedding cache to avoid recomputation
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger

from ingestion.chunking import DocumentChunk


# ---------------------------------------------------------------------------
# Embedding Models
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings.

        Returns:
            List of embedding vectors (floats).
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimensionality."""
        ...


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder using sentence-transformers (runs locally)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64):
        """
        Args:
            model_name: HuggingFace model identifier.
            batch_size: Batch size for encoding.
        """
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"Loaded SentenceTransformer: {model_name} (dim={self._dim})")

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self._dim


class OpenAIEmbedder(BaseEmbedder):
    """Embedder using OpenAI's embedding API."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        batch_size: int = 512,
    ):
        """
        Args:
            model_name: OpenAI embedding model name.
            batch_size: Max texts per API call.
        """
        import openai
        self._client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = model_name
        self._batch_size = batch_size
        self._dim = 1536 if "small" in model_name else 3072
        logger.info(f"Initialized OpenAI embedder: {model_name}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = self._client.embeddings.create(
                input=batch, model=self._model
            )
            all_embeddings.extend([d.embedding for d in response.data])
        return all_embeddings

    @property
    def dimension(self) -> int:
        return self._dim


class CachedEmbedder(BaseEmbedder):
    """
    Wraps any embedder with a disk-based cache.

    Cache key is a SHA256 hash of the text content.
    Useful for avoiding repeated API calls or model inference.
    """

    def __init__(self, embedder: BaseEmbedder, cache_dir: str = ".cache/embeddings"):
        """
        Args:
            embedder: Underlying embedder to wrap.
            cache_dir: Directory to store cache files.
        """
        self._embedder = embedder
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = self._load_cache()
        logger.debug(
            f"Cache loaded: {len(self._cache)} embeddings from {cache_dir}"
        )

    def _cache_path(self) -> Path:
        return self._cache_dir / "embedding_cache.pkl"

    def _load_cache(self) -> dict[str, list[float]]:
        path = self._cache_path()
        if path.exists():
            with path.open("rb") as f:
                return pickle.load(f)
        return {}

    def _save_cache(self) -> None:
        with self._cache_path().open("wb") as f:
            pickle.dump(self._cache, f)

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = [None] * len(texts)  # type: ignore
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for i, text in enumerate(texts):
            key = self._hash(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                missing_indices.append(i)
                missing_texts.append(text)

        if missing_texts:
            logger.debug(f"Computing {len(missing_texts)} new embeddings (cache miss)")
            new_embeddings = self._embedder.embed(missing_texts)
            for idx, text, emb in zip(missing_indices, missing_texts, new_embeddings):
                key = self._hash(text)
                self._cache[key] = emb
                results[idx] = emb
            self._save_cache()

        return results

    @property
    def dimension(self) -> int:
        return self._embedder.dimension


def create_embedder(
    provider: str = "sentence_transformers",
    model_name: str = "all-MiniLM-L6-v2",
    cache_enabled: bool = True,
    cache_dir: str = ".cache/embeddings",
) -> BaseEmbedder:
    """
    Factory function for creating embedders.

    Args:
        provider: 'sentence_transformers' or 'openai'.
        model_name: Model identifier.
        cache_enabled: Whether to wrap with caching layer.
        cache_dir: Cache directory path.

    Returns:
        BaseEmbedder instance.
    """
    if provider == "openai":
        embedder: BaseEmbedder = OpenAIEmbedder(model_name=model_name)
    elif provider == "sentence_transformers":
        embedder = SentenceTransformerEmbedder(model_name=model_name)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

    if cache_enabled:
        embedder = CachedEmbedder(embedder, cache_dir=cache_dir)

    return embedder


# ---------------------------------------------------------------------------
# Vector Stores
# ---------------------------------------------------------------------------

class BaseVectorStore(ABC):
    """Abstract base for vector store backends."""

    @abstractmethod
    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Index chunks with their embeddings."""
        ...

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 20) -> list[DocumentChunk]:
        """Retrieve top-k most similar chunks."""
        ...

    @abstractmethod
    def save(self) -> None:
        """Persist the index to disk."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load the index from disk."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Number of indexed chunks."""
        ...


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-based vector store using Inner Product (cosine similarity
    when embeddings are L2-normalized).

    Serializes both the FAISS index and metadata as separate files.
    """

    def __init__(self, index_path: str = ".cache/faiss_index", dimension: int = 384):
        """
        Args:
            index_path: Directory for persisting the FAISS index.
            dimension: Embedding vector size.
        """
        self._index_path = Path(index_path)
        self._index_path.mkdir(parents=True, exist_ok=True)
        self._dimension = dimension
        self._index = None
        self._chunks: list[DocumentChunk] = []

    def _init_index(self):
        import faiss
        self._index = faiss.IndexFlatIP(self._dimension)

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        import faiss
        if self._index is None:
            self._init_index()

        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self._index.add(vectors)
        self._chunks.extend(chunks)
        logger.info(f"FAISS: indexed {len(chunks)} chunks (total: {len(self._chunks)})")

    def search(self, query_embedding: list[float], top_k: int = 20) -> list[DocumentChunk]:
        if self._index is None or len(self._chunks) == 0:
            return []

        import faiss
        vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(vector)
        k = min(top_k, len(self._chunks))
        scores, indices = self._index.search(vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                chunk = self._chunks[idx]
                chunk.metadata["retrieval_score"] = float(score)
                results.append(chunk)

        return results

    def save(self) -> None:
        import faiss
        faiss.write_index(self._index, str(self._index_path / "index.faiss"))
        with (self._index_path / "chunks.pkl").open("wb") as f:
            pickle.dump(self._chunks, f)
        logger.info(f"FAISS index saved to {self._index_path}")

    def load(self) -> None:
        import faiss
        index_file = self._index_path / "index.faiss"
        chunks_file = self._index_path / "chunks.pkl"
        if not index_file.exists():
            raise FileNotFoundError(f"No FAISS index found at {self._index_path}")
        self._index = faiss.read_index(str(index_file))
        with chunks_file.open("rb") as f:
            self._chunks = pickle.load(f)
        logger.info(f"FAISS: loaded {len(self._chunks)} chunks from {self._index_path}")

    def __len__(self) -> int:
        return len(self._chunks)


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB-based vector store.

    Uses Chroma's persistence to store both vectors and metadata.
    """

    def __init__(
        self,
        persist_dir: str = ".cache/chroma",
        collection_name: str = "rag_documents",
    ):
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        collection = self._get_collection()
        collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source": c.source,
                    "doc_type": c.doc_type,
                    "chunk_index": c.chunk_index,
                    **{k: str(v) for k, v in c.metadata.items()},
                }
                for c in chunks
            ],
        )
        logger.info(f"Chroma: indexed {len(chunks)} chunks")

    def search(self, query_embedding: list[float], top_k: int = 20) -> list[DocumentChunk]:
        collection = self._get_collection()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
        )
        chunks = []
        for i, (doc_id, text, meta) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
            )
        ):
            chunk = DocumentChunk(
                chunk_id=doc_id,
                text=text,
                source=meta.get("source", ""),
                doc_type=meta.get("doc_type", ""),
                chunk_index=int(meta.get("chunk_index", 0)),
                metadata={
                    **meta,
                    "retrieval_score": 1 - results["distances"][0][i],
                },
            )
            chunks.append(chunk)
        return chunks

    def save(self) -> None:
        # Chroma auto-persists; this is a no-op
        logger.debug("Chroma: auto-persisted")

    def load(self) -> None:
        self._get_collection()

    def __len__(self) -> int:
        return self._get_collection().count()


def create_vector_store(
    provider: str = "faiss",
    index_path: str = ".cache/faiss_index",
    chroma_persist_dir: str = ".cache/chroma",
    collection_name: str = "rag_documents",
    dimension: int = 384,
) -> BaseVectorStore:
    """
    Factory function for vector stores.

    Args:
        provider: 'faiss' or 'chroma'.
        index_path: FAISS index directory.
        chroma_persist_dir: ChromaDB persist path.
        collection_name: ChromaDB collection name.
        dimension: Embedding dimension (required for FAISS).

    Returns:
        BaseVectorStore instance.
    """
    if provider == "faiss":
        return FAISSVectorStore(index_path=index_path, dimension=dimension)
    elif provider == "chroma":
        return ChromaVectorStore(
            persist_dir=chroma_persist_dir, collection_name=collection_name
        )
    else:
        raise ValueError(f"Unknown vector store provider: {provider}")


# ---------------------------------------------------------------------------
# Embedding Pipeline Orchestrator
# ---------------------------------------------------------------------------

class EmbeddingPipeline:
    """
    Orchestrates the full embedding pipeline:
      1. Batch-embed document chunks
      2. Store in vector database
      3. Persist for later retrieval
    """

    def __init__(self, embedder: BaseEmbedder, vector_store: BaseVectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def index(self, chunks: list[DocumentChunk], batch_size: int = 64) -> None:
        """
        Embed and index a list of document chunks.

        Args:
            chunks: Chunks to embed and store.
            batch_size: Embedding batch size.
        """
        if not chunks:
            logger.warning("No chunks to index.")
            return

        logger.info(f"Embedding {len(chunks)} chunks...")
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)
        self.vector_store.add_chunks(chunks, embeddings)
        self.vector_store.save()
        logger.success(f"Indexed {len(chunks)} chunks successfully.")

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.

        Args:
            query: Query text.

        Returns:
            Embedding vector.
        """
        return self.embedder.embed([query])[0]
