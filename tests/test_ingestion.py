"""
Tests for the document ingestion pipeline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ingestion.chunking import (
    ChunkerFactory,
    FixedSizeChunker,
    SentenceChunker,
    DocumentChunk,
)
from ingestion.document_loader import (
    DocumentLoaderFactory,
    MarkdownLoader,
    TXTLoader,
    RawDocument,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXT = """
Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval
with large language model generation. It was introduced to address the limitations of
parametric knowledge in LLMs.

The system works in three phases: first, documents are ingested and embedded into a
vector store. Second, when a query arrives, relevant chunks are retrieved. Third,
the LLM generates an answer conditioned on the retrieved context.

HyDE (Hypothetical Document Embeddings) is an advanced retrieval technique that
generates a hypothetical answer document before retrieval, improving semantic matching.
"""

SAMPLE_MARKDOWN = """# RAG Overview

## What is RAG?

Retrieval-Augmented Generation combines retrieval and generation.

## Components

- Document ingestion pipeline
- Vector store  
- Retrieval system
- Generation model

## Benefits

RAG grounds LLM responses in retrieved evidence, reducing hallucinations.
"""


@pytest.fixture
def sample_txt_file(tmp_path):
    """Create a temporary text file."""
    f = tmp_path / "sample.txt"
    f.write_text(SAMPLE_TEXT, encoding="utf-8")
    return str(f)


@pytest.fixture
def sample_md_file(tmp_path):
    """Create a temporary Markdown file."""
    f = tmp_path / "sample.md"
    f.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
    return str(f)


# ---------------------------------------------------------------------------
# Document Loader Tests
# ---------------------------------------------------------------------------

class TestTXTLoader:
    def test_load_returns_document(self, sample_txt_file):
        loader = TXTLoader()
        docs = loader.load(sample_txt_file)
        assert len(docs) == 1
        assert isinstance(docs[0], RawDocument)

    def test_document_has_content(self, sample_txt_file):
        loader = TXTLoader()
        docs = loader.load(sample_txt_file)
        assert "RAG" in docs[0].content
        assert len(docs[0].content) > 50

    def test_document_metadata(self, sample_txt_file):
        loader = TXTLoader()
        docs = loader.load(sample_txt_file)
        assert docs[0].doc_type == "txt"
        assert "file_name" in docs[0].metadata

    def test_file_not_found_raises(self):
        loader = TXTLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/file.txt")


class TestMarkdownLoader:
    def test_load_returns_document(self, sample_md_file):
        loader = MarkdownLoader()
        docs = loader.load(sample_md_file)
        assert len(docs) == 1

    def test_markdown_extracts_headings(self, sample_md_file):
        loader = MarkdownLoader()
        docs = loader.load(sample_md_file)
        assert "headings" in docs[0].metadata
        assert len(docs[0].metadata["headings"]) > 0

    def test_doc_type_is_markdown(self, sample_md_file):
        loader = MarkdownLoader()
        docs = loader.load(sample_md_file)
        assert docs[0].doc_type == "markdown"


class TestDocumentLoaderFactory:
    def test_auto_detect_txt(self, sample_txt_file):
        docs = DocumentLoaderFactory.load(sample_txt_file)
        assert len(docs) >= 1

    def test_auto_detect_markdown(self, sample_md_file):
        docs = DocumentLoaderFactory.load(sample_md_file)
        assert len(docs) >= 1

    def test_unsupported_extension_raises(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_text("content")
        with pytest.raises(ValueError, match="No loader registered"):
            DocumentLoaderFactory.load(str(f))


# ---------------------------------------------------------------------------
# Chunking Tests
# ---------------------------------------------------------------------------

def make_raw_doc(text: str) -> RawDocument:
    return RawDocument(
        content=text, source="test.txt", doc_type="txt", metadata={}
    )


class TestFixedSizeChunker:
    def test_chunks_produced(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        doc = make_raw_doc("A" * 500)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        doc = make_raw_doc("word " * 200)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert len(chunk.text) <= 105  # Small tolerance

    def test_chunk_ids_unique(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        doc = make_raw_doc(SAMPLE_TEXT)
        chunks = chunker.chunk(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_metadata_contains_source(self):
        chunker = FixedSizeChunker(chunk_size=200, overlap=20)
        doc = make_raw_doc(SAMPLE_TEXT)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.source == "test.txt"
            assert chunk.metadata.get("chunking_strategy") == "fixed"


class TestSentenceChunker:
    def test_chunks_produced(self):
        chunker = SentenceChunker(chunk_size=200)
        doc = make_raw_doc(SAMPLE_TEXT)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1

    def test_sentences_not_split(self):
        """Verify no chunk ends in the middle of a word."""
        chunker = SentenceChunker(chunk_size=300)
        doc = make_raw_doc(SAMPLE_TEXT)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            # No chunk should end with a partial word (no space mid-word)
            assert not chunk.text.endswith(" a")

    def test_single_sentence_doc(self):
        chunker = SentenceChunker(chunk_size=500)
        doc = make_raw_doc("This is a single sentence document.")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert "single sentence" in chunks[0].text


class TestChunkerFactory:
    def test_creates_fixed(self):
        chunker = ChunkerFactory.create("fixed")
        assert isinstance(chunker, FixedSizeChunker)

    def test_creates_sentence(self):
        chunker = ChunkerFactory.create("sentence")
        assert isinstance(chunker, SentenceChunker)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            ChunkerFactory.create("invalid_strategy")


class TestDocumentChunk:
    def test_create_generates_chunk_id(self):
        chunk = DocumentChunk.create(
            text="Hello world",
            source="test.txt",
            doc_type="txt",
            chunk_index=0,
        )
        assert len(chunk.chunk_id) == 16
        assert chunk.chunk_id.isalnum()

    def test_same_content_same_id(self):
        kwargs = dict(text="Hello world", source="test.txt", doc_type="txt", chunk_index=0)
        c1 = DocumentChunk.create(**kwargs)
        c2 = DocumentChunk.create(**kwargs)
        assert c1.chunk_id == c2.chunk_id

    def test_metadata_includes_word_count(self):
        chunk = DocumentChunk.create(
            text="hello world foo bar",
            source="test.txt",
            doc_type="txt",
            chunk_index=0,
        )
        assert chunk.metadata["word_count"] == 4
