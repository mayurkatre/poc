"""
Document Loader Module
======================
Modular loaders for PDF, Markdown, TXT, and Web URL sources.

Each loader implements the BaseLoader interface and returns a list of
RawDocument objects that feed into the chunking pipeline.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from loguru import logger


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class RawDocument:
    """Represents a loaded document before chunking."""

    content: str
    source: str
    doc_type: str  # pdf | markdown | txt | url
    metadata: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        """Stable hash-based document identifier."""
        return hashlib.sha256(self.source.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Base Loader
# ---------------------------------------------------------------------------

class BaseLoader(ABC):
    """Abstract base class for all document loaders."""

    @abstractmethod
    def load(self, source: str) -> list[RawDocument]:
        """
        Load documents from the given source.

        Args:
            source: File path or URL.

        Returns:
            List of RawDocument objects.
        """
        ...

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove excessive whitespace and normalize line endings."""
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()


# ---------------------------------------------------------------------------
# PDF Loader
# ---------------------------------------------------------------------------

class PDFLoader(BaseLoader):
    """Loads PDF documents page by page using pypdf."""

    def load(self, source: str) -> list[RawDocument]:
        """
        Load a PDF file, extracting text per page.

        Args:
            source: Path to the PDF file.

        Returns:
            List of RawDocument (one per page with page metadata).
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("Install pypdf: pip install pypdf")

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {source}")

        reader = PdfReader(str(path))
        documents: list[RawDocument] = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = self._clean_text(text)
            if not text:
                logger.debug(f"Skipping empty page {page_num} in {source}")
                continue

            documents.append(
                RawDocument(
                    content=text,
                    source=str(path),
                    doc_type="pdf",
                    metadata={
                        "page_number": page_num,
                        "total_pages": len(reader.pages),
                        "file_name": path.name,
                    },
                )
            )

        logger.info(f"Loaded {len(documents)} pages from {path.name}")
        return documents


# ---------------------------------------------------------------------------
# Markdown Loader
# ---------------------------------------------------------------------------

class MarkdownLoader(BaseLoader):
    """Loads Markdown files, preserving structure metadata."""

    def load(self, source: str) -> list[RawDocument]:
        """
        Load a Markdown file as a single document.

        Args:
            source: Path to the Markdown file.

        Returns:
            Single-item list with the full document.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {source}")

        raw = path.read_text(encoding="utf-8")
        text = self._clean_text(raw)

        # Extract headings for metadata
        headings = re.findall(r"^#{1,3}\s+(.+)$", text, re.MULTILINE)

        doc = RawDocument(
            content=text,
            source=str(path),
            doc_type="markdown",
            metadata={
                "file_name": path.name,
                "headings": headings[:10],  # first 10 headings as context
            },
        )
        logger.info(f"Loaded markdown: {path.name} ({len(text)} chars)")
        return [doc]


# ---------------------------------------------------------------------------
# TXT Loader
# ---------------------------------------------------------------------------

class TXTLoader(BaseLoader):
    """Loads plain text files."""

    def load(self, source: str) -> list[RawDocument]:
        """
        Load a plain text file.

        Args:
            source: Path to the text file.

        Returns:
            Single-item list with the document.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {source}")

        text = path.read_text(encoding="utf-8", errors="replace")
        text = self._clean_text(text)

        doc = RawDocument(
            content=text,
            source=str(path),
            doc_type="txt",
            metadata={"file_name": path.name},
        )
        logger.info(f"Loaded text: {path.name} ({len(text)} chars)")
        return [doc]


# ---------------------------------------------------------------------------
# Web URL Loader
# ---------------------------------------------------------------------------

class WebURLLoader(BaseLoader):
    """Loads web pages, extracting main content via html2text."""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def load(self, source: str) -> list[RawDocument]:
        """
        Fetch and extract text from a web URL.

        Args:
            source: HTTP/HTTPS URL.

        Returns:
            Single-item list with the extracted page content.
        """
        try:
            import html2text
            import requests
        except ImportError:
            raise ImportError("Install requests and html2text.")

        parsed = urlparse(source)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

        headers = {"User-Agent": "RAG-POC/1.0 (document ingestion bot)"}
        response = requests.get(source, headers=headers, timeout=self.timeout)
        response.raise_for_status()

        converter = html2text.HTML2Text()
        converter.ignore_links = False
        converter.ignore_images = True
        converter.body_width = 0

        text = converter.handle(response.text)
        text = self._clean_text(text)

        doc = RawDocument(
            content=text,
            source=source,
            doc_type="url",
            metadata={
                "url": source,
                "domain": parsed.netloc,
                "status_code": response.status_code,
            },
        )
        logger.info(f"Loaded URL: {source} ({len(text)} chars)")
        return [doc]


# ---------------------------------------------------------------------------
# Loader Factory
# ---------------------------------------------------------------------------

class DocumentLoaderFactory:
    """
    Factory that selects the appropriate loader based on source type.

    Supports automatic detection from file extension or URL scheme.
    New loaders can be registered without modifying this class.
    """

    _loaders: dict[str, type[BaseLoader]] = {
        ".pdf": PDFLoader,
        ".md": MarkdownLoader,
        ".markdown": MarkdownLoader,
        ".txt": MarkdownLoader,
        "url": WebURLLoader,
    }

    @classmethod
    def register(cls, extension: str, loader_cls: type[BaseLoader]) -> None:
        """Register a new loader for a given extension."""
        cls._loaders[extension] = loader_cls
        logger.debug(f"Registered loader {loader_cls.__name__} for {extension}")

    @classmethod
    def get_loader(cls, source: str) -> BaseLoader:
        """
        Resolve the correct loader for a given source.

        Args:
            source: File path or URL string.

        Returns:
            Instantiated loader.

        Raises:
            ValueError: If no loader is found for the source type.
        """
        if source.startswith("http://") or source.startswith("https://"):
            return cls._loaders["url"]()

        ext = Path(source).suffix.lower()
        if ext not in cls._loaders:
            raise ValueError(
                f"No loader registered for extension '{ext}'. "
                f"Supported: {list(cls._loaders.keys())}"
            )
        return cls._loaders[ext]()

    @classmethod
    def load(cls, source: str) -> list[RawDocument]:
        """
        Convenience method: detect loader and load documents.

        Args:
            source: File path or URL.

        Returns:
            List of RawDocument objects.
        """
        loader = cls.get_loader(source)
        return loader.load(source)
