"""
Document Ingestion CLI
=======================
Ingests one or more documents or directories into the RAG vector store.

Usage:
  python ingest.py ./documents
  python ingest.py ./docs/architecture.md
  python ingest.py https://example.com/article
  python ingest.py ./docs --strategy fixed --chunk-size 256
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

import config
from ingestion.chunking import ChunkerFactory, DocumentChunk
from ingestion.document_loader import DocumentLoaderFactory, RawDocument
from ingestion.embedding_pipeline import (
    EmbeddingPipeline,
    create_embedder,
    create_vector_store,
)
from retrieval.hybrid_search import BM25Index

console = Console()
app = typer.Typer(help="RAG Document Ingestion CLI", add_completion=False)

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}


def _collect_sources(path_str: str) -> list[str]:
    """
    Collect all document sources from a path (file, directory, or URL).

    Args:
        path_str: File path, directory path, or URL.

    Returns:
        List of source strings (paths or URLs).
    """
    if path_str.startswith("http://") or path_str.startswith("https://"):
        return [path_str]

    path = Path(path_str)
    if not path.exists():
        console.print(f"[red]Path not found: {path_str}[/red]")
        return []

    if path.is_file():
        return [str(path)]

    # Directory: recursively find supported files
    sources = []
    for ext in SUPPORTED_EXTENSIONS:
        sources.extend(str(p) for p in path.rglob(f"*{ext}"))

    sources.sort()
    return sources


@app.command()
def ingest(
    source: str = typer.Argument(..., help="File path, directory, or URL to ingest"),
    strategy: Optional[str] = typer.Option(
        None, help="Chunking strategy: semantic, sentence, fixed"
    ),
    chunk_size: Optional[int] = typer.Option(None, help="Chunk size in characters"),
    overlap: Optional[int] = typer.Option(None, help="Chunk overlap in characters"),
    vector_store_provider: Optional[str] = typer.Option(
        None, help="Vector store: faiss or chroma"
    ),
    dry_run: bool = typer.Option(
        False, help="Show what would be ingested without indexing"
    ),
):
    """
    Ingest documents into the RAG vector store.

    Supports PDF, Markdown, TXT files and web URLs.
    Directories are recursively scanned for supported file types.
    """
    console.print("\n[bold blue]RAG Document Ingestion[/bold blue]")
    console.print("=" * 50)

    # Load config
    cfg = config.get_settings()
    ing_cfg = cfg.get("ingestion", {})
    emb_cfg = cfg.get("embedding", {})
    vs_cfg = cfg.get("vector_store", {})

    # Override config with CLI args
    final_strategy = strategy or ing_cfg.get("chunking_strategy", "semantic")
    final_chunk_size = chunk_size or ing_cfg.get("chunk_size", 512)
    final_overlap = overlap or ing_cfg.get("chunk_overlap", 64)
    final_vs_provider = vector_store_provider or vs_cfg.get("provider", "faiss")

    console.print(f"Source:           {source}")
    console.print(f"Strategy:         {final_strategy}")
    console.print(f"Chunk size:       {final_chunk_size} chars")
    console.print(f"Overlap:          {final_overlap} chars")
    console.print(f"Vector store:     {final_vs_provider}")
    console.print(f"Embedding model:  {emb_cfg.get('model_name', 'all-MiniLM-L6-v2')}")

    # Collect sources
    sources = _collect_sources(source)
    if not sources:
        console.print("[yellow]No documents found.[/yellow]")
        raise typer.Exit(0)

    console.print(f"\nFound [bold]{len(sources)}[/bold] document(s) to ingest")
    for src in sources[:10]:
        console.print(f"  • {src}")
    if len(sources) > 10:
        console.print(f"  ... and {len(sources) - 10} more")

    if dry_run:
        console.print("\n[yellow]Dry run — no indexing performed.[/yellow]")
        raise typer.Exit(0)

    # Initialize pipeline components
    console.print("\n[dim]Initializing pipeline...[/dim]")

    embedder = create_embedder(
        provider=emb_cfg.get("provider", "sentence_transformers"),
        model_name=emb_cfg.get("model_name", "all-MiniLM-L6-v2"),
        cache_enabled=emb_cfg.get("cache_enabled", True),
        cache_dir=emb_cfg.get("cache_dir", ".cache/embeddings"),
    )

    vector_store = create_vector_store(
        provider=final_vs_provider,
        index_path=vs_cfg.get("index_path", ".cache/faiss_index"),
        chroma_persist_dir=vs_cfg.get("chroma_persist_dir", ".cache/chroma"),
        collection_name=vs_cfg.get("collection_name", "rag_documents"),
        dimension=embedder.dimension,
    )

    # Try to load existing index for incremental updates
    try:
        vector_store.load()
        console.print(
            f"[green]Loaded existing index:[/green] {len(vector_store)} chunks"
        )
    except FileNotFoundError:
        console.print("[dim]No existing index — creating fresh index.[/dim]")

    embedding_pipeline = EmbeddingPipeline(embedder, vector_store)
    chunker = ChunkerFactory.create(
        strategy=final_strategy,
        chunk_size=final_chunk_size,
        overlap=final_overlap,
    )

    # Process documents
    total_docs = 0
    total_chunks = 0
    failed = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting documents...", total=len(sources))

        for src in sources:
            progress.update(task, description=f"Processing: {Path(src).name[:40]}")

            try:
                # Load document
                raw_docs: list[RawDocument] = DocumentLoaderFactory.load(src)

                # Chunk documents
                chunks: list[DocumentChunk] = []
                for doc in raw_docs:
                    chunks.extend(chunker.chunk(doc))

                if not chunks:
                    logger.warning(f"No chunks produced for {src}")
                    progress.advance(task)
                    continue

                # Index chunks
                embedding_pipeline.index(chunks)
                total_docs += len(raw_docs)
                total_chunks += len(chunks)

                logger.info(
                    f"Ingested {src}: "
                    f"{len(raw_docs)} pages → {len(chunks)} chunks"
                )

            except Exception as e:
                logger.error(f"Failed to ingest {src}: {e}")
                failed.append((src, str(e)))

            progress.advance(task)

    # Build BM25 index
    console.print("\n[dim]Building BM25 keyword index...[/dim]")
    if hasattr(vector_store, "_chunks") and vector_store._chunks:
        bm25 = BM25Index()
        bm25.build(vector_store._chunks)
        console.print(
            f"[green]BM25 index built:[/green] {len(vector_store._chunks)} documents"
        )

    # Summary
    console.print("\n[bold green]Ingestion Complete![/bold green]")
    console.print(f"  Documents processed:  {total_docs}")
    console.print(f"  Chunks indexed:       {total_chunks}")
    console.print(f"  Total indexed:        {len(vector_store)}")
    console.print(f"  Failed:               {len(failed)}")

    if failed:
        console.print("\n[yellow]Failed sources:[/yellow]")
        for src, err in failed:
            console.print(f"  [red]✗[/red] {src}: {err}")

    console.print(
        f"\n[dim]Index saved to: "
        f"{vs_cfg.get('index_path', '.cache/faiss_index')}[/dim]"
    )


if __name__ == "__main__":
    app()
