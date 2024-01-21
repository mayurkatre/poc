"""
RAG Query CLI
=============
Interactive and single-shot query interface for the RAG system.

Usage:
  python query.py "What is retrieval augmented generation?"
  python query.py --interactive
  python query.py "Explain HyDE" --no-rerank --top-k 3
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

import config
from ingestion.embedding_pipeline import (
    EmbeddingPipeline,
    create_embedder,
    create_vector_store,
)
from reranking.cross_encoder import create_reranker
from retrieval.hybrid_search import BM25Index, HybridRetriever
from retrieval.hyde_retriever import HyDERetriever
from generation.rag_pipeline import RAGPipeline

console = Console()
app = typer.Typer(help="RAG Query CLI", add_completion=False)


def _build_pipeline(
    cfg: dict,
    use_hyde: bool = False,
    no_rerank: bool = False,
    top_k: Optional[int] = None,
) -> RAGPipeline:
    """Initialize the full RAG pipeline from config."""
    emb_cfg = cfg.get("embedding", {})
    vs_cfg = cfg.get("vector_store", {})
    ret_cfg = cfg.get("retrieval", {})
    rr_cfg = cfg.get("reranking", {})
    gen_cfg = cfg.get("generation", {})

    embedder = create_embedder(
        provider=emb_cfg.get("provider", "sentence_transformers"),
        model_name=emb_cfg.get("model_name", "all-MiniLM-L6-v2"),
        cache_enabled=emb_cfg.get("cache_enabled", True),
    )

    vector_store = create_vector_store(
        provider=vs_cfg.get("provider", "faiss"),
        index_path=vs_cfg.get("index_path", ".cache/faiss_index"),
        chroma_persist_dir=vs_cfg.get("chroma_persist_dir", ".cache/chroma"),
        collection_name=vs_cfg.get("collection_name", "rag_documents"),
        dimension=embedder.dimension,
    )

    try:
        vector_store.load()
    except FileNotFoundError:
        console.print(
            "[red]No index found. Run `python ingest.py ./documents` first.[/red]"
        )
        raise typer.Exit(1)

    embedding_pipeline = EmbeddingPipeline(embedder, vector_store)

    bm25_index = BM25Index()
    if hasattr(vector_store, "_chunks") and vector_store._chunks:
        bm25_index.build(vector_store._chunks)

    final_top_k = top_k or ret_cfg.get("final_top_k", 5)

    if use_hyde:
        retriever = HyDERetriever(
            vector_store=vector_store,
            embedding_pipeline=embedding_pipeline,
            top_k=vs_cfg.get("top_k", 20),
            llm_model=gen_cfg.get("model", "gpt-4o-mini"),
        )
    else:
        retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_pipeline=embedding_pipeline,
            bm25_index=bm25_index,
            top_k=vs_cfg.get("top_k", 20),
            final_top_k=final_top_k,
            use_mmr=ret_cfg.get("mmr_enabled", True),
        )

    reranker = create_reranker(
        enabled=not no_rerank and rr_cfg.get("enabled", True),
        model=rr_cfg.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        top_n=rr_cfg.get("top_n", 5),
    )

    return RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        llm_model=gen_cfg.get("model", "gpt-4o-mini"),
        temperature=gen_cfg.get("temperature", 0.0),
        max_tokens=gen_cfg.get("max_tokens", 1024),
        query_rewriting_enabled=cfg.get("query_rewriting", {}).get("enabled", True),
        cache_enabled=True,
    )


def _print_response(question: str, pipeline: RAGPipeline):
    """Execute query and print formatted response."""
    console.print(f"\n[dim]Querying: {question}[/dim]")

    response = pipeline.query(question)

    # Print answer
    console.print(
        Panel(
            Markdown(response.answer),
            title="[bold green]Answer[/bold green]",
            border_style="green",
        )
    )

    # Print sources
    if response.sources:
        table = Table(
            title="Sources",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", width=3)
        table.add_column("Document", style="cyan")
        table.add_column("Chunk", justify="right", width=8)
        table.add_column("Page", justify="right", width=6)
        table.add_column("Preview", no_wrap=False)

        for i, src in enumerate(response.sources, start=1):
            table.add_row(
                str(i),
                src.document,
                src.chunk_id[:8],
                str(src.page_number or "—"),
                src.text[:120] + "..." if len(src.text) > 120 else src.text,
            )

        console.print(table)

    console.print(
        f"\n[dim]Strategy: {response.retrieval_strategy} | "
        f"Sources: {len(response.sources)} | "
        f"Latency: {response.latency_ms:.0f}ms[/dim]"
    )


@app.command()
def query(
    question: Optional[str] = typer.Argument(None, help="Question to ask"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
    hyde: bool = typer.Option(False, "--hyde", help="Use HyDE retrieval"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Disable reranking"),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Number of sources to return"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """
    Query the RAG system.

    Ask a question and receive an answer grounded in your documents,
    complete with source citations.
    """
    cfg = config.get_settings()
    pipeline = _build_pipeline(cfg, use_hyde=hyde, no_rerank=no_rerank, top_k=top_k)

    if interactive:
        console.print("\n[bold blue]RAG Interactive Query[/bold blue]")
        console.print("Type your question and press Enter. Type 'exit' to quit.\n")

        while True:
            try:
                q = console.input("[bold cyan]Question:[/bold cyan] ").strip()
                if q.lower() in ("exit", "quit", "q"):
                    console.print("[dim]Goodbye![/dim]")
                    break
                if not q:
                    continue

                if json_output:
                    import json
                    response = pipeline.query(q)
                    console.print_json(json.dumps(response.to_dict(), indent=2))
                else:
                    _print_response(q, pipeline)

            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted. Goodbye![/dim]")
                break

    elif question:
        if json_output:
            import json
            response = pipeline.query(question)
            print(json.dumps(response.to_dict(), indent=2))
        else:
            _print_response(question, pipeline)

    else:
        console.print("[red]Provide a question or use --interactive mode.[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
