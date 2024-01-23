"""
RAG Evaluation Runner
======================
Executes the complete evaluation harness:
  1. Load evaluation dataset
  2. Run each question through the RAG pipeline
  3. Score with LLM judge (faithfulness, relevancy, precision, recall)
  4. Compute retrieval metrics
  5. Generate results.json and report.md

Usage:
  python evaluation/evaluate.py
  python evaluation/evaluate.py --dataset evaluation/dataset.json --limit 5
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from evaluation.metrics import (
    GenerationMetrics,
    LLMJudge,
    RetrievalMetrics,
    compute_summary,
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
)
from ingestion.chunking import ChunkerFactory
from ingestion.document_loader import DocumentLoaderFactory
from ingestion.embedding_pipeline import (
    EmbeddingPipeline,
    create_embedder,
    create_vector_store,
)
from reranking.cross_encoder import create_reranker
from retrieval.hybrid_search import BM25Index, HybridRetriever
from generation.rag_pipeline import RAGPipeline

console = Console()
app = typer.Typer(help="RAG Evaluation Harness")


def _build_pipeline(cfg: dict) -> RAGPipeline:
    """Build the full RAG pipeline from config."""
    emb_cfg = cfg.get("embedding", {})
    embedder = create_embedder(
        provider=emb_cfg.get("provider", "sentence_transformers"),
        model_name=emb_cfg.get("model_name", "all-MiniLM-L6-v2"),
        cache_enabled=emb_cfg.get("cache_enabled", True),
    )

    vs_cfg = cfg.get("vector_store", {})
    vector_store = create_vector_store(
        provider=vs_cfg.get("provider", "faiss"),
        index_path=vs_cfg.get("index_path", ".cache/faiss_index"),
        dimension=embedder.dimension,
    )

    try:
        vector_store.load()
        logger.info(f"Loaded index: {len(vector_store)} chunks")
    except FileNotFoundError:
        logger.warning("No index found. Results will be empty without ingestion.")

    embedding_pipeline = EmbeddingPipeline(embedder, vector_store)

    bm25_index = BM25Index()
    if hasattr(vector_store, "_chunks") and vector_store._chunks:
        bm25_index.build(vector_store._chunks)

    ret_cfg = cfg.get("retrieval", {})
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_pipeline=embedding_pipeline,
        bm25_index=bm25_index,
        top_k=vs_cfg.get("top_k", 20),
        final_top_k=ret_cfg.get("final_top_k", 5),
        use_mmr=ret_cfg.get("mmr_enabled", True),
    )

    rr_cfg = cfg.get("reranking", {})
    reranker = create_reranker(
        enabled=rr_cfg.get("enabled", True),
        top_n=rr_cfg.get("top_n", 5),
    )

    gen_cfg = cfg.get("generation", {})
    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        llm_model=gen_cfg.get("model", "gpt-4o-mini"),
        temperature=0.0,
        query_rewriting_enabled=False,  # Disable during eval for consistency
        cache_enabled=False,
    )
    return pipeline


def _generate_markdown_report(
    summary,
    per_sample_results: list[dict],
    output_path: str,
) -> None:
    """Generate a human-readable evaluation report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# RAG System Evaluation Report",
        f"\n**Generated:** {now}",
        f"**Samples evaluated:** {summary.num_samples}",
        "\n---\n",
        "## Summary Metrics",
        "\n### Generation Quality",
        "| Metric | Score |",
        "|--------|-------|",
        f"| Faithfulness | {summary.avg_faithfulness:.3f} |",
        f"| Answer Relevancy | {summary.avg_answer_relevancy:.3f} |",
        f"| Context Precision | {summary.avg_context_precision:.3f} |",
        f"| Context Recall | {summary.avg_context_recall:.3f} |",
        "\n### Retrieval Quality",
        "| Metric | Score |",
        "|--------|-------|",
        f"| Precision@K | {summary.avg_precision_at_k:.3f} |",
        f"| Recall@K | {summary.avg_recall_at_k:.3f} |",
        f"| MRR | {summary.avg_mrr:.3f} |",
        "\n---\n",
        "## Per-Sample Results",
        "",
    ]

    for r in per_sample_results:
        lines += [
            f"### {r['query_id']}: {r['question'][:80]}",
            f"- **Faithfulness:** {r.get('faithfulness', 'N/A')}",
            f"- **Answer Relevancy:** {r.get('answer_relevancy', 'N/A')}",
            f"- **Context Precision:** {r.get('context_precision', 'N/A')}",
            f"- **Context Recall:** {r.get('context_recall', 'N/A')}",
            f"- **Sources retrieved:** {r.get('num_sources', 0)}",
            f"- **Latency:** {r.get('latency_ms', 0):.0f}ms",
            "",
        ]

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report written to {output_path}")


@app.command()
def run(
    dataset: str = typer.Option("evaluation/dataset.json", help="Path to eval dataset"),
    output: str = typer.Option("evaluation/results.json", help="Path for results JSON"),
    report: str = typer.Option("evaluation/report.md", help="Path for Markdown report"),
    limit: Optional[int] = typer.Option(None, help="Limit number of samples"),
    skip_generation_eval: bool = typer.Option(
        False, help="Skip LLM-based generation evaluation (faster)"
    ),
):
    """Run the full RAG evaluation pipeline."""
    console.print("\n[bold blue]RAG Evaluation Harness[/bold blue]")
    console.print("=" * 50)

    # Load config and build pipeline
    cfg = config.get_settings()
    pipeline = _build_pipeline(cfg)
    judge = LLMJudge(model="gpt-4o-mini") if not skip_generation_eval else None

    # Load dataset
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        console.print(f"[red]Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    with dataset_path.open() as f:
        samples = json.load(f)

    if limit:
        samples = samples[:limit]

    console.print(f"Evaluating [bold]{len(samples)}[/bold] samples...")

    generation_metrics: list[GenerationMetrics] = []
    retrieval_metrics: list[RetrievalMetrics] = []
    per_sample_results: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running evaluation...", total=len(samples))

        for sample in samples:
            query_id = sample["id"]
            question = sample["question"]
            ground_truth = sample.get("ground_truth", "")

            progress.update(task, description=f"[{query_id}] {question[:50]}...")

            try:
                # Run RAG pipeline
                t0 = time.monotonic()
                response = pipeline.query(question)
                latency = (time.monotonic() - t0) * 1000

                context_texts = [s.text for s in response.sources]
                retrieved_ids = [s.chunk_id for s in response.sources]

                # Retrieval metrics (we use document name match as proxy)
                relevant_doc = sample.get("relevant_document", "")
                relevant_set = {
                    cid
                    for cid, src in zip(retrieved_ids, [s.document for s in response.sources])
                    if relevant_doc.lower() in src.lower()
                }

                ret_metrics = RetrievalMetrics(
                    query_id=query_id,
                    precision_at_k=precision_at_k(retrieved_ids, relevant_set, k=5),
                    recall_at_k=recall_at_k(retrieved_ids, relevant_set, k=5),
                    mrr=mean_reciprocal_rank(retrieved_ids, relevant_set),
                    num_retrieved=len(retrieved_ids),
                    num_relevant=len(relevant_set),
                )
                retrieval_metrics.append(ret_metrics)

                # Generation metrics
                result_entry = {
                    "query_id": query_id,
                    "question": question,
                    "answer": response.answer,
                    "ground_truth": ground_truth,
                    "sources": [s.document for s in response.sources],
                    "latency_ms": latency,
                    "num_sources": len(response.sources),
                    **ret_metrics.__dict__,
                }

                if judge and context_texts:
                    gen_metrics = judge.evaluate_sample(
                        query_id=query_id,
                        question=question,
                        answer=response.answer,
                        context_chunks=context_texts,
                        ground_truth=ground_truth,
                    )
                    generation_metrics.append(gen_metrics)
                    result_entry.update({
                        "faithfulness": gen_metrics.faithfulness,
                        "answer_relevancy": gen_metrics.answer_relevancy,
                        "context_precision": gen_metrics.context_precision,
                        "context_recall": gen_metrics.context_recall,
                    })

                per_sample_results.append(result_entry)

            except Exception as e:
                logger.error(f"Error evaluating {query_id}: {e}")
                per_sample_results.append({
                    "query_id": query_id,
                    "question": question,
                    "error": str(e),
                })

            progress.advance(task)

    # Compute summary
    summary = compute_summary(generation_metrics, retrieval_metrics)

    # Save results
    results = {
        "summary": summary.to_dict(),
        "per_sample": per_sample_results,
        "evaluated_at": datetime.now().isoformat(),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output}")

    # Generate report
    _generate_markdown_report(summary, per_sample_results, report)

    # Print summary table
    console.print("\n[bold green]Evaluation Complete![/bold green]\n")
    table = Table(title="Evaluation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green", justify="right")

    table.add_row("Faithfulness", f"{summary.avg_faithfulness:.3f}")
    table.add_row("Answer Relevancy", f"{summary.avg_answer_relevancy:.3f}")
    table.add_row("Context Precision", f"{summary.avg_context_precision:.3f}")
    table.add_row("Context Recall", f"{summary.avg_context_recall:.3f}")
    table.add_row("---", "---")
    table.add_row("Retrieval Precision@K", f"{summary.avg_precision_at_k:.3f}")
    table.add_row("Retrieval Recall@K", f"{summary.avg_recall_at_k:.3f}")
    table.add_row("MRR", f"{summary.avg_mrr:.3f}")

    console.print(table)
    console.print(f"\nResults: [blue]{output}[/blue]")
    console.print(f"Report:  [blue]{report}[/blue]")


if __name__ == "__main__":
    app()
