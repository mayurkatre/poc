"""
RAG Evaluation Metrics
=======================
Implements evaluation metrics for retrieval and generation quality.

Retrieval Metrics:
  - Precision@K, Recall@K, MRR

Generation Metrics (LLM-as-judge via OpenRouter):
  - Faithfulness: Is the answer grounded in the retrieved context?
  - Answer Relevancy: Does the answer address the question?
  - Context Precision: Are the retrieved chunks actually useful?
  - Context Recall: Were all necessary facts retrieved?
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

from loguru import logger

from config.openrouter import get_client


# ---------------------------------------------------------------------------
# Retrieval Metrics
# ---------------------------------------------------------------------------

@dataclass
class RetrievalMetrics:
    query_id: str
    precision_at_k: float
    recall_at_k: float
    mrr: float
    num_retrieved: int
    num_relevant: int


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return sum(1 for rid in top_k if rid in relevant_ids) / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 1.0
    top_k = retrieved_ids[:k]
    return sum(1 for rid in top_k if rid in relevant_ids) / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Generation Metrics (LLM-as-Judge via OpenRouter)
# ---------------------------------------------------------------------------

@dataclass
class GenerationMetrics:
    query_id: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


FAITHFULNESS_PROMPT = """You are evaluating a RAG system's answer for faithfulness.

QUESTION: {question}

CONTEXT (retrieved documents):
{context}

GENERATED ANSWER: {answer}

TASK: Rate how faithfully the answer is grounded in the provided context.
- Score 1.0: Every claim in the answer is directly supported by the context
- Score 0.5: Most claims are supported, but some may be inferred
- Score 0.0: Answer contains significant information not present in context

Respond with ONLY a JSON object: {{"score": <float 0.0-1.0>, "reason": "<brief explanation>"}}"""

ANSWER_RELEVANCY_PROMPT = """You are evaluating whether a RAG answer is relevant to the question.

QUESTION: {question}

GENERATED ANSWER: {answer}

TASK: Rate how well the answer addresses the question.
- Score 1.0: Answer directly and completely addresses the question
- Score 0.5: Answer partially addresses the question
- Score 0.0: Answer does not address the question at all

Respond with ONLY a JSON object: {{"score": <float 0.0-1.0>, "reason": "<brief explanation>"}}"""

CONTEXT_PRECISION_PROMPT = """You are evaluating the precision of retrieved context.

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

TASK: What fraction of the retrieved context chunks are relevant to answering the question?
- Score 1.0: All retrieved chunks are relevant
- Score 0.5: About half the chunks are relevant
- Score 0.0: No chunks are relevant

Respond with ONLY a JSON object: {{"score": <float 0.0-1.0>, "reason": "<brief explanation>"}}"""

CONTEXT_RECALL_PROMPT = """You are evaluating whether the retrieved context contains all necessary information.

QUESTION: {question}

GROUND TRUTH ANSWER: {ground_truth}

RETRIEVED CONTEXT:
{context}

TASK: Does the retrieved context contain all the information needed to produce the ground truth answer?
- Score 1.0: Context contains all necessary information
- Score 0.5: Context contains some but not all necessary information
- Score 0.0: Context is missing critical information

Respond with ONLY a JSON object: {{"score": <float 0.0-1.0>, "reason": "<brief explanation>"}}"""


class LLMJudge:
    """
    Uses OpenRouter to score RAG responses on multiple dimensions.

    Implements the "LLM-as-judge" evaluation paradigm (RAGAS-style).
    Any OpenRouter model can be used as the judge.
    """

    def __init__(self, model: str = "openai/gpt-4o-mini"):
        """
        Args:
            model: OpenRouter model string for the judge.
                   Recommended: 'openai/gpt-4o-mini' (fast + cheap)
                   Higher quality: 'openai/gpt-4o' or 'anthropic/claude-sonnet-4-5'
        """
        self.model = model

    def _judge(self, prompt: str) -> tuple[float, str]:
        """Call OpenRouter judge and parse score from JSON response."""
        import json

        client = get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
            parsed = json.loads(raw)
            return float(parsed["score"]), parsed.get("reason", "")
        except Exception as e:
            logger.warning(f"LLM judge error [{self.model}]: {e}")
            return 0.5, f"Error: {e}"

    def _format_context(self, context_chunks: list[str]) -> str:
        return "\n\n".join(
            f"[Chunk {i+1}]: {chunk[:400]}"
            for i, chunk in enumerate(context_chunks)
        )

    def score_faithfulness(self, question: str, answer: str, context_chunks: list[str]) -> tuple[float, str]:
        context = self._format_context(context_chunks)
        return self._judge(FAITHFULNESS_PROMPT.format(question=question, context=context, answer=answer))

    def score_answer_relevancy(self, question: str, answer: str) -> tuple[float, str]:
        return self._judge(ANSWER_RELEVANCY_PROMPT.format(question=question, answer=answer))

    def score_context_precision(self, question: str, context_chunks: list[str]) -> tuple[float, str]:
        context = self._format_context(context_chunks)
        return self._judge(CONTEXT_PRECISION_PROMPT.format(question=question, context=context))

    def score_context_recall(self, question: str, ground_truth: str, context_chunks: list[str]) -> tuple[float, str]:
        context = self._format_context(context_chunks)
        return self._judge(CONTEXT_RECALL_PROMPT.format(question=question, ground_truth=ground_truth, context=context))

    def evaluate_sample(
        self,
        query_id: str,
        question: str,
        answer: str,
        context_chunks: list[str],
        ground_truth: str,
    ) -> GenerationMetrics:
        """
        Run all generation metrics for a single sample via OpenRouter.

        Args:
            query_id: Sample identifier.
            question: Original question.
            answer: Generated answer.
            context_chunks: Retrieved context texts.
            ground_truth: Reference answer.

        Returns:
            GenerationMetrics with all scores.
        """
        faithfulness, _ = self.score_faithfulness(question, answer, context_chunks)
        relevancy, _ = self.score_answer_relevancy(question, answer)
        precision, _ = self.score_context_precision(question, context_chunks)
        recall, _ = self.score_context_recall(question, ground_truth, context_chunks)

        logger.debug(
            f"[{query_id}] Judge [{self.model}]: "
            f"faith={faithfulness:.2f} rel={relevancy:.2f} "
            f"prec={precision:.2f} rec={recall:.2f}"
        )

        return GenerationMetrics(
            query_id=query_id,
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            context_precision=precision,
            context_recall=recall,
        )


# ---------------------------------------------------------------------------
# Aggregate Statistics
# ---------------------------------------------------------------------------

@dataclass
class EvaluationSummary:
    num_samples: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
    avg_precision_at_k: float
    avg_recall_at_k: float
    avg_mrr: float

    def to_dict(self) -> dict:
        return {
            "num_samples": self.num_samples,
            "generation": {
                "faithfulness": round(self.avg_faithfulness, 4),
                "answer_relevancy": round(self.avg_answer_relevancy, 4),
                "context_precision": round(self.avg_context_precision, 4),
                "context_recall": round(self.avg_context_recall, 4),
            },
            "retrieval": {
                "precision_at_k": round(self.avg_precision_at_k, 4),
                "recall_at_k": round(self.avg_recall_at_k, 4),
                "mrr": round(self.avg_mrr, 4),
            },
        }


def compute_summary(
    generation_metrics: list[GenerationMetrics],
    retrieval_metrics: list[RetrievalMetrics],
) -> EvaluationSummary:
    def avg(vals):
        return sum(vals) / len(vals) if vals else 0.0

    return EvaluationSummary(
        num_samples=max(len(generation_metrics), len(retrieval_metrics)),
        avg_faithfulness=avg([m.faithfulness for m in generation_metrics]),
        avg_answer_relevancy=avg([m.answer_relevancy for m in generation_metrics]),
        avg_context_precision=avg([m.context_precision for m in generation_metrics]),
        avg_context_recall=avg([m.context_recall for m in generation_metrics]),
        avg_precision_at_k=avg([m.precision_at_k for m in retrieval_metrics]),
        avg_recall_at_k=avg([m.recall_at_k for m in retrieval_metrics]),
        avg_mrr=avg([m.mrr for m in retrieval_metrics]),
    )
