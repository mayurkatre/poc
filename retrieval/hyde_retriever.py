"""
HyDE Retriever (Hypothetical Document Embeddings)
==================================================
Reference: Gao et al. 2022 - "Precise Zero-Shot Dense Retrieval without Relevance Labels"

Key Idea:
  Instead of embedding the raw query, ask the LLM to generate a hypothetical
  answer document, then embed THAT document for retrieval. This bridges the
  query-document vocabulary gap.

  Query → LLM (OpenRouter) → Hypothetical Doc → Embed → Vector Search → Real Docs
"""

from __future__ import annotations

from loguru import logger

from config.openrouter import get_client
from ingestion.embedding_pipeline import BaseVectorStore, EmbeddingPipeline
from retrieval.base_retriever import BaseRetriever, RetrievalResult


class HyDERetriever(BaseRetriever):
    """
    Retriever using Hypothetical Document Embeddings (HyDE).

    Generates a synthetic answer via OpenRouter, then uses its embedding
    to query the vector store. Significantly improves retrieval for
    technical or complex queries.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_pipeline: EmbeddingPipeline,
        top_k: int = 20,
        llm_model: str = "openai/gpt-4o-mini",
        temperature: float = 0.7,
    ):
        """
        Args:
            vector_store: Indexed vector store.
            embedding_pipeline: For embedding the hypothetical document.
            top_k: Number of candidates to retrieve.
            llm_model: OpenRouter model string (e.g. 'openai/gpt-4o-mini').
            temperature: Higher temperature = more diverse hypotheses.
        """
        super().__init__(vector_store, embedding_pipeline, top_k)
        self.llm_model = llm_model
        self.temperature = temperature
        self._hyde_prompt = (
            "Please write a detailed, factual paragraph that would answer "
            "the following question. Write as if you are an expert explaining "
            "this topic. Focus on being precise and informative.\n\n"
            "Question: {query}\n\nAnswer:"
        )

    def _generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical answer document via OpenRouter.

        Args:
            query: The user's question.

        Returns:
            Hypothetical answer text for embedding.
        """
        try:
            client = get_client()
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "user",
                        "content": self._hyde_prompt.format(query=query),
                    }
                ],
                temperature=self.temperature,
                max_tokens=256,
            )
            hypothesis = response.choices[0].message.content.strip()
            logger.debug(
                f"HyDE hypothesis [{self.llm_model}]: {hypothesis[:120]}..."
            )
            return hypothesis
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}. Falling back to direct query.")
            return query

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Retrieve using HyDE: generate hypothesis → embed → search.

        Args:
            query: User query.

        Returns:
            RetrievalResult with matched chunks.
        """
        logger.info(f"HyDE retrieval [{self.llm_model}]: '{query[:80]}'")

        hypothesis = self._generate_hypothetical_document(query)
        hypothesis_embedding = self._embed_query(hypothesis)
        chunks = self.vector_store.search(hypothesis_embedding, top_k=self.top_k)

        return RetrievalResult(
            chunks=chunks,
            query=query,
            strategy="hyde",
            metadata={
                "hypothesis": hypothesis,
                "model": self.llm_model,
            },
        )
