"""
HyDE Retriever (Hypothetical Document Embeddings)
==================================================
Reference: Gao et al. 2022 - "Precise Zero-Shot Dense Retrieval without Relevance Labels"

Key Idea:
  Instead of embedding the raw query, ask the LLM to generate a hypothetical
  answer document, then embed THAT document for retrieval. This bridges the
  query-document vocabulary gap.

  Query → LLM → Hypothetical Doc → Embed → Vector Search → Real Docs
"""

from __future__ import annotations

import os
from typing import Optional

from loguru import logger

from ingestion.chunking import DocumentChunk
from ingestion.embedding_pipeline import BaseVectorStore, EmbeddingPipeline
from retrieval.base_retriever import BaseRetriever, RetrievalResult


class HyDERetriever(BaseRetriever):
    """
    Retriever using Hypothetical Document Embeddings (HyDE).

    Generates a synthetic answer using an LLM, then uses the embedding
    of that answer to query the vector store. This dramatically improves
    retrieval for technical or complex queries.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_pipeline: EmbeddingPipeline,
        top_k: int = 20,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.7,
    ):
        """
        Args:
            vector_store: Indexed vector store.
            embedding_pipeline: For embedding the hypothetical document.
            top_k: Number of candidates to retrieve.
            llm_model: OpenAI model for hypothesis generation.
            temperature: LLM sampling temperature (higher = more diverse).
        """
        super().__init__(vector_store, embedding_pipeline, top_k)
        self.llm_model = llm_model
        self.temperature = temperature
        self._hyde_prompt = (
            "Please write a detailed, factual paragraph that would answer "
            "the following question. Write as if you are an expert explaining "
            "this topic. Focus on being precise and informative.\n\n"
            "Question: {query}\n\n"
            "Answer:"
        )

    def _generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical answer document using an LLM.

        Args:
            query: The user's question.

        Returns:
            Hypothetical answer text for embedding.
        """
        try:
            import openai
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
            logger.debug(f"HyDE hypothesis: {hypothesis[:120]}...")
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
        logger.info(f"HyDE retrieval for: '{query[:80]}'")

        # Step 1: Generate hypothetical document
        hypothesis = self._generate_hypothetical_document(query)

        # Step 2: Embed the hypothesis (not the raw query)
        hypothesis_embedding = self._embed_query(hypothesis)

        # Step 3: Search vector store with hypothesis embedding
        chunks = self.vector_store.search(hypothesis_embedding, top_k=self.top_k)

        return RetrievalResult(
            chunks=chunks,
            query=query,
            strategy="hyde",
            metadata={
                "hypothesis": hypothesis,
                "hypothesis_length": len(hypothesis),
            },
        )
