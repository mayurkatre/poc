# RAG Overview

## What is Retrieval-Augmented Generation?

Retrieval-Augmented Generation (RAG) is a technique that enhances large language models (LLMs) by grounding their responses in retrieved external knowledge. Instead of relying solely on parametric knowledge encoded during pretraining, RAG systems retrieve relevant documents at inference time and condition the generation on that retrieved context.

RAG was introduced by Lewis et al. (2020) and has since become a foundational architecture for knowledge-intensive NLP tasks, particularly for enterprise applications where accuracy and auditability matter.

## Why RAG?

Large language models suffer from several well-documented limitations:

**Hallucination:** LLMs confidently generate factually incorrect statements, particularly for specific facts, recent events, or domain-specific knowledge not well-represented in training data.

**Knowledge cutoff:** LLMs cannot access information created after their training cutoff date without external retrieval mechanisms.

**Source attribution:** Pure LLMs cannot reliably cite their sources because their knowledge is distributed across billions of parameters, not stored as discrete documents.

**Domain adaptation:** Fine-tuning LLMs on domain-specific data is expensive and requires retraining whenever new information is added.

RAG addresses all four limitations by externalizing the knowledge base into a searchable index, retrieving relevant information at query time, and grounding the LLM's response in that retrieved evidence.

## The RAG Pipeline

A standard RAG system consists of the following stages:

### 1. Document Ingestion

Documents are loaded, cleaned, split into chunks, embedded, and stored in a vector database. This is typically a batch process that runs when new documents are added to the knowledge base.

### 2. Query Processing

When a user submits a query, the system embeds the query using the same embedding model used during ingestion, then performs similarity search against the indexed document embeddings.

### 3. Retrieval

The top-K most similar document chunks are retrieved. Advanced systems use hybrid search (combining dense vector search with sparse keyword search) and apply reranking models to refine the initial retrieval results.

### 4. Generation

The retrieved chunks are formatted as context and provided to the LLM along with the original query. The LLM generates an answer grounded in the provided context, and the system records which document chunks were used.

### 5. Source Attribution

The final response includes citations back to the source documents and specific chunks used, enabling users to verify the information.
