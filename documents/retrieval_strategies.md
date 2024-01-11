# Advanced Retrieval Strategies

## Dense Retrieval

Dense retrieval uses neural embedding models to encode both queries and documents as dense vectors in a shared semantic space. Similarity is computed using inner product or cosine similarity.

The core advantage of dense retrieval is its ability to match semantically equivalent phrases even when they use completely different vocabulary. For example, "automobile accident" and "car crash" would have high similarity despite sharing no words.

Popular dense retrieval models include bi-encoders like sentence-transformers (e.g., all-MiniLM-L6-v2, all-mpnet-base-v2), OpenAI's text-embedding models, and Cohere's embed models.

## Sparse Retrieval (BM25)

BM25 (Best Match 25) is a classic sparse retrieval algorithm that scores documents based on term frequency and inverse document frequency. It excels at exact keyword matching and is highly effective for queries that contain specific technical terms, proper nouns, or identifiers.

The BM25 score for a document D given query Q is computed as:

```
score(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D|/avgdl))
```

Where f(qi, D) is the term frequency, |D| is document length, avgdl is average document length, and k1 (typically 1.5) and b (typically 0.75) are tuning parameters.

## Hybrid Search

Hybrid search combines dense and sparse retrieval to leverage the complementary strengths of both approaches. Documents retrieved by dense search that are missed by BM25 (due to vocabulary mismatch) can be recovered, while BM25 can catch exact matches that embedding similarity misses.

### Reciprocal Rank Fusion (RRF)

RRF is a common fusion strategy for combining multiple ranked lists without requiring score normalization:

```
RRF_score(d) = Σ 1 / (k + rank(d, list_i))
```

The constant k (typically 60) prevents high-ranked results from dominating the fusion. RRF is parameter-free (beyond k) and robust to score scale differences between rankers.

## HyDE (Hypothetical Document Embeddings)

HyDE (Gao et al., 2022) is a novel retrieval technique that bridges the vocabulary gap between user queries and document content. Instead of embedding the raw query, HyDE:

1. Uses an LLM to generate a hypothetical answer document for the query
2. Embeds the hypothetical document (which uses richer, document-like vocabulary)
3. Uses the hypothetical document's embedding to retrieve real documents

HyDE significantly improves retrieval for complex, technical queries where the query is short and the relevant documents use specialized vocabulary. However, it adds latency due to the extra LLM call.

## Maximum Marginal Relevance (MMR)

MMR (Carbonell & Goldstein, 1998) addresses the problem of redundant retrieval results. Standard similarity search tends to return multiple chunks from the same document section, wasting the context window.

MMR iteratively selects documents by maximizing:

```
MMR_score(d) = λ × sim(d, query) - (1 - λ) × max_sim(d, selected_docs)
```

The parameter λ controls the relevance-diversity tradeoff:
- λ = 1.0: Pure relevance ranking (equivalent to standard retrieval)
- λ = 0.0: Maximum diversity (ignores query relevance)
- λ = 0.5: Balanced (recommended default)

## Cross-Encoder Reranking

Cross-encoders jointly encode the query and document together, allowing full self-attention between all query and document tokens. This produces much more accurate relevance scores than bi-encoders but is significantly slower.

The standard pattern is:
1. Retrieve top-20 candidates using fast bi-encoder
2. Rerank using cross-encoder to select top-5

Models like `cross-encoder/ms-marco-MiniLM-L-6-v2` are fine-tuned on the MS MARCO passage ranking dataset and provide strong performance for retrieval tasks.

## Query Rewriting

Query rewriting generates multiple alternative phrasings of the original query and retrieves documents for all variants. Results are merged by deduplication. This improves recall when the original query uses terminology that doesn't match document vocabulary.

Typical implementation:
1. Use LLM to generate 3 query variants
2. Retrieve for each variant independently  
3. Merge results by chunk_id deduplication
4. Proceed with reranking on the merged set
