# 🤖 RAG POC — Full Stack RAG System with React Frontend

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![React 18](https://img.shields.io/badge/React-18-blue.svg)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-blue.svg)](https://www.typescriptlang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-ready, full-stack RAG system** with modern React frontend and FastAPI backend, demonstrating advanced retrieval techniques, evaluation methodology, and engineering best practices.

## 🎯 What's New - Full Stack!

This project now includes a **beautiful, modern React frontend** with:
- 🎨 Real-time query interface
- ⚙️ Advanced configuration controls  
- 📚 Live source viewer
- 📱 Responsive design
- 🔄 Live system status

Plus an **enhanced FastAPI backend** with:
- 🔧 Dynamic pipeline configuration
- 🌊 Streaming responses (SSE)
- 🎛️ Advanced query parameters
- 📊 Comprehensive API docs

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG System                               │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Ingestion   │    │  Retrieval   │    │   Generation     │  │
│  │              │    │              │    │                  │  │
│  │ PDF/MD/TXT   │    │ Dense Search │    │  GPT-4o-mini     │  │
│  │ Web URLs     │──▶ │ BM25         │──▶ │  Source Citation │  │
│  │ Semantic     │    │ HyDE         │    │  Query Rewriting │  │
│  │ Chunking     │    │ MMR          │    │  Streaming       │  │
│  │ FAISS/Chroma │    │ Cross-Enc.   │    │  Response Cache  │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  FastAPI     │    │  Evaluation  │    │  ADR Docs        │  │
│  │              │    │              │    │                  │  │
│  │ POST /query  │    │ Faithfulness │    │ 5 Architecture   │  │
│  │ GET  /health │    │ Relevancy    │    │ Decision Records │  │
│  │ POST /ingest │    │ RAGAS-style  │    │                  │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### Core (7/7 Required)
- ✅ **Document Ingestion Pipeline** — PDF, Markdown, TXT, Web URLs with semantic chunking
- ✅ **Advanced Retrieval** — HyDE, MMR, Hybrid Search (Dense + BM25), Cross-encoder reranking
- ✅ **Source Attribution** — Every answer includes document, chunk ID, page number, and text snippet
- ✅ **Evaluation Harness** — RAGAS-style metrics: faithfulness, relevancy, precision, recall
- ✅ **ADR Documentation** — 5 architecture decision records with context, decisions, and consequences
- ✅ **FastAPI Interface** — REST API with streaming, ingest, and health endpoints
- ✅ **Engineering Quality** — Typed Python, docstrings, logging, tests, modular design

### Bonus Features (6/5 Required)
- ✅ **Query Rewriting** — Multi-variant query expansion for improved recall
- ✅ **Streaming Responses** — Server-Sent Events for progressive answer delivery
- ✅ **Embedding Caching** — Disk-based SHA256-keyed cache for zero recomputation
- ✅ **Observability** — Structured logging with loguru, LangSmith integration (optional)
- ✅ **Response Caching** — Query-level response cache for repeated questions
- ✅ **Guardrails** — System prompt constrains generation to retrieved context only

---

## 📁 Project Structure

```
rag-poc/
│
├── frontend/                   # ⭐ NEW! React Frontend
│   ├── src/
│   │   ├── App.tsx            # Main React component
│   │   ├── App.css            # Component styles
│   │   ├── index.css          # Global styles
│   │   └── main.tsx           # Entry point
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── README.md              # Frontend-specific docs
│
├── api/
│   └── app.py                 # Enhanced FastAPI backend
│                              # - Dynamic pipeline config
│                              # - Streaming SSE support
│                              # - Advanced query params
│
├── ingestion/
│   ├── document_loader.py     # PDF, MD, TXT, URL loaders
│   ├── chunking.py            # Semantic/sentence/fixed chunking
│   └── embedding_pipeline.py  # Embeddings + FAISS/Chroma
│
├── retrieval/
│   ├── base_retriever.py      # Abstract retriever interface
│   ├── hyde_retriever.py      # HyDE retrieval
│   ├── hybrid_search.py       # Dense + BM25 + RRF
│   └── mmr.py                 # Maximal Marginal Relevance
│
├── reranking/
│   └── cross_encoder.py       # Cross-encoder reranking
│
├── generation/
│   └── rag_pipeline.py        # Full RAG pipeline
│
├── evaluation/                # RAGAS-style evaluation
├── adr/                       # Architecture Decision Records
├── config/                    # Configuration
├── tests/                     # Unit tests
├── documents/                 # Sample documents
│
├── ingest.py                  # CLI ingestion
├── query.py                   # CLI querying
├── start-fullstack.ps1        # Windows startup script
├── start-fullstack.sh         # Linux/Mac startup script
├── RUN_FULLSTACK.md           # Detailed full-stack guide
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Option A: Full Stack (Recommended)

**Start both frontend and backend:**

Windows PowerShell:
```powershell
.\start-fullstack.ps1
```

Linux/Mac:
```bash
chmod +x start-fullstack.sh
./start-fullstack.sh
```

This will automatically:
1. ✅ Check all dependencies
2. ✅ Ingest documents if needed
3. ✅ Start FastAPI backend (port 8000)
4. ✅ Start React frontend (port 3000)

Then visit: **http://localhost:3000** 🎨

### Option B: Manual Setup

#### Step 1 — Install Dependencies

**Backend:**
```bash
git clone https://github.com/mayurkatre/poc.git
cd poc
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

### Step 2 — Configure

```bash
cp .env.example .env  # Add your OpenRouter key (sk-or-v1-...)
# Add your OpenRouter key — get it free at openrouter.ai/keys
# OPENAI_API_KEY=sk-or-v1-...
# OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

### Step 3 — Ingest Documents

```bash
# Ingest the sample documents
python ingest.py ./documents

# Ingest your own documents (PDF, MD, TXT, or URL)
python ingest.py ./my-docs
python ingest.py ./research-paper.pdf
python ingest.py https://docs.example.com/article
```

### Step 4 — Query via CLI

```bash
# Single question
python query.py "What is retrieval augmented generation?"

# Interactive mode
python query.py --interactive

# Use HyDE retrieval
python query.py "Explain MMR" --hyde

# JSON output
python query.py "What is BM25?" --json
```

### Step 5 — Start API Server

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

**Example API call:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain retrieval augmented generation"}'
```

**Response:**
```json
{
  "answer": "RAG improves LLM reliability by grounding responses...",
  "sources": [
    {
      "document": "rag_overview.md",
      "chunk_id": "a3f8c12d",
      "text": "Retrieval-Augmented Generation (RAG) is a technique...",
      "chunk_index": 2,
      "page_number": null
    }
  ],
  "retrieval_strategy": "hybrid",
  "latency_ms": 847.3
}
```

### Step 6 — Use the Web Interface 🎨

With both servers running (backend on port 8000, frontend on port 3000):

1. **Open your browser**: http://localhost:3000
2. **Type your question** in the query box
3. **Adjust settings** (optional):
   - Enable HyDE for better retrieval
   - Disable reranking for speed
   - Set Top-K results
   - Adjust temperature
4. **Click "Ask RAG"**
5. **View answer** with cited sources

### Step 7 — Run Evaluation

```bash
# Full evaluation with LLM judge
python evaluation/evaluate.py

# Skip LLM evaluation (faster, retrieval metrics only)
python evaluation/evaluate.py --skip-generation-eval

# Limit to first 5 samples
python evaluation/evaluate.py --limit 5
```

Results saved to `evaluation/results.json` and `evaluation/report.md`.

---

## Configuration

Edit `config/settings.yaml` to customize behavior:

```yaml
# Switch retrieval strategy
retrieval:
  strategy: "hybrid"          # hybrid | dense | bm25 | hyde
  mmr_enabled: true
  final_top_k: 5

# Switch embedding provider
embedding:
  provider: "openai"          # sentence_transformers | openai
  model_name: "text-embedding-3-small"

# Switch vector store
vector_store:
  provider: "chroma"          # faiss | chroma

# Disable reranking for lower latency
reranking:
  enabled: false
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=. --cov-report=term-missing

# Specific test file
pytest tests/test_retrieval.py -v
```

---

## Architecture Decision Records

Key design decisions are documented in `adr/`:

| ADR | Decision | Key Tradeoff |
|-----|----------|--------------|
| [ADR-001](adr/ADR-001-vector-database.md) | FAISS as default vector store | Speed vs. managed service |
| [ADR-002](adr/ADR-002-retrieval-strategy.md) | Hybrid search with RRF fusion | Recall vs. latency |
| [ADR-003](adr/ADR-003-chunking-strategy.md) | Semantic chunking | Quality vs. ingest speed |
| [ADR-004](adr/ADR-004-embedding-model.md) | all-MiniLM-L6-v2 default | Cost vs. quality |
| [ADR-005](adr/ADR-005-reranking-approach.md) | Cross-encoder reranking | Precision vs. latency |

---

## Retrieval Pipeline Deep Dive

```
User Query
    │
    ├─ Query Rewriting (optional)
    │   └─ LLM generates 3 query variants
    │
    ├─ HyDE (optional)
    │   └─ LLM generates hypothetical answer → embed
    │
    ├─ Dense Vector Search (Top-20)
    │   └─ query embedding → FAISS cosine similarity
    │
    ├─ BM25 Keyword Search (Top-20)
    │   └─ tokenize → TF-IDF scoring
    │
    ├─ RRF Fusion
    │   └─ merge ranked lists → deduplicate → re-rank
    │
    ├─ MMR Diversification
    │   └─ balance relevance vs. diversity
    │
    ├─ Cross-Encoder Reranking (Top-20 → Top-5)
    │   └─ joint (query, doc) scoring
    │
    └─ LLM Generation with Citations
        └─ grounded answer + source references
```

---

## Evaluation Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| Retrieval | Precision@K | Fraction of retrieved docs that are relevant |
| Retrieval | Recall@K | Fraction of relevant docs that are retrieved |
| Retrieval | MRR | Mean Reciprocal Rank |
| Generation | Faithfulness | Answer grounded in retrieved context (LLM judge) |
| Generation | Answer Relevancy | Answer addresses the question (LLM judge) |
| Generation | Context Precision | Retrieved chunks are useful (LLM judge) |
| Generation | Context Recall | Context contains all needed information (LLM judge) |

---







after the execution :

1. Environment SetupCreated .env file with your OpenRouter API key:.......
Installed all required Python packages from requirements.txt
Resolved Windows Long Path issues during installation

3. Document Ingestion ✅

Successfully ingested 2 documents:
rag_overview.md → 23 chunks
retrieval_strategies.md → 33 chunks
Total: 56 chunks indexed using FAISS vector store
Built BM25 hybrid search index

3. Query Test ✅
Ran the query: "What is RAG?"Results:
Answer Generated: "Retrieval-Augmented Generation (RAG) is a technique that enhances large language models (LLMs) by grounding their responses in retrieved external knowledge."
Sources Retrieved: 5 relevant chunks from the documents
Retrieval Strategy: HybridRetriever with query rewriting
Latency: ~33 seconds (includes model download time)
Key Features Working:
✅ Semantic chunking
✅ FAISS vector indexing
✅ Hybrid search (dense + BM25)
✅ MMR (Maximal Marginal Relevance) reranking
✅ Cross-encoder reranking
✅ Query rewriting with LLM
✅ OpenRouter API integrationThe system is now ready to use! You can ask more questions with:
bash
python query.py "Your question here"
Or use interactive mode:
bash
python query.py --interactive
