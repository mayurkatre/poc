# Starting RAG System - Full Stack Application

This guide explains how to run the complete RAG system with FastAPI backend and React frontend.

## Quick Start

### Option 1: Manual Startup (Recommended for Development)

**Terminal 1 - Start FastAPI Backend:**
```bash
# From project root directory
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Backend will be available at: http://localhost:8000
API docs at: http://localhost:8000/docs

**Terminal 2 - Start React Frontend:**
```bash
cd frontend
npm run dev
```

Frontend will be available at: http://localhost:3000

### Option 2: Using Startup Script

**Windows PowerShell:**
```powershell
.\start-fullstack.ps1
```

**Linux/Mac:**
```bash
./start-fullstack.sh
```

## Prerequisites

Before starting, ensure you have:

1. ✅ **Python 3.10+** installed
2. ✅ **Node.js 18+** and npm installed
3. ✅ **Dependencies installed**: 
   ```bash
   pip install -r requirements.txt
   cd frontend && npm install
   ```
4. ✅ **Documents ingested**: Run `python ingest.py ./documents`

## Architecture

```
┌─────────────────┐         ┌─────────────────┐
│  React Frontend │ ──────▶ │  FastAPI Backend│
│   (Port 3000)   │  /api   │   (Port 8000)   │
└─────────────────┘         └─────────────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │  RAG Pipeline   │
                            │  - FAISS Index  │
                            │  - BM25 Search  │
                            │  - LLM (OpenRouter)│
                            └─────────────────┘
```

## Directory Structure

```
rag-poc-updated/
├── api/                    # FastAPI backend
│   └── app.py             # Main API application
├── frontend/              # React frontend
│   ├── src/
│   │   ├── App.tsx       # Main React component
│   │   ├── index.css     # Styles
│   │   └── main.tsx      # Entry point
│   ├── package.json
│   └── vite.config.ts
├── ingestion/             # Document ingestion
├── retrieval/             # Retrieval strategies
├── generation/            # LLM generation
├── reranking/            # Cross-encoder reranking
├── config/               # Configuration
├── documents/            # Your documents
├── ingest.py            # CLI for ingestion
└── query.py             # CLI for querying
```

## Testing the System

1. **Check Backend Health:**
   - Visit: http://localhost:8000/health
   - Should show: `{"status": "healthy", "indexed_chunks": 56}`

2. **Check Frontend:**
   - Visit: http://localhost:3000
   - You should see the RAG query interface

3. **Test a Query:**
   - Type: "What is RAG?"
   - Click "Ask RAG"
   - View answer with sources

## Configuration

### Backend Settings

Edit `config/settings.yaml` to configure:
- Embedding model
- Vector store provider (FAISS/Chroma)
- Retrieval strategy
- LLM model (via OpenRouter)

### Frontend Settings

Edit `frontend/vite.config.ts` to change:
- Frontend port (default: 3000)
- Backend proxy URL (default: http://localhost:8000)

## Production Deployment

### Build Frontend for Production:
```bash
cd frontend
npm run build
```

### Serve with FastAPI:
Copy `dist/` contents to a static files directory and serve with FastAPI:

```python
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="dist", html=True), name="static")
```

## Troubleshooting

### Backend won't start
- Check if port 8000 is available
- Ensure all Python dependencies are installed
- Verify `.env` file exists with API key

### Frontend shows blank page
- Check browser console for errors
- Ensure backend is running on port 8000
- Check network tab for failed API requests

### CORS errors
- Backend has CORS enabled by default
- Check `app.add_middleware(CORSMiddleware, ...)` in `api/app.py`

### API connection refused
- Frontend proxies `/api` to `http://localhost:8000`
- Make sure backend is running before starting frontend

## Performance Tips

1. **Use GPU for embeddings** (if available):
   - Install `torch` with CUDA support
   - Set `embedding.device` in config

2. **Enable caching**:
   - Embeddings are cached by default in `.cache/embeddings`
   - FAISS index is cached in `.cache/faiss_index`

3. **Optimize chunk size**:
   - Smaller chunks = more precise retrieval
   - Larger chunks = more context
   - Default: 512 chars with 64 overlap

## Support

For issues or questions:
- Check API docs: http://localhost:8000/docs
- Review logs in terminal windows
- Inspect browser DevTools Network tab
