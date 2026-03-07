# RAG System - React Frontend

Modern React-based web interface for the RAG (Retrieval-Augmented Generation) system.

## Features

- 🎨 **Beautiful UI** - Modern, responsive design with gradient theme
- ⚡ **Real-time Queries** - Ask questions and get AI-generated answers
- 📚 **Source Citations** - View all referenced document chunks
- ⚙️ **Advanced Settings** - Configure retrieval strategy, temperature, top-k
- 🔄 **Live Status** - Monitor indexed documents and system health
- 📱 **Responsive** - Works on desktop, tablet, and mobile

## Prerequisites

- Node.js 18+ and npm/yarn
- Python 3.10+ with FastAPI backend running

## Installation

1. **Install dependencies:**
```bash
npm install
```

2. **Start the FastAPI backend** (in the parent directory):
```bash
cd ..
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

3. **Start the React development server:**
```bash
npm run dev
```

The app will be available at: http://localhost:3000

## Usage

1. **Ask a Question**: Type your question in the text area
2. **Configure Settings** (optional):
   - Enable HyDE for hypothetical document embedding
   - Disable reranking for faster results
   - Adjust Top-K to control number of sources
   - Set Temperature for answer creativity
3. **View Results**: Get instant answers with cited sources

## API Configuration

The frontend proxies API requests to the backend at `http://localhost:8000`. You can change this in `vite.config.ts`:

```typescript
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000', // Change port here
      changeOrigin: true,
    }
  }
}
```

## Build for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **TanStack Query (React Query)** - Data fetching and caching
- **Axios** - HTTP client
- **React Markdown** - Render markdown answers
- **Lucide React** - Beautiful icons

## Troubleshooting

### Backend Connection Error
Make sure the FastAPI server is running on port 8000. Check with:
```bash
curl http://localhost:8000/health
```

### CORS Issues
The backend has CORS enabled by default. If you encounter issues, check the CORS settings in `api/app.py`.

## License

MIT
