#!/bin/bash
# RAG System - Full Stack Startup Script (Linux/Mac)
# Starts both FastAPI backend and React frontend

echo "🚀 Starting RAG System - Full Stack"
echo "====================================="

# Check if Python dependencies are installed
echo -e "\n📦 Checking Python dependencies..."
if python -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "✅ Python dependencies OK"
else
    echo "❌ Missing Python dependencies. Run: pip install -r requirements.txt"
    exit 1
fi

# Check if Node.js is installed
echo -e "\n📦 Checking Node.js..."
if command -v node &> /dev/null; then
    node_version=$(node --version)
    echo "✅ Node.js installed: $node_version"
else
    echo "❌ Node.js not found. Please install Node.js 18+"
    exit 1
fi

# Check if frontend dependencies are installed
echo -e "\n📦 Checking frontend dependencies..."
if [ -d "frontend/node_modules" ]; then
    echo "✅ Frontend dependencies OK"
else
    echo "⚠️  Frontend dependencies missing. Installing..."
    cd frontend
    npm install
    cd ..
fi

# Check if documents are ingested
echo -e "\n📚 Checking if documents are ingested..."
if [ -f ".cache/faiss_index.index" ]; then
    echo "✅ Documents already ingested"
else
    echo "⚠️  No index found. Ingesting documents..."
    python ingest.py ./documents
fi

echo -e "\n🎯 Starting servers..."
echo "====================================="

# Start FastAPI backend in background
echo -e "\n📡 Starting FastAPI Backend on http://localhost:8000"
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start React frontend in background
echo -e "\n🎨 Starting React Frontend on http://localhost:3000"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo -e "\n✅ Servers started!"
echo "====================================="
echo -e "\n📊 Backend API:  http://localhost:8000"
echo -e "📚 API Docs:     http://localhost:8000/docs"
echo -e "🎨 Frontend:     http://localhost:3000"
echo -e "\n⏹️  Press Ctrl+C to stop all servers"
echo "=====================================\n"

# Cleanup function
cleanup() {
    echo -e "\n🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID 2>/dev/null
    wait $FRONTEND_PID 2>/dev/null
    echo "✅ Servers stopped"
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT SIGTERM

# Wait for processes
wait
