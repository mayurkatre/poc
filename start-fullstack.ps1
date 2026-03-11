# RAG System - Full Stack Startup Script (PowerShell)
# Starts both FastAPI backend and React frontend

Write-Host "🚀 Starting RAG System - Full Stack" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

# Check if Python dependencies are installed
Write-Host "`n📦 Checking Python dependencies..." -ForegroundColor Yellow
try {
    $null = python -c "import fastapi, uvicorn" 2>$null
    Write-Host "✅ Python dependencies OK" -ForegroundColor Green
} catch {
    Write-Host "❌ Missing Python dependencies. Run: pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}

# Check if Node.js is installed
Write-Host "`n📦 Checking Node.js..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version
    Write-Host "✅ Node.js installed: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found. Please install Node.js 18+" -ForegroundColor Red
    exit 1
}

# Check if frontend dependencies are installed
Write-Host "`n📦 Checking frontend dependencies..." -ForegroundColor Yellow
if (Test-Path "frontend\node_modules") {
    Write-Host "✅ Frontend dependencies OK" -ForegroundColor Green
} else {
    Write-Host "⚠️  Frontend dependencies missing. Installing..." -ForegroundColor Yellow
    Set-Location frontend
    npm install
    Set-Location ..
}

# Check if documents are ingested
Write-Host "`n📚 Checking if documents are ingested..." -ForegroundColor Yellow
if (Test-Path ".cache\faiss_index.index") {
    Write-Host "✅ Documents already ingested" -ForegroundColor Green
} else {
    Write-Host "⚠️  No index found. Ingesting documents..." -ForegroundColor Yellow
    python ingest.py ./documents
}

Write-Host "`n🎯 Starting servers..." -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan

# Start FastAPI backend in a new window
Write-Host "`n📡 Starting FastAPI Backend on http://localhost:8000" -ForegroundColor Cyan
$backendJob = Start-Job -ScriptBlock {
    Set-Location $using:PSScriptRoot
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
}

# Wait a moment for backend to start
Start-Sleep -Seconds 3

# Start React frontend in a new window
Write-Host "🎨 Starting React Frontend on http://localhost:3000" -ForegroundColor Cyan
$frontendJob = Start-Job -ScriptBlock {
    Set-Location $using:PSScriptRoot\frontend
    npm run dev
}

Write-Host "`n✅ Servers started!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "`n📊 Backend API:  http://localhost:8000" -ForegroundColor White
Write-Host "📚 API Docs:     http://localhost:8000/docs" -ForegroundColor White
Write-Host "🎨 Frontend:     http://localhost:3000" -ForegroundColor White
Write-Host "`n⏹️  Press Ctrl+C to stop all servers" -ForegroundColor Yellow
Write-Host "=====================================`n" -ForegroundColor Cyan

# Monitor jobs
try {
    while ($true) {
        Start-Sleep -Seconds 2
        
        # Check if jobs are still running
        $backendState = Get-Job -Id $backendJob.Id | Select-Object -ExpandProperty State
        $frontendState = Get-Job -Id $frontendJob.Id | Select-Object -ExpandProperty State
        
        if ($backendState -eq "Failed" -or $frontendState -eq "Failed") {
            Write-Host "`n❌ One of the servers crashed!" -ForegroundColor Red
            break
        }
        
        # Output any job output
        $backendOutput = Receive-Job -Id $backendJob.Id -Keep
        if ($backendOutput) {
            $backendOutput | ForEach-Object { Write-Host $_ -ForegroundColor Gray }
        }
        
        $frontendOutput = Receive-Job -Id $frontendJob.Id -Keep
        if ($frontendOutput) {
            $frontendOutput | ForEach-Object { Write-Host $_ -ForegroundColor Gray }
        }
    }
}
finally {
    # Cleanup on exit
    Write-Host "`n🛑 Stopping servers..." -ForegroundColor Yellow
    Stop-Job -Id $backendJob.Id
    Stop-Job -Id $frontendJob.Id
    Remove-Job -Id $backendJob.Id
    Remove-Job -Id $frontendJob.Id
    Write-Host "✅ Servers stopped" -ForegroundColor Green
}
