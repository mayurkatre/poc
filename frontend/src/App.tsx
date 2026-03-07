import React, { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Send, Settings, BookOpen, Activity, AlertCircle, CheckCircle } from 'lucide-react';
import './App.css';

// API base URL
const API_BASE = '/api';

// Types
interface Source {
  document: string;
  chunk_id: string;
  text: string;
  chunk_index: number;
  page_number?: number;
}

interface QueryResponse {
  answer: string;
  sources: Source[];
  query: string;
  retrieval_strategy: string;
  latency_ms: number;
}

interface HealthResponse {
  status: string;
  version: string;
  indexed_chunks: number;
}

// API calls
const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

const fetchHealth = async (): Promise<HealthResponse> => {
  const response = await api.get('/health');
  return response.data;
};

const submitQuery = async (data: {
  question: string;
  top_k?: number;
  use_hyde?: boolean;
  no_rerank?: boolean;
  temperature?: number;
}): Promise<QueryResponse> => {
  const response = await api.post('/query', data);
  return response.data;
};

// Main App Component
function App() {
  // State
  const [question, setQuestion] = useState('');
  const [useHyde, setUseHyde] = useState(false);
  const [noRerank, setNoRerank] = useState(false);
  const [topK, setTopK] = useState(5);
  const [temperature, setTemperature] = useState(0.0);

  // Queries
  const { data: health, isLoading: healthLoading, error: healthError } = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    refetchInterval: 30000, // Check every 30 seconds
  });

  // Mutations
  const queryMutation = useMutation({
    mutationFn: submitQuery,
  });

  // Handlers
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    queryMutation.mutate({
      question,
      top_k: topK,
      use_hyde: useHyde,
      no_rerank: noRerank,
      temperature,
    });
  };

  const handleClear = () => {
    setQuestion('');
    queryMutation.reset();
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <h1>🤖 RAG Query System</h1>
        <p>Retrieval-Augmented Generation with OpenRouter</p>
        {health && !healthLoading && (
          <div className="stats-badge">
            <Activity size={16} />
            <span>{health.indexed_chunks} chunks indexed</span>
            {health.status === 'healthy' ? (
              <CheckCircle size={16} />
            ) : (
              <AlertCircle size={16} />
            )}
          </div>
        )}
      </header>

      {/* Main Content */}
      <div className="main-content">
        {/* Query Section */}
        <section className="query-section">
          <h2 className="section-title">
            <Send size={24} />
            Ask a Question
          </h2>
          
          <form onSubmit={handleSubmit} className="query-form">
            <div className="textarea-group">
              <textarea
                className="question-textarea"
                placeholder="What is RAG? Explain how it works..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                disabled={queryMutation.isPending}
              />
            </div>

            <button 
              type="submit" 
              className="ask-button"
              disabled={queryMutation.isPending || !question.trim()}
            >
              {queryMutation.isPending ? (
                <>
                  <div className="spinner-small" />
                  Processing...
                </>
              ) : (
                <>
                  <Send size={20} />
                  Ask RAG
                </>
              )}
            </button>
          </form>

          {/* Settings */}
          <div className="settings-section" style={{ marginTop: '2rem' }}>
            <h2 className="section-title">
              <Settings size={24} />
              Configuration
            </h2>

            <div className="settings-grid">
              <div className="setting-item">
                <label className="setting-toggle">
                  <input
                    type="checkbox"
                    checked={useHyde}
                    onChange={(e) => setUseHyde(e.target.checked)}
                  />
                  <span className="setting-label">Use HyDE Retrieval</span>
                </label>
              </div>

              <div className="setting-item">
                <label className="setting-toggle">
                  <input
                    type="checkbox"
                    checked={noRerank}
                    onChange={(e) => setNoRerank(e.target.checked)}
                  />
                  <span className="setting-label">Disable Reranking</span>
                </label>
              </div>

              <div className="setting-item">
                <label className="setting-label">
                  Top-K Results: {topK}
                </label>
                <input
                  type="range"
                  min="1"
                  max="20"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  className="slider-input"
                />
              </div>

              <div className="setting-item">
                <label className="setting-label">
                  Temperature: {temperature.toFixed(1)}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="slider-input"
                />
              </div>
            </div>

            <button onClick={handleClear} className="clear-button">
              Clear Results
            </button>
          </div>
        </section>

        {/* Answer Section */}
        <section className="answer-section">
          <h2 className="section-title">
            <BookOpen size={24} />
            Answer & Sources
          </h2>

          {queryMutation.isPending && (
            <div className="loading-spinner">
              <div className="spinner" />
              <p>Searching and generating answer...</p>
            </div>
          )}

          {queryMutation.error && (
            <div className="error-message">
              <AlertCircle size={24} />
              <p>Error: {(queryMutation.error as Error).message}</p>
            </div>
          )}

          {queryMutation.data && (
            <div className="answer-content">
              {/* Metadata */}
              <div className="metadata-bar">
                <div className="metric">
                  <span className="metric-label">Strategy</span>
                  <span className="metric-value">{queryMutation.data.retrieval_strategy}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Sources</span>
                  <span className="metric-value">{queryMutation.data.sources.length}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Latency</span>
                  <span className="metric-value">{queryMutation.data.latency_ms.toFixed(0)}ms</span>
                </div>
              </div>

              {/* Answer */}
              <div className="answer-text">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {queryMutation.data.answer}
                </ReactMarkdown>
              </div>

              {/* Sources */}
              {queryMutation.data.sources.length > 0 && (
                <div className="sources-list">
                  <h3 className="sources-title">
                    <BookOpen size={20} />
                    Sources ({queryMutation.data.sources.length})
                  </h3>

                  {queryMutation.data.sources.map((source, index) => (
                    <div key={source.chunk_id} className="source-item">
                      <div className="source-header">
                        <span className="source-document">{source.document}</span>
                        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                          <span className="source-chunk-id">{source.chunk_id.substring(0, 8)}</span>
                          {source.page_number && (
                            <span className="source-page">Page {source.page_number}</span>
                          )}
                        </div>
                      </div>
                      <div className="source-text">{source.text}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {!queryMutation.isPending && !queryMutation.data && !queryMutation.error && (
            <div className="empty-state">
              <div className="empty-state-icon">📚</div>
              <p>Ask a question to get started!</p>
              <p style={{ fontSize: '0.9rem', marginTop: '0.5rem' }}>
                The AI will search through your documents and provide an answer with citations.
              </p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

export default App;
