import React, { useState } from 'react';
import { Search, Loader2, Clock, AlignLeft, ShieldAlert } from 'lucide-react';
import { askPipeline, type QueryResponse } from '../api';

export default function QueryPage() {
  const [question, setQuestion] = useState('What are the main causes of the war?');
  const [loading, setLoading] = useState(false);

  const [results, setResults] = useState<{
    rag: QueryResponse | null;
    graphrag: QueryResponse | null;
    agentic: QueryResponse | null;
  }>({
    rag: null,
    graphrag: null,
    agentic: null
  });

  const [errors, setErrors] = useState<{
    rag: string | null;
    graphrag: string | null;
    agentic: string | null;
  }>({ rag: null, graphrag: null, agentic: null });

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setResults({ rag: null, graphrag: null, agentic: null });
    setErrors({ rag: null, graphrag: null, agentic: null });

    const runPipeline = async (type: 'rag' | 'graphrag' | 'agentic') => {
      try {
        const res = await askPipeline(type, question);
        setResults(prev => ({ ...prev, [type]: res }));
      } catch (err: any) {
        setErrors(prev => ({ ...prev, [type]: err.message }));
      }
    };

    await Promise.all([
      runPipeline('rag'),
      runPipeline('graphrag'),
      runPipeline('agentic')
    ]);

    setLoading(false);
  };

  const renderResultColumn = (title: string, type: 'rag' | 'graphrag' | 'agentic') => {
    const res = results[type];
    const err = errors[type];

    return (
      <div className="card pipeline-column">
        <h3 className="pipeline-header">{title}</h3>

        {loading && !res && !err && (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '2rem' }}>
            <Loader2 className="animate-spin text-accent" size={32} color="#3b82f6" />
          </div>
        )}

        {err && (
          <div className="answer-box" style={{ borderColor: 'var(--danger)', color: 'var(--danger)' }}>
            <ShieldAlert size={20} style={{ marginBottom: '0.5rem' }} />
            {err}
          </div>
        )}

        {res && (
          <>
            <div className="metric-row">
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}><Clock size={14}/> Latency</span>
              <span className="badge badge-blue">{(res.latency_ms / 1000).toFixed(2)}s</span>
            </div>

            <div className="metric-row">
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}><AlignLeft size={14}/> Total Tokens</span>
              <span className="badge badge-purple">{res.tokens?.total || 'N/A'}</span>
            </div>

            {res.steps !== undefined && (
              <div className="metric-row">
                <span>Agent Steps</span>
                <span className="badge badge-orange">{res.steps}</span>
              </div>
            )}

            <div className="answer-box">
              {res.answer}
            </div>

            {res.citations && res.citations.length > 0 && (
              <div style={{ marginTop: '1rem' }}>
                <h4 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Citations</h4>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                  {res.citations.map((c, i) => (
                    <span key={i} className="badge badge-green" style={{ fontSize: '0.7rem' }}>{c.substring(0, 20)}...</span>
                  ))}
                </div>
              </div>
            )}

            {res.tools_used && (
              <div style={{ marginTop: '1rem' }}>
                <h4 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Tools Used</h4>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                  {res.tools_used.map((t, i) => (
                    <span key={i} className="badge" style={{ fontSize: '0.7rem', background: 'rgba(255,255,255,0.1)' }}>{t}</span>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    );
  };

  return (
    <div className="animate-fade-in">
      <h1>Query Testing</h1>
      <p>Ask a question to see how each pipeline retrieves information and formulates an answer.</p>

      <form onSubmit={handleQuery} className="card" style={{ marginBottom: '2rem' }}>
        <div className="input-group">
          <label className="input-label">Evaluation Question</label>
          <div style={{ display: 'flex', gap: '1rem' }}>
            <input
              type="text"
              className="input-field"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="E.g. What entities are related to the central thesis?"
            />
            <button type="submit" className="btn btn-primary" disabled={loading} style={{ padding: '0 1.5rem' }}>
              {loading ? <Loader2 className="animate-spin" size={20} /> : <Search size={20} />}
              Run All
            </button>
          </div>
        </div>
      </form>

      <div className="grid-cols-3">
        {renderResultColumn('Traditional RAG', 'rag')}
        {renderResultColumn('GraphRAG', 'graphrag')}
        {renderResultColumn('Agentic GraphRAG', 'agentic')}
      </div>
    </div>
  );
}
