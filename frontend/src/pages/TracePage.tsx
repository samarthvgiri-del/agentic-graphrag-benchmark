import { useState } from 'react';
import { Network, Database, CheckCircle, FileText, Search, XCircle } from 'lucide-react';

const MOCK_TRACE = [
  { step: 1, type: 'search', tool: 'entity_search', status: 'success', detail: 'Found 3 entities related to "AI scaling laws".' },
  { step: 2, type: 'graph', tool: 'graph_traversal', status: 'success', detail: 'Traversed from [Scaling Laws] to [Compute Power] and [Data Constraints].' },
  { step: 3, type: 'evaluate', tool: 'evidence_evaluation', status: 'warning', detail: 'Missing evidence on how data constraints affect the final outcome.' },
  { step: 4, type: 'retrieve', tool: 'document_retrieval', status: 'success', detail: 'Retrieved 2 documents resolving data constraint ambiguity.' },
  { step: 5, type: 'evaluate', tool: 'evidence_evaluation', status: 'success', detail: 'Sufficient evidence reached. Proceeding to answer formulation.' },
  { step: 6, type: 'stop', tool: 'STOP', status: 'info', detail: 'Generated final grounded answer.' },
];

export default function TracePage() {
  const [trace] = useState(MOCK_TRACE);

  const getIcon = (tool: string) => {
    switch (tool) {
      case 'entity_search': return <Search size={16} color="var(--accent-primary)" />;
      case 'graph_traversal': return <Network size={16} color="var(--accent-secondary)" />;
      case 'evidence_evaluation': return <CheckCircle size={16} color="var(--warning)" />;
      case 'document_retrieval': return <FileText size={16} color="var(--success)" />;
      case 'vector_search': return <Database size={16} color="#38bdf8" />;
      case 'STOP': return <XCircle size={16} color="var(--danger)" />;
      default: return <Search size={16} />;
    }
  };

  return (
    <div className="animate-fade-in">
      <h1>Agent Trace Viewer</h1>
      <p>Analyze the step-by-step decision making process of the Agentic Orchestrator.</p>

      <div className="card" style={{ maxWidth: '800px', margin: '0 auto', padding: '2rem' }}>
        <h2 style={{ marginBottom: '2rem' }}>Investigation Trace #Q-1042</h2>

        <div style={{ marginLeft: '1rem' }}>
          {trace.map((item, index) => (
            <div key={index} className="trace-step">
              <div className="trace-icon">
                {getIcon(item.tool)}
              </div>
              <div className="trace-content">
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ fontWeight: 600, color: 'var(--text-main)' }}>Step {item.step}: {item.tool}</span>
                  <span className={`badge ${item.status === 'success' ? 'badge-green' : item.status === 'warning' ? 'badge-orange' : item.status === 'danger' ? 'badge-purple' : 'badge-blue'}`}>
                    {item.status.toUpperCase()}
                  </span>
                </div>
                <p style={{ margin: 0, fontSize: '0.9rem' }}>{item.detail}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
