

export default function ArchitecturePage() {
  return (
    <div className="animate-fade-in">
      <h1>System Architecture</h1>
      <p>Agentic GraphRAG system diagram and component breakdown.</p>
      <div className="card">
        <h2 style={{ color: 'var(--accent-primary)' }}>Tech Stack</h2>
        <ul>
          <li><strong>LLM Inference:</strong> Ollama (Local)</li>
          <li><strong>Vector Store:</strong> Qdrant (Local)</li>
          <li><strong>Graph DB:</strong> TigerGraph Savanna</li>
          <li><strong>Backend:</strong> FastAPI + Python</li>
          <li><strong>Frontend:</strong> React + Vite</li>
        </ul>
      </div>
    </div>
  );
}
