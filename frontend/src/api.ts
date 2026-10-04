export const API_BASE_URL = 'http://127.0.0.1:8000';

export interface QueryResponse {
  answer: string;
  latency_ms: number;
  tokens: {
    input: number;
    output: number;
    total: number;
  };
  citations: string[];
  steps?: number;
  tools_used?: string[];
  trace?: any[];
  stop_reason?: string;
}

export async function askPipeline(pipeline: 'rag' | 'graphrag' | 'agentic', question: string): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE_URL}/api/${pipeline}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  });

  if (!response.ok) {
    throw new Error(`Failed to query ${pipeline} pipeline`);
  }

  return response.json();
}
