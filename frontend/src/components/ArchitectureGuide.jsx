import React from 'react';

export default function ArchitectureGuide() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', height: '100%', overflowY: 'auto' }}>
      <div className="glass-panel" style={{ padding: '20px' }}>
        <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '6px' }}>
          Architecture Specification & Technical Rationale
        </h2>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          Detailed technical design trade-offs and interview answer reference guide for senior technical interviewers.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
        <div className="glass-panel" style={{ padding: '16px' }}>
          <h3 style={{ fontSize: '0.9rem', color: 'var(--primary-accent)', marginBottom: '8px' }}>
            1. Dual LLM Provider Engine (Ollama & Groq)
          </h3>
          <p style={{ fontSize: '0.82rem', lineHeight: 1.55 }}>
            <strong>Rationale:</strong> Ollama provides 100% local, privacy-preserving model execution on system hardware. Groq Cloud API provides ultra-low latency sub-second inference using Llama 3.3 70B and Mixtral models. The system supports runtime switching between both providers.
          </p>
        </div>

        <div className="glass-panel" style={{ padding: '16px' }}>
          <h3 style={{ fontSize: '0.9rem', color: 'var(--primary-accent)', marginBottom: '8px' }}>
            2. Multi-Store Vector Database (FAISS & Qdrant Cloud)
          </h3>
          <p style={{ fontSize: '0.82rem', lineHeight: 1.55 }}>
            <strong>Rationale:</strong> FAISS IndexFlatIP provides zero-dependency, sub-millisecond local in-memory cosine similarity search. Qdrant Cloud handles distributed cloud vector indexing over HTTP. A unified vector store manager decouples vector storage from application logic.
          </p>
        </div>

        <div className="glass-panel" style={{ padding: '16px' }}>
          <h3 style={{ fontSize: '0.9rem', color: 'var(--primary-accent)', marginBottom: '8px' }}>
            3. Chat Session & Conversational Context Management
          </h3>
          <p style={{ fontSize: '0.82rem', lineHeight: 1.55 }}>
            <strong>Rationale:</strong> Chat history is partitioned by persistent Session IDs. Recent conversation turns are appended to the RAG context block before sending generation prompts to the LLM, enabling multi-turn conversational memory.
          </p>
        </div>

        <div className="glass-panel" style={{ padding: '16px' }}>
          <h3 style={{ fontSize: '0.9rem', color: 'var(--primary-accent)', marginBottom: '8px' }}>
            4. Dense Embedding & Recursive Chunking
          </h3>
          <p style={{ fontSize: '0.82rem', lineHeight: 1.55 }}>
            <strong>Rationale:</strong> 384-dimensional dense vectors (`all-MiniLM-L6-v2`) provide optimal balance between memory footprint and semantic recall. Recursive character text splitting (500 chars, 100 overlap) preserves sentence boundaries.
          </p>
        </div>
      </div>
    </div>
  );
}
