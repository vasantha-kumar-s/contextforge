import React, { useState, useEffect } from 'react';

export default function KnowledgeBase({ vectorStore }) {
  const [stats, setStats] = useState({ total_documents: 0, total_chunks: 0, filenames: [] });
  const [chunks, setChunks] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [statusMsg, setStatusMsg] = useState('');

  const fetchStats = async () => {
    try {
      const res = await fetch('/api/documents/stats');
      if (res.ok) setStats(await res.json());
    } catch (e) {
      console.error(e);
    }
  };

  const fetchChunks = async () => {
    try {
      const res = await fetch('/api/documents/chunks?limit=50');
      if (res.ok) setChunks(await res.json());
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchStats();
    fetchChunks();
  }, [vectorStore]);

  const handleFileUpload = async (file) => {
    if (!file) return;
    setIsUploading(true);
    setStatusMsg(`Vectorizing document ${file.name}...`);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('chunk_size', 500);
    formData.append('chunk_overlap', 100);

    try {
      const res = await fetch('/api/documents/upload', { method: 'POST', body: formData });
      const data = await res.json();
      setStatusMsg(data.message);
      fetchStats();
      fetchChunks();
    } catch (e) {
      setStatusMsg(`Upload error: ${e.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleLoadSample = async () => {
    setIsUploading(true);
    setStatusMsg("Indexing engineering sample documents into FAISS & Qdrant Cloud...");
    try {
      const res = await fetch('/api/documents/load-sample-data', { method: 'POST' });
      const data = await res.json();
      setStatusMsg(data.message);
      fetchStats();
      fetchChunks();
    } catch (e) {
      setStatusMsg(`Sample load error: ${e.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDeleteDoc = async (filename) => {
    try {
      await fetch(`/api/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' });
      fetchStats();
      fetchChunks();
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="doc-grid">
      <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <h2 style={{ fontSize: '1.05rem', fontWeight: 600 }}>Document Workspace & Indexer</h2>

        <label className="upload-box">
          <input 
            type="file" 
            accept=".pdf,.txt,.md" 
            style={{ display: 'none' }}
            onChange={(e) => e.target.files && handleFileUpload(e.target.files[0])}
          />
          <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>Upload Document (PDF / TXT / MD)</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Recursive Character Chunking (500 chars, 100 overlap)
          </div>
        </label>

        <div style={{ background: 'var(--badge-bg)', padding: '14px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
          <div style={{ fontWeight: 600, fontSize: '0.82rem', marginBottom: '4px' }}>
            Technical Sample Documents Loader
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '10px' }}>
            Populate FAISS & Qdrant Cloud vector stores with standard technical specifications.
          </p>
          <button className="send-btn" onClick={handleLoadSample} disabled={isUploading} style={{ fontSize: '0.78rem', padding: '6px 14px' }}>
            Load Technical Sample Documents
          </button>
        </div>

        {statusMsg && (
          <div style={{ fontSize: '0.8rem', color: 'var(--primary-accent)', fontFamily: 'var(--font-mono)' }}>
            {statusMsg}
          </div>
        )}

        <div style={{ flex: 1, overflowY: 'auto' }}>
          <h3 style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '8px', textTransform: 'uppercase' }}>
            Indexed Documents ({stats.filenames ? stats.filenames.length : 0})
          </h3>
          <table className="data-table">
            <thead>
              <tr>
                <th>Document Name</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {stats.filenames && stats.filenames.map((fname, i) => (
                <tr key={i}>
                  <td>{fname}</td>
                  <td>
                    <button 
                      onClick={() => handleDeleteDoc(fname)}
                      style={{ background: 'transparent', border: 'none', color: '#f43f5e', cursor: 'pointer', fontSize: '0.78rem' }}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column' }}>
        <h2 style={{ fontSize: '1.05rem', fontWeight: 600, marginBottom: '14px' }}>
          Vector Chunk Inspector ({chunks.length})
        </h2>

        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {chunks.map((c, idx) => (
            <div 
              key={idx} 
              style={{
                background: 'var(--badge-bg)',
                border: '1px solid var(--border-color)',
                borderRadius: '6px',
                padding: '10px',
                fontSize: '0.8rem',
                lineHeight: 1.5
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--primary-accent)', fontSize: '0.72rem', fontFamily: 'var(--font-mono)', marginBottom: '4px' }}>
                <span>{c.filename} (Page {c.page_number})</span>
                <span>Chunk #{c.chunk_index} ({c.char_count} chars)</span>
              </div>
              <div style={{ color: 'var(--text-main)' }}>
                {c.text.length > 180 ? c.text.substring(0, 180) + '...' : c.text}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
