import React, { useState, useEffect } from 'react';

function renderFormattedContent(content) {
  if (!content) return null;

  const lines = content.split('\n');
  return (
    <div className="formatted-markdown" style={{ lineHeight: 1.6 }}>
      {lines.map((line, lIdx) => {
        let processedLine = line;

        const isBullet = /^\s*[*•-]\s+(.*)/.test(line);
        const isNumbered = /^\s*(\d+)\.\s+(.*)/.test(line);

        let match;
        if (isBullet) {
          match = line.match(/^\s*[*•-]\s+(.*)/);
          processedLine = match[1];
        } else if (isNumbered) {
          match = line.match(/^\s*(\d+)\.\s+(.*)/);
          processedLine = `${match[1]}. ${match[2]}`;
        }

        const parts = [];
        let pKey = 0;
        const regex = /(\*\*[^*]+\*\*|\[Source \d+:[^\]]+\])/g;
        let lastIndex = 0;
        let m;

        while ((m = regex.exec(processedLine)) !== null) {
          if (m.index > lastIndex) {
            parts.push(processedLine.substring(lastIndex, m.index));
          }
          const token = m[0];
          if (token.startsWith('**') && token.endsWith('**')) {
            parts.push(<strong key={pKey++} style={{ color: 'var(--text-main)', fontWeight: 600 }}>{token.slice(2, -2)}</strong>);
          } else if (token.startsWith('[Source')) {
            parts.push(
              <span key={pKey++} style={{
                background: 'var(--badge-bg)',
                border: '1px solid var(--border-color)',
                borderRadius: '4px',
                padding: '1px 6px',
                fontSize: '0.68rem',
                color: 'var(--text-dim)',
                fontFamily: 'var(--font-mono)',
                marginLeft: '4px',
                display: 'inline-block'
              }}>
                {token.slice(1, -1)}
              </span>
            );
          }
          lastIndex = regex.lastIndex;
        }
        if (lastIndex < processedLine.length) {
          parts.push(processedLine.substring(lastIndex));
        }

        if (isBullet) {
          return (
            <div key={lIdx} style={{ paddingLeft: '16px', margin: '2px 0', display: 'flex', gap: '6px' }}>
              <span style={{ color: 'var(--text-dim)' }}>•</span>
              <div>{parts}</div>
            </div>
          );
        }

        if (isNumbered) {
          return (
            <div key={lIdx} style={{ margin: '6px 0 2px 0' }}>
              {parts}
            </div>
          );
        }

        return (
          <div key={lIdx} style={{ minHeight: line.trim() === '' ? '6px' : 'auto', margin: '2px 0' }}>
            {parts}
          </div>
        );
      })}
    </div>
  );
}

export default function ChatWorkbench({ provider, model, ragMode }) {
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState('default_session');
  const [messages, setMessages] = useState([]);
  const [inputQuery, setInputQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedCitation, setSelectedCitation] = useState(null);
  const [docStats, setDocStats] = useState({ total_documents: 0, total_chunks: 0, filenames: [] });

  const fetchSessions = async () => {
    try {
      const res = await fetch('/api/chat/sessions');
      if (res.ok) {
        const data = await res.json();
        setSessions(data || []);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const loadSessionMessages = async (sessionId) => {
    try {
      const res = await fetch(`/api/chat/sessions/${sessionId}`);
      if (res.ok) {
        const data = await res.json();
        setMessages(data.messages || []);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const fetchDocStats = async () => {
    try {
      const res = await fetch('/api/documents/stats');
      if (res.ok) {
        const data = await res.json();
        setDocStats(data);
      }
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchSessions();
    loadSessionMessages(activeSessionId);
    fetchDocStats();
  }, [activeSessionId]);

  const handleCreateSession = async () => {
    try {
      const res = await fetch('/api/chat/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: `Session ${sessions.length + 1}` })
      });
      if (res.ok) {
        const newSession = await res.json();
        setActiveSessionId(newSession.session_id);
        fetchSessions();
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleDeleteSession = async (sessionId, e) => {
    e.stopPropagation();
    try {
      await fetch(`/api/chat/sessions/${sessionId}`, { method: 'DELETE' });
      fetchSessions();
      if (activeSessionId === sessionId) {
        setActiveSessionId('default_session');
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleSend = async (queryText = inputQuery) => {
    const query = queryText.trim();
    if (!query || isLoading) return;

    const userMessage = { role: 'user', content: query };
    setMessages(prev => [...prev, userMessage]);
    setInputQuery('');
    setIsLoading(true);

    try {
      const res = await fetch('/api/chat/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          session_id: activeSessionId,
          provider: provider || 'groq',
          model: model || 'llama-3.3-70b-versatile',
          top_k: 4,
          rag_mode: ragMode
        })
      });

      const data = await res.json();
      
      const assistantMessage = {
        role: 'assistant',
        content: data.answer || 'No response generated.',
        citations: data.citations || [],
        telemetry: data.telemetry || null
      };

      setMessages(prev => [...prev, assistantMessage]);
      fetchSessions();
    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: `Connection error: ${err.message}`,
          citations: [],
          telemetry: null
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      {/* Sessions Sidebar */}
      <div className="session-sidebar glass-panel">
        <button className="new-session-btn" onClick={handleCreateSession}>
          + New Chat
        </button>

        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginTop: '6px' }}>
          Sessions ({sessions.length})
        </div>

        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {(sessions || []).map((s) => (
            <div 
              key={s.session_id}
              className={`session-item ${activeSessionId === s.session_id ? 'active' : ''}`}
              onClick={() => setActiveSessionId(s.session_id)}
              style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
            >
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{s.title}</span>
              {s.session_id !== 'default_session' && (
                <button 
                  onClick={(e) => handleDeleteSession(s.session_id, e)}
                  style={{ background: 'transparent', border: 'none', color: 'var(--text-dim)', cursor: 'pointer', fontSize: '0.75rem' }}
                >
                  ✕
                </button>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Main Chat Stream */}
      <div className="chat-main glass-panel">
        <div className="messages-list">
          {(!messages || messages.length === 0) && (
            <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', textAlign: 'center', marginTop: '40px' }}>
              No messages in this session. Ask a question to query document context.
            </div>
          )}

          {(messages || []).map((msg, idx) => (
            <div key={idx} className={`message-bubble ${msg.role}`}>
              {renderFormattedContent(msg.content)}

              {/* Citations */}
              {msg.citations && msg.citations.length > 0 && (
                <div style={{ marginTop: '8px', paddingTop: '6px', borderTop: '1px solid var(--border-color)' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: '2px' }}>
                    Sources ({msg.citations.length}):
                  </div>
                  <div>
                    {msg.citations.map((c, cIdx) => (
                      <span 
                        key={cIdx} 
                        className="citation-tag"
                        onClick={() => setSelectedCitation(c)}
                      >
                        {c.filename} (p.{c.page_number}) • {c.similarity_score}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Message Telemetry Footer */}
              {msg.telemetry && (
                <div className="msg-telemetry-footer">
                  <span>Retrieval: {msg.telemetry.retrieval_ms} ms</span>
                  <span>•</span>
                  <span>Gen: {msg.telemetry.generation_ms} ms</span>
                  <span>•</span>
                  <span>Total: {msg.telemetry.total_sec} s</span>
                </div>
              )}
            </div>
          ))}

          {isLoading && (
            <div className="message-bubble assistant" style={{ color: 'var(--text-muted)' }}>
              Generating response...
            </div>
          )}
        </div>

        <div className="chat-input-area">
          <input
            type="text"
            className="chat-input"
            placeholder={ragMode ? "Ask a question..." : "Ask direct LLM query..."}
            value={inputQuery}
            onChange={(e) => setInputQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          />
          <button className="send-btn" onClick={() => handleSend()} disabled={isLoading}>
            Send
          </button>
        </div>
      </div>

      {/* Right Sidebar: Uploaded Docs & Index Stats */}
      <div className="docs-sidebar glass-panel">
        <div style={{ fontSize: '0.72rem', fontWeight: 600, textTransform: 'uppercase', color: 'var(--text-muted)' }}>
          Indexed Documents ({docStats.total_documents || (docStats.filenames ? docStats.filenames.length : 0)})
        </div>

        <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
          Qdrant Cloud Chunks: {docStats.total_chunks || 0}
        </div>

        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '4px' }}>
          {(docStats.filenames || []).length > 0 ? (
            docStats.filenames.map((fname, idx) => (
              <div key={idx} className="doc-list-item">
                <div className="doc-list-name">{fname}</div>
                <div className="doc-list-meta">Indexed in Qdrant & MySQL</div>
              </div>
            ))
          ) : (
            <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
              No documents loaded.
            </div>
          )}
        </div>
      </div>

      {/* Citation Popover Modal */}
      {selectedCitation && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(0,0,0,0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
          padding: '20px'
        }}>
          <div className="glass-panel" style={{ maxWidth: '540px', width: '100%', padding: '18px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <h3 style={{ fontSize: '0.9rem', color: 'var(--text-main)' }}>
                {selectedCitation.filename} (Page {selectedCitation.page_number})
              </h3>
              <button 
                onClick={() => setSelectedCitation(null)}
                style={{ background: 'transparent', border: 'none', color: 'var(--text-main)', cursor: 'pointer', fontSize: '1rem' }}
              >
                ✕
              </button>
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginBottom: '10px', fontFamily: 'var(--font-mono)' }}>
              Similarity Score: {selectedCitation.similarity_score} • Chunk #{selectedCitation.chunk_index}
            </div>
            <div style={{
              background: 'var(--code-bg)',
              padding: '12px',
              borderRadius: '6px',
              fontSize: '0.82rem',
              lineHeight: 1.5,
              border: '1px solid var(--border-color)',
              maxHeight: '260px',
              overflowY: 'auto',
              color: 'var(--text-muted)'
            }}>
              {selectedCitation.snippet}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
