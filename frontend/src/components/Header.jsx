import React from 'react';

export default function Header({
  activeTab,
  setActiveTab,
  theme,
  setTheme,
  provider,
  setProvider,
  model,
  setModel,
  availableModels,
  ragMode,
  setRagMode
}) {
  const toggleTheme = () => {
    const nextTheme = theme === 'dark' ? 'light' : 'dark';
    setTheme(nextTheme);
    document.documentElement.setAttribute('data-theme', nextTheme);
  };

  const safeModelsList = Array.isArray(availableModels) ? availableModels : [];
  const currentProviderObj = safeModelsList.find(p => p && p.id === provider);
  const currentProviderModels = currentProviderObj && Array.isArray(currentProviderObj.models) ? currentProviderObj.models : [];

  return (
    <header className="navbar glass-panel">
      <div className="brand-section">
        <h1 className="brand-title">Namma-RAG</h1>
      </div>

      <nav className="nav-tabs">
        <button 
          className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => setActiveTab('chat')}
        >
          Chat
        </button>
        <button 
          className={`tab-btn ${activeTab === 'docs' ? 'active' : ''}`}
          onClick={() => setActiveTab('docs')}
        >
          Files
        </button>
      </nav>

      <div className="controls-bar">
        {/* Pipeline Toggle RAG vs LLM */}
        <div className="mode-toggle-group">
          <button 
            className={`mode-btn ${ragMode ? 'active' : ''}`}
            onClick={() => setRagMode(true)}
            title="Grounded RAG Mode"
          >
            RAG
          </button>
          <button 
            className={`mode-btn ${!ragMode ? 'active' : ''}`}
            onClick={() => setRagMode(false)}
            title="Direct LLM Mode"
          >
            LLM
          </button>
        </div>

        {/* LLM Provider */}
        <select 
          className="select-control"
          value={provider}
          onChange={(e) => {
            const p = e.target.value;
            setProvider(p);
            const pObj = safeModelsList.find(m => m && m.id === p);
            const pModels = pObj && Array.isArray(pObj.models) ? pObj.models : [];
            if (pModels.length > 0) setModel(pModels[0].name);
          }}
          title="LLM Provider"
        >
          <option value="groq">Groq Cloud</option>
          <option value="ollama">Ollama Local</option>
        </select>

        {/* Model Picker */}
        <select 
          className="select-control"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          title="LLM Model"
        >
          {currentProviderModels.length > 0 ? (
            currentProviderModels.map(m => (
              <option key={m.name} value={m.name}>
                {m.name} {m.size_gb ? `(${m.size_gb}GB)` : ''}
              </option>
            ))
          ) : (
            <option value={model}>{model}</option>
          )}
        </select>

        {/* Theme Toggle */}
        <button className="theme-toggle-btn" onClick={toggleTheme}>
          {theme === 'dark' ? 'Light' : 'Dark'}
        </button>
      </div>
    </header>
  );
}
