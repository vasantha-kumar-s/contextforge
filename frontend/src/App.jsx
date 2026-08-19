import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import ChatWorkbench from './components/ChatWorkbench';
import KnowledgeBase from './components/KnowledgeBase';

export default function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [theme, setTheme] = useState('dark');
  const [provider, setProvider] = useState('groq');
  const [model, setModel] = useState('llama-3.3-70b-versatile');
  const [ragMode, setRagMode] = useState(true);
  const [availableModels, setAvailableModels] = useState([]);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const fetchProvidersAndModels = async () => {
    try {
      const res = await fetch('/api/models');
      if (res.ok) {
        const data = await res.json();
        setAvailableModels(data.providers || []);
      }
    } catch (err) {
      console.error("Failed to fetch models", err);
    }
  };

  useEffect(() => {
    fetchProvidersAndModels();
  }, []);

  return (
    <div className="app-container">
      <Header 
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        theme={theme}
        setTheme={setTheme}
        provider={provider}
        setProvider={setProvider}
        model={model}
        setModel={setModel}
        availableModels={availableModels}
        ragMode={ragMode}
        setRagMode={setRagMode}
      />

      <main className="main-content">
        {activeTab === 'chat' && (
          <ChatWorkbench 
            provider={provider} 
            model={model} 
            ragMode={ragMode}
          />
        )}
        {activeTab === 'docs' && (
          <KnowledgeBase />
        )}
      </main>
    </div>
  );
}
