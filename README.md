# RAG Chatbot

A high-performance Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **Qdrant**, **SentenceTransformers**, and **Groq / Ollama** LLM synthesis. Features document parsing, semantic chunking, vector similarity retrieval, and a **React + Vite** chat interface.

---

## ✨ Features

- **Document Ingestion**: PDF parsing and semantic chunking via PyMuPDF.
- **Vector Search**: Similarity search powered by Qdrant vector database and SentenceTransformers.
- **LLM Synthesis**: High-speed inference using Groq API (`llama-3`) or local Ollama models.
- **Session Management**: Persistent multi-turn conversation tracking and RAG retrieval evaluation.
- **Modern Interface**: Clean React 18 chat UI with live reference context citation.

---

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, PyMuPDF, SentenceTransformers, SQLAlchemy
- **Vector DB & LLM**: Qdrant, Groq API / Ollama
- **Frontend**: React 18, Vite, Vanilla CSS

---

## 🚀 Quickstart

### 1. Clone & Environment Setup
```bash
git clone https://github.com/vasantha-kumar-s/rag-chatbot.git
cd rag-chatbot
```

### 2. Backend Setup
```bash
cd backend

# Create environment configuration
cp .env.example .env
# Update .env with your GROQ_API_KEY and QDRANT parameters

python -m venv venv
# Windows: .\venv\Scripts\Activate.ps1 | Linux/macOS: source venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 📄 License
MIT
