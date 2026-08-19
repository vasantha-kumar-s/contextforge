import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/chat", tags=["Chat & Q&A"])

_retriever = None
_ollama_client = None
_groq_client = None
_vector_store = None
_session_manager = None

def init_chat_router(retriever, ollama_client, groq_client, vector_store, session_manager):
    global _retriever, _ollama_client, _groq_client, _vector_store, _session_manager
    _retriever = retriever
    _ollama_client = ollama_client
    _groq_client = groq_client
    _vector_store = vector_store
    _session_manager = session_manager

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default_session"
    provider: Optional[str] = "groq"  # "groq" or "ollama"
    model: Optional[str] = "llama-3.3-70b-versatile"
    top_k: Optional[int] = 4
    rag_mode: Optional[bool] = True

class SessionCreateRequest(BaseModel):
    title: Optional[str] = None

@router.get("/sessions")
def list_sessions():
    if not _session_manager:
        return []
    return _session_manager.list_sessions()

@router.post("/sessions")
def create_session(req: SessionCreateRequest):
    if not _session_manager:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    return _session_manager.create_session(title=req.title)

@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    if not _session_manager:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    session = _session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if not _session_manager:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    _session_manager.delete_session(session_id)
    return {"message": f"Session '{session_id}' deleted"}

@router.post("/query")
async def chat_query(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    session_id = req.session_id or "default_session"
    start_total = time.perf_counter()

    # 1. Retrieve context from Qdrant Cloud via RAGRetriever
    if req.rag_mode:
        chunks, retrieval_ms = _retriever.retrieve(req.query, top_k=req.top_k)
        context = _retriever.format_context(chunks)
        
        citations = [
            {
                "filename": c.get("filename", "Doc"),
                "page_number": c.get("page_number", 1),
                "chunk_index": c.get("chunk_index", 1),
                "similarity_score": round(c.get("similarity_score", 0.0), 4),
                "snippet": c.get("text", "")[:200] + ("..." if len(c.get("text", "")) > 200 else "")
            }
            for c in chunks
        ]
    else:
        retrieval_ms = 0.0
        chunks = []
        citations = []
        context = ""

    # 2. Get conversational history from MySQL
    history = _session_manager.get_history(session_id, limit=6) if _session_manager else []

    # 3. Generate response (Groq API or Ollama)
    provider_type = (req.provider or "groq").lower()
    
    if provider_type == "groq":
        model_name = req.model if req.model in [m["name"] for m in _groq_client.AVAILABLE_MODELS] else "llama-3.3-70b-versatile"
        if req.rag_mode:
            answer, gen_ms = await _groq_client.generate_grounded_answer(req.query, context, history=history, model=model_name)
        else:
            answer, gen_ms = await _groq_client.generate_direct_answer(req.query, model=model_name)
    else:
        model_name = req.model or "qwen3:8b"
        if req.rag_mode:
            answer, gen_ms = await _ollama_client.generate_grounded_answer(req.query, context, model=model_name)
        else:
            answer, gen_ms = await _ollama_client.generate_direct_answer(req.query, model=model_name)

    total_ms = (time.perf_counter() - start_total) * 1000.0

    telemetry = {
        "retrieval_ms": round(retrieval_ms, 2),
        "generation_ms": round(gen_ms, 2),
        "total_ms": round(total_ms, 2),
        "total_sec": round(total_ms / 1000.0, 2)
    }

    # 4. Save to MySQL session history
    if _session_manager:
        _session_manager.add_message(session_id, "user", req.query)
        _session_manager.add_message(session_id, "assistant", answer, citations=citations, telemetry=telemetry)

    return {
        "session_id": session_id,
        "query": req.query,
        "answer": answer,
        "provider": provider_type,
        "model": model_name,
        "vector_store": "qdrant",
        "rag_mode": req.rag_mode,
        "citations": citations,
        "telemetry": telemetry
    }
