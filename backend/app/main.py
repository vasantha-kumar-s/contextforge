import os
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.db import init_db
from app.core.embeddings import EmbeddingService
from app.core.vector_store import QdrantVectorStore
from app.core.session_manager import ChatSessionManager
from app.core.retriever import RAGRetriever
from app.core.ollama_client import OllamaClient
from app.core.groq_client import GroqClient
from app.core.evaluator import RAGEvaluator

from app.routers import documents, chat, models, eval

# Initialize MySQL tables
init_db()

# Core singletons
embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
vector_store = QdrantVectorStore(collection_name="namma_rag_knowledge", dimension=384)
session_manager = ChatSessionManager()
retriever = RAGRetriever(embedding_service=embedding_service, vector_store=vector_store)
ollama_client = OllamaClient(base_url="http://localhost:11434", default_model="qwen3:8b")
groq_client = GroqClient()
evaluator = RAGEvaluator(retriever=retriever, ollama_client=ollama_client)

@asynccontextmanager
async def lifespan(app: FastAPI):
    documents.init_document_router(embedding_service, vector_store)
    chat.init_chat_router(retriever, ollama_client, groq_client, vector_store, session_manager)
    models.init_models_router(ollama_client, groq_client)
    eval.init_eval_router(evaluator)

    stats = vector_store.get_stats()
    if stats.get("total_chunks", 0) == 0:
        print("Qdrant Cloud empty. Auto-ingesting sample technical documentation...")
        documents.load_sample_data()

    yield

app = FastAPI(
    title="Namma-RAG Assistant",
    description="Full-stack RAG System with FastAPI, Qdrant Cloud, MySQL (root:root), Groq, Ollama, and React.",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(models.router)
app.include_router(eval.router)

@app.get("/api/health")
def health():
    return {
        "status": "online",
        "app_name": "Namma-RAG",
        "qdrant_stats": vector_store.get_stats(),
        "ollama_url": ollama_client.base_url,
        "groq_configured": bool(groq_client.api_key),
        "mysql_database": "namma_rag"
    }

# Mount static frontend build
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist"))
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="static_assets")
    
    @app.get("/")
    async def serve_root():
        return FileResponse(
            os.path.join(frontend_dist, "index.html"),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        if full_path.startswith("api"):
            return None
        file_path = os.path.join(frontend_dist, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(
            os.path.join(frontend_dist, "index.html"),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
