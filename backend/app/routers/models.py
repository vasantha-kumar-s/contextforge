from fastapi import APIRouter
from app.core.ollama_client import OllamaClient
from app.core.groq_client import GroqClient

router = APIRouter(prefix="/api/models", tags=["Models & Providers"])

_ollama_client = None
_groq_client = None

def init_models_router(ollama_client: OllamaClient, groq_client: GroqClient):
    global _ollama_client, _groq_client
    _ollama_client = ollama_client
    _groq_client = groq_client

@router.get("")
async def get_models():
    """Returns available LLM models grouped by provider (Ollama and Groq)."""
    ollama_models = []
    if _ollama_client:
        ollama_models = await _ollama_client.list_models()

    groq_models = GroqClient.AVAILABLE_MODELS if _groq_client and _groq_client.api_key else []

    return {
        "providers": [
            {
                "id": "ollama",
                "name": "Ollama (Local Runtime)",
                "status": "online" if ollama_models else "offline",
                "models": ollama_models
            },
            {
                "id": "groq",
                "name": "Groq Cloud API",
                "status": "online" if groq_models else "offline",
                "models": groq_models
            }
        ]
    }
