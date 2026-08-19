import httpx
import time
import json
from typing import List, Dict, Any, AsyncGenerator, Tuple, Optional

class OllamaClient:
    """
    Async client for locally hosted Ollama service (http://localhost:11434).
    """

    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "qwen3:8b"):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model

    async def list_models(self) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    return [
                        {
                            "name": m.get("name"),
                            "size_gb": round(m.get("size", 0) / (1024 ** 3), 2),
                            "modified_at": m.get("modified_at")
                        }
                        for m in models
                    ]
        except Exception as e:
            print(f"Ollama list_models error: {e}")
        return []

    def _format_error(self, response_text: str) -> str:
        if "requires more system memory" in response_text:
            return (
                "Ollama System RAM Limit: Model requires more RAM than currently available. "
                "Tip: Switch provider to 'Groq Cloud API' in the top header for instant sub-second response with zero local RAM usage."
            )
        return f"Ollama Error: {response_text}"

    async def generate_grounded_answer(
        self,
        query: str,
        context: str,
        model: Optional[str] = None
    ) -> Tuple[str, float]:
        selected_model = model or self.default_model

        system_prompt = (
            "You are an expert technical document assistant. "
            "Answer the user's question accurately based ONLY on the provided document context below. "
            "If the context does not contain enough information, state that clearly. "
            "Always cite document source filenames and page numbers when referencing facts."
        )

        prompt = f"DOCUMENT CONTEXT:\n{context}\n\nUSER QUESTION:\n{query}\n\nGROUNDED ANSWER:"

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": selected_model,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "").strip(), elapsed_ms
                else:
                    return self._format_error(response.text), elapsed_ms
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return f"Failed to connect to local Ollama service: {str(e)}", elapsed_ms

    async def generate_direct_answer(
        self,
        query: str,
        model: Optional[str] = None
    ) -> Tuple[str, float]:
        selected_model = model or self.default_model
        prompt = f"Answer the following question directly to the best of your knowledge:\n{query}"

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": selected_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7
            }
        }

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "").strip(), elapsed_ms
                else:
                    return self._format_error(response.text), elapsed_ms
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return f"Failed to connect to local Ollama service: {str(e)}", elapsed_ms
