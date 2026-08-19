import os
import time
import asyncio
import httpx
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

class GroqClient:
    """
    Async HTTP client for Groq Cloud API (https://api.groq.com/openai/v1/chat/completions).
    Features automatic retry with exponential backoff for rate limits/timeouts and fallback model execution.
    """

    AVAILABLE_MODELS = [
        {
            "name": "llama-3.3-70b-versatile",
            "provider": "Groq",
            "context_window": 128000,
            "description": "70B parameter flagship model for grounded document RAG Q&A, complex reasoning, and strict citation accuracy.",
            "reason": "Primary RAG production engine due to top-tier benchmark accuracy on grounded QA and zero local RAM footprint."
        },
        {
            "name": "llama-3.1-8b-instant",
            "provider": "Groq",
            "context_window": 128000,
            "description": "8B parameter ultra-fast model (~150ms latency) for synthetic dataset generation and critique filtering.",
            "reason": "Ideal for high-speed batch evaluation, critique agents (Groundedness/Relevance/Standalone), and instant fallback."
        },
        {
            "name": "deepseek-r1-distill-llama-70b",
            "provider": "Groq",
            "context_window": 128000,
            "description": "70B distilled reasoning model specialized in chain-of-thought multi-hop RAG analysis.",
            "reason": "Enables step-by-step logical reasoning over complex multi-document contexts and technical specifications."
        },
        {
            "name": "mixtral-8x7b-32768",
            "provider": "Groq",
            "context_window": 32768,
            "description": "High-throughput Mixture-of-Experts (MoE) 8x7B architecture model.",
            "reason": "High token generation speed for synthesizing large document context blocks."
        },
        {
            "name": "gemma2-9b-it",
            "provider": "Groq",
            "context_window": 8192,
            "description": "Google's 9B parameter compact instruction-tuned model.",
            "reason": "Provides fast instruction-following lightweight responses for chat turns."
        }
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"

    async def _post_with_retry(
        self,
        payload: Dict[str, Any],
        timeout: float = 45.0,
        max_retries: int = 3
    ) -> Tuple[str, float]:
        if not self.api_key:
            return "Groq API key not configured.", 0.0

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        start_time = time.perf_counter()
        last_error = ""

        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    res = await client.post(self.endpoint, headers=headers, json=payload)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                    if res.status_code == 200:
                        data = res.json()
                        ans = data["choices"][0]["message"]["content"].strip()
                        return ans, elapsed_ms
                    elif res.status_code == 429:
                        last_error = f"Groq Rate Limit (429): {res.text}"
                        if attempt < max_retries:
                            await asyncio.sleep(1.5 * (attempt + 1))
                            continue
                    elif res.status_code >= 500:
                        last_error = f"Groq Server Error ({res.status_code}): {res.text}"
                        if attempt < max_retries:
                            await asyncio.sleep(1.0 * (attempt + 1))
                            continue
                    else:
                        return f"Groq API Error ({res.status_code}): {res.text}", elapsed_ms
            except (httpx.TimeoutException, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                last_error = f"Groq API Timeout: {str(e)}"
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return f"Groq API Exception: {str(e)}", elapsed_ms

        # Fallback to llama-3.1-8b-instant if primary model repeatedly failed
        if payload.get("model") != "llama-3.1-8b-instant":
            fallback_payload = dict(payload)
            fallback_payload["model"] = "llama-3.1-8b-instant"
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    res = await client.post(self.endpoint, headers=headers, json=fallback_payload)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    if res.status_code == 200:
                        data = res.json()
                        return data["choices"][0]["message"]["content"].strip(), elapsed_ms
            except Exception:
                pass

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return f"Failed to connect to Groq API after retries: {last_error}", elapsed_ms

    async def generate_grounded_answer(
        self,
        query: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
        model: str = "llama-3.3-70b-versatile"
    ) -> Tuple[str, float]:
        system_message = {
            "role": "system",
            "content": (
                "You are an expert engineering document assistant. "
                "Answer the user's question accurately based ONLY on the provided document context below. "
                "If the context does not contain enough information to answer the question, state: "
                "'The provided context does not contain information to answer this question.' "
                "Always cite document sources when referencing facts."
            )
        }

        messages = [system_message]

        if history:
            for turn in history[-6:]:
                messages.append({"role": turn["role"], "content": turn["content"]})

        user_content = f"DOCUMENT CONTEXT:\n{context}\n\nUSER QUESTION:\n{query}"
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2
        }

        return await self._post_with_retry(payload, timeout=45.0)

    async def generate_direct_answer(
        self,
        query: str,
        model: str = "llama-3.3-70b-versatile"
    ) -> Tuple[str, float]:
        messages = [
            {"role": "user", "content": query}
        ]

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7
        }

        return await self._post_with_retry(payload, timeout=45.0)
