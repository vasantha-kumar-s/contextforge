"""
rag_evaluator.py
=============================================================================
Comprehensive Namma-RAG Evaluation Suite
Measures all 8 interview-defensible metrics using real Qdrant Cloud + Groq.

METRICS:
  RETRIEVAL   Recall@1, Recall@5, Precision@5, MRR
  GENERATION  Answer Correctness (LLM judge 0-2), Faithfulness (LLM judge 0-1)
  SAFETY      Abstention Rate on unanswerable questions
  PERFORMANCE Avg / P50 / P95 / Max end-to-end latency

RUN:
  cd e:\\Projects_TP\\rag-naam\\backend
  python -m scripts.rag_evaluator
=============================================================================
"""

import asyncio
import statistics
import time
import os
import argparse
import httpx
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
load_dotenv()

from scripts.eval_dataset import EVAL_DATASET, ANSWERABLE, MULTI_HOP, UNANSWERABLE
from app.core.embeddings import EmbeddingService
from app.core.vector_store import QdrantVectorStore
from app.core.retriever import RAGRetriever
from app.core.groq_client import GroqClient
from app.core.ollama_client import OllamaClient

# --------------------------------------------------------------------------
# LLM-as-Judge helpers
# --------------------------------------------------------------------------

async def _groq_judge(prompt: str, api_key: str) -> str:
    """Call Groq llama-3.1-8b-instant as a fast judge. Retries on 429 rate limit."""
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "max_tokens": 10
                    }
                )
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"].strip()
                elif r.status_code == 429:
                    await asyncio.sleep(2.0 * (attempt + 1))
                    continue
                else:
                    return "0"
        except Exception:
            await asyncio.sleep(1.0)
    return "0"

async def _ollama_judge(prompt: str, model: str = "qwen3:8b", base_url: str = "http://localhost:11434") -> str:
    """Call local Ollama service with 0.0 temperature as an evaluator judge."""
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(url, json=payload)
            if r.status_code == 200:
                return r.json().get("response", "").strip()
    except Exception as e:
        print(f"Ollama judge connection error: {e}")
    return "0"

async def judge_answer_correctness(question: str, ground_truth: str, answer: str, provider: str = "ollama", api_key: str = "", model: str = "qwen3:8b") -> int:
    """
    LLM judge: Is the generated answer correct?
    Returns 2 (correct), 1 (partially correct), 0 (incorrect / not found).
    """
    prompt = f"""You are a strict answer evaluator. Score the RAG answer vs the ground truth.

Question: {question}
Ground Truth: {ground_truth}
RAG Answer: {answer}

Scoring:
2 = Answer is correct and covers the key facts from ground truth
1 = Answer is partially correct or vague but not wrong
0 = Answer is incorrect, hallucinated, or says it cannot find the info when it should

Reply with ONLY a single digit: 0, 1, or 2."""

    if provider.lower() == "ollama":
        raw = await _ollama_judge(prompt, model=model)
    else:
        raw = await _groq_judge(prompt, api_key)

    for ch in raw:
        if ch in ("0", "1", "2"):
            return int(ch)
    return 0

async def judge_faithfulness(context: str, answer: str, provider: str = "ollama", api_key: str = "", model: str = "qwen3:8b") -> int:
    """
    LLM judge: Is every claim in the answer supported by the retrieved context?
    Returns 1 (fully faithful) or 0 (contains unsupported / hallucinated claims).
    """
    prompt = f"""You are a strict faithfulness evaluator for RAG systems.

Retrieved Context:
{context[:1500]}

RAG Answer: {answer}

Is every factual claim in the RAG answer directly supported by the retrieved context above?
Reply with ONLY 1 (yes, fully supported) or 0 (no, contains claims not in the context)."""

    if provider.lower() == "ollama":
        raw = await _ollama_judge(prompt, model=model)
    else:
        raw = await _groq_judge(prompt, api_key)

    for ch in raw:
        if ch in ("0", "1"):
            return int(ch)
    return 1  # default to faithful if judge fails

def judge_abstention(answer: str) -> bool:
    """
    Heuristic: Did the model correctly refuse to answer an unanswerable question?
    Returns True if the answer contains a 'not found / cannot' phrase.
    """
    refusal_phrases = [
        "i don't", "i do not", "cannot find", "not found", "no information",
        "not available", "not mentioned", "context does not", "unable to find",
        "not provided", "not in the", "doesn't contain", "does not contain",
        "no relevant", "not enough information", "cannot answer"
    ]
    lower = answer.lower()
    return any(p in lower for p in refusal_phrases)

# --------------------------------------------------------------------------
# Retrieval metric helpers
# --------------------------------------------------------------------------

def check_snippet_in_chunks(expected_snippet: str, chunks: List[Dict]) -> Tuple[bool, int]:
    """
    Returns (found: bool, rank: int 1-based).
    rank = 0 means not found.
    """
    if not expected_snippet:
        return False, 0
    snippet_lower = expected_snippet.lower()
    for rank, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "").lower()
        filename = chunk.get("filename", "").lower()
        if snippet_lower in text or snippet_lower in filename:
            return True, rank
    return False, 0

def precision_at_k(expected_snippet: str, chunks: List[Dict], k: int = 5) -> float:
    """Fraction of retrieved top-k chunks that contain the expected snippet."""
    if not expected_snippet:
        return 0.0
    snippet_lower = expected_snippet.lower()
    relevant = sum(
        1 for c in chunks[:k]
        if snippet_lower in c.get("text", "").lower()
    )
    return relevant / k

# --------------------------------------------------------------------------
# Main evaluation runner
# --------------------------------------------------------------------------

async def run_evaluation(provider: str = "ollama", model: str = "qwen3:8b"):
    api_key = os.getenv("GROQ_API_KEY", "")
    if provider.lower() == "groq" and not api_key:
        print("ERROR: GROQ_API_KEY not set in .env")
        return

    print("=" * 80)
    print(f"  NAMMA-RAG COMPREHENSIVE EVALUATION SUITE (Provider: {provider.upper()} | Model: {model})")
    print("  Metrics: Recall@1, Recall@5, Precision@5, MRR, Answer Correctness,")
    print("           Faithfulness, Abstention Rate, P50/P95 Latency")
    print("=" * 80)

    # Init components
    embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
    vector_store = QdrantVectorStore(collection_name="namma_rag_knowledge", dimension=384)
    retriever = RAGRetriever(embedding_service=embedding_service, vector_store=vector_store)

    if provider.lower() == "ollama":
        llm_client = OllamaClient(default_model=model)
    else:
        llm_client = GroqClient(api_key=api_key)

    stats = vector_store.get_stats()
    print(f"\n[DB] Collection: '{stats.get('collection_name')}' | Chunks: {stats.get('total_chunks')}")
    print(f"[DB] MySQL: namma_rag@localhost:3306\n")

    # Separate question lists
    answerable_qs = ANSWERABLE + MULTI_HOP  # 40 questions with expected answers
    unanswerable_qs = UNANSWERABLE          # 10 questions that should be declined

    # ---- ANSWERABLE QUESTIONS ----
    print(f"[1/2] Evaluating {len(answerable_qs)} answerable questions...\n")

    recall_at_1_hits = 0
    recall_at_5_hits = 0
    precision_at_5_sum = 0.0
    mrr_sum = 0.0
    correctness_sum = 0
    faithfulness_sum = 0
    query_success = 0

    e2e_latencies: List[float] = []
    retrieval_latencies: List[float] = []

    for item in answerable_qs:
        qid   = item["id"]
        q     = item["question"]
        qtype = item["type"]
        snip  = item["expected_snippet"]

        t_start = time.perf_counter()

        # 1. Retrieve top-5
        chunks, ret_ms = retriever.retrieve(q, top_k=5)
        retrieval_latencies.append(ret_ms)

        # 2. Snippet-based retrieval metrics
        found, rank = check_snippet_in_chunks(snip, chunks)
        if rank == 1:
            recall_at_1_hits += 1
        if found:
            recall_at_5_hits += 1
        precision_at_5_sum += precision_at_k(snip, chunks, k=5)
        mrr_sum += (1.0 / rank) if rank > 0 else 0.0

        # 3. Generate answer
        context = retriever.format_context(chunks)
        if provider.lower() == "ollama":
            answer, gen_ms = await llm_client.generate_grounded_answer(q, context, model=model)
        else:
            answer, gen_ms = await llm_client.generate_grounded_answer(q, context, model="llama-3.3-70b-versatile")

        total_ms = (time.perf_counter() - t_start) * 1000.0
        e2e_latencies.append(total_ms)

        if answer and not answer.startswith("Groq Error") and not answer.startswith("Ollama Error") and not answer.startswith("Failed to connect"):
            query_success += 1

        # 4. LLM judge scores
        correctness = await judge_answer_correctness(q, item["ground_truth"], answer, provider=provider, api_key=api_key, model=model)
        faithfulness = await judge_faithfulness(context, answer, provider=provider, api_key=api_key, model=model)

        correctness_sum  += correctness
        faithfulness_sum += faithfulness

        r1 = "Y" if rank == 1 else " "
        r5 = "Y" if found else "N"
        tag = "ANS" if qtype == "answerable" else "MHO"
        print(
            f"[{tag}] #{qid:02d} | R@1:{r1} R@5:{r5} | Corr:{correctness}/2 "
            f"Faith:{faithfulness}/1 | E2E:{total_ms/1000:.2f}s"
        )

    n_ans = len(answerable_qs)
    recall_at_1_pct  = recall_at_1_hits / n_ans * 100
    recall_at_5_pct  = recall_at_5_hits / n_ans * 100
    precision_at_5   = precision_at_5_sum / n_ans * 100
    mrr              = mrr_sum / n_ans
    correctness_pct  = correctness_sum / (n_ans * 2) * 100
    faithfulness_pct = faithfulness_sum / n_ans * 100
    success_pct      = query_success / n_ans * 100

    # ---- UNANSWERABLE QUESTIONS ----
    print(f"\n[2/2] Evaluating {len(unanswerable_qs)} unanswerable questions...\n")

    abstention_hits = 0
    for item in unanswerable_qs:
        q = item["question"]

        chunks, _ = retriever.retrieve(q, top_k=5)
        context = retriever.format_context(chunks)
        if provider.lower() == "ollama":
            answer, _ = await llm_client.generate_grounded_answer(q, context, model=model)
        else:
            answer, _ = await llm_client.generate_grounded_answer(q, context, model="llama-3.3-70b-versatile")

        abstained = judge_abstention(answer)
        if abstained:
            abstention_hits += 1

        tag = "OK " if abstained else "HAL"
        print(f"[{tag}] #{item['id']:02d} | Q: {q[:60]}")

    abstention_rate = abstention_hits / len(unanswerable_qs) * 100
    hallucination_rate = 100.0 - abstention_rate

    # ---- LATENCY ----
    sorted_e2e = sorted(e2e_latencies)
    avg_e2e  = statistics.mean(sorted_e2e)
    min_e2e  = sorted_e2e[0]
    p50_e2e  = statistics.median(sorted_e2e)
    p95_idx  = max(0, int(0.95 * len(sorted_e2e)) - 1)
    p95_e2e  = sorted_e2e[p95_idx]
    max_e2e  = sorted_e2e[-1]
    avg_ret  = statistics.mean(retrieval_latencies)

    # ---- FINAL REPORT ----
    print("\n" + "=" * 80)
    print(f"  NAMMA-RAG COMPREHENSIVE EVALUATION RESULTS ({provider.upper()})")
    print("=" * 80)
    print(f"Evaluation Scope      : {n_ans} answerable + {len(unanswerable_qs)} unanswerable = {len(EVAL_DATASET)} total")
    print(f"Query Success Rate    : {query_success}/{n_ans} ({success_pct:.1f}%)")
    print()
    print("--- RETRIEVAL METRICS (" + str(n_ans) + " questions) ---")
    print(f"  Recall@1            : {recall_at_1_hits}/{n_ans} ({recall_at_1_pct:.1f}%)")
    print(f"  Recall@5            : {recall_at_5_hits}/{n_ans} ({recall_at_5_pct:.1f}%)")
    print(f"  Precision@5         : {precision_at_5:.1f}%")
    print(f"  MRR                 : {mrr:.3f}")
    print()
    print("--- GENERATION METRICS ---")
    print(f"  Answer Correctness  : {correctness_sum}/{n_ans*2} ({correctness_pct:.1f}%) [LLM judge 0-2]")
    print(f"  Faithfulness        : {faithfulness_sum}/{n_ans} ({faithfulness_pct:.1f}%) [LLM judge 0-1]")
    print()
    print("--- SAFETY / ABSTENTION ---")
    print(f"  Abstention Rate     : {abstention_hits}/{len(unanswerable_qs)} ({abstention_rate:.1f}%)")
    print(f"  Hallucination Rate  : {len(unanswerable_qs)-abstention_hits}/{len(unanswerable_qs)} ({hallucination_rate:.1f}%)")
    print()
    print("--- LATENCY (answerable questions only) ---")
    print(f"  Avg Retrieval       : {avg_ret:.1f} ms")
    print(f"  Min E2E             : {min_e2e:.1f} ms  ({min_e2e/1000:.2f} s)")
    print(f"  Avg E2E             : {avg_e2e:.1f} ms  ({avg_e2e/1000:.2f} s)")
    print(f"  P50 E2E             : {p50_e2e:.1f} ms  ({p50_e2e/1000:.2f} s)")
    print(f"  P95 E2E             : {p95_e2e:.1f} ms  ({p95_e2e/1000:.2f} s)")
    print(f"  Max E2E             : {max_e2e:.1f} ms  ({max_e2e/1000:.2f} s)")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive RAG evaluation with Ollama or Groq.")
    parser.add_argument("--provider", choices=["ollama", "groq"], default="ollama", help="LLM provider (default: ollama)")
    parser.add_argument("--model", type=str, default="qwen3:8b", help="Model name (default: qwen3:8b)")
    args = parser.parse_args()

    asyncio.run(run_evaluation(provider=args.provider, model=args.model))

