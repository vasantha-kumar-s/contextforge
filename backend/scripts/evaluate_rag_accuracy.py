import time
import asyncio
import statistics
import argparse
from typing import List, Dict, Any

from app.core.embeddings import EmbeddingService
from app.core.vector_store import QdrantVectorStore
from app.core.retriever import RAGRetriever
from app.core.groq_client import GroqClient
from app.core.ollama_client import OllamaClient

# Benchmark Suite matching exact document filenames in Qdrant Cloud
BENCHMARK_DATASET = [
    {
        "id": 1,
        "question": "What is Namma-RAG architecture?",
        "expected_filename": ["PROJECT_DOCUMENTATION", "rag_architecture_spec"],
        "expected_keywords": ["PyMuPDF", "SentenceTransformers", "Qdrant", "MySQL", "Groq", "Ollama"]
    },
    {
        "id": 2,
        "question": "Where are vector chunks and embeddings stored in Namma-RAG?",
        "expected_filename": ["PROJECT_DOCUMENTATION", "vector_database_guide"],
        "expected_keywords": ["Qdrant", "AWS EU-Central", "384-d", "Cosine"]
    },
    {
        "id": 3,
        "question": "How is relational session data saved in MySQL?",
        "expected_filename": ["PROJECT_DOCUMENTATION", "vector_database_guide"],
        "expected_keywords": ["MySQL", "3306", "namma_rag", "session"]
    },
    {
        "id": 4,
        "question": "What LLM providers are supported in Namma-RAG?",
        "expected_filename": ["PROJECT_DOCUMENTATION"],
        "expected_keywords": ["Groq", "Ollama"]
    },
    {
        "id": 5,
        "question": "Who is the author of the published research paper and what is their affiliation?",
        "expected_filename": ["Published_Paper_6-2-9-272.pdf"],
        "expected_keywords": ["Sadik Lakhani", "P P Savani University"]
    },
    {
        "id": 6,
        "question": "What is the DOI of the published research paper?",
        "expected_filename": ["Published_Paper_6-2-9-272.pdf"],
        "expected_keywords": ["10.33545/27076636.2025.v6.i2b.127"]
    },
    {
        "id": 7,
        "question": "What image resolution and preprocessing steps are described in the published paper?",
        "expected_filename": ["Published_Paper_6-2-9-272.pdf"],
        "expected_keywords": ["256", "normalized", "resolution"]
    },
    {
        "id": 8,
        "question": "How many citations and reads does the published research paper have?",
        "expected_filename": ["Published_Paper_6-2-9-272.pdf"],
        "expected_keywords": ["0 citations", "38 reads"]
    },
    {
        "id": 9,
        "question": "What plant leaf types are analyzed in the research paper dataset?",
        "expected_filename": ["Published_Paper_6-2-9-272.pdf"],
        "expected_keywords": ["maize", "potato", "leaves"]
    },
    {
        "id": 10,
        "question": "What references are cited in the published research paper?",
        "expected_filename": ["Published_Paper_6-2-9-272.pdf"],
        "expected_keywords": ["Singh", "Sharma", "Kriti", "Pathak"]
    },
    {
        "id": 11,
        "question": "What system latency telemetry is displayed in the Namma-RAG interface?",
        "expected_filename": ["PROJECT_DOCUMENTATION"],
        "expected_keywords": ["Retrieval", "Gen", "Total"]
    },
    {
        "id": 12,
        "question": "What chunk size and overlap parameters are used for text splitting?",
        "expected_filename": ["PROJECT_DOCUMENTATION"],
        "expected_keywords": ["800", "150", "chunk"]
    },
    {
        "id": 13,
        "question": "What database tables are initialized in MySQL for Namma-RAG?",
        "expected_filename": ["PROJECT_DOCUMENTATION"],
        "expected_keywords": ["documents", "chat_sessions", "chat_messages"]
    },
    {
        "id": 14,
        "question": "What embedding model is used for dense vector generation in Namma-RAG?",
        "expected_filename": ["PROJECT_DOCUMENTATION"],
        "expected_keywords": ["SentenceTransformers", "all-MiniLM-L6-v2", "384"]
    },
    {
        "id": 15,
        "question": "What themes are supported in the Namma-RAG user interface?",
        "expected_filename": ["PROJECT_DOCUMENTATION"],
        "expected_keywords": ["Dark", "Light", "theme"]
    }
]

def score_answer_correctness(answer: str, expected_keywords: List[str]) -> int:
    if not answer or "error" in answer.lower():
        return 0

    lower_ans = answer.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in lower_ans)
    ratio = matches / len(expected_keywords)

    if ratio >= 0.4:
        return 2
    elif ratio > 0.0:
        return 1
    else:
        return 0

async def run_accuracy_evaluation(provider: str = "ollama", model: str = "qwen3:8b"):
    print("=" * 80)
    print(f"  NAMMA-RAG ACCURACY EVALUATION BENCHMARK (Provider: {provider.upper()} | Model: {model})  ")
    print("=" * 80)

    print("\n[1/3] Initializing Components & Connecting to Qdrant Cloud...")
    embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
    vector_store = QdrantVectorStore(collection_name="namma_rag_knowledge", dimension=384)
    retriever = RAGRetriever(embedding_service=embedding_service, vector_store=vector_store)

    if provider.lower() == "ollama":
        llm_client = OllamaClient(default_model=model)
    else:
        llm_client = GroqClient()

    v_stats = vector_store.get_stats()
    print(f"[+] Qdrant Cloud Collection: '{v_stats.get('collection_name')}' | Total Chunks: {v_stats.get('total_chunks')}")

    print(f"\n[2/3] Evaluating RAG Pipeline across {len(BENCHMARK_DATASET)} Ground-Truth Test Cases...\n")

    recall_at_5_hits = 0
    total_correctness_points = 0
    max_possible_points = len(BENCHMARK_DATASET) * 2

    retrieval_latencies: List[float] = []
    generation_latencies: List[float] = []
    e2e_latencies: List[float] = []
    successful_queries = 0

    for item in BENCHMARK_DATASET:
        qid = item["id"]
        q = item["question"]
        expected_fns = item["expected_filename"]
        expected_kws = item["expected_keywords"]

        t_start = time.perf_counter()
        
        # 1. Retrieval Step
        chunks, ret_ms = retriever.retrieve(q, top_k=5)
        retrieval_latencies.append(ret_ms)

        # Check Recall@5
        retrieved_filenames = [c.get("filename", "") for c in chunks]
        is_recalled = any(
            any(target_fn.lower() in ret_fn.lower() for target_fn in expected_fns)
            for ret_fn in retrieved_filenames
        )
        if is_recalled:
            recall_at_5_hits += 1

        # 2. Generation Step
        context = retriever.format_context(chunks)
        if provider.lower() == "ollama":
            answer, gen_ms = await llm_client.generate_grounded_answer(q, context, model=model)
        else:
            answer, gen_ms = await llm_client.generate_grounded_answer(q, context, model=model or "llama-3.3-70b-versatile")
        generation_latencies.append(gen_ms)

        total_ms = (time.perf_counter() - t_start) * 1000.0
        e2e_latencies.append(total_ms)

        # 3. Answer Correctness Evaluation (0, 1, 2)
        score = score_answer_correctness(answer, expected_kws)
        total_correctness_points += score

        if answer and not answer.startswith("Groq Error") and not answer.startswith("Ollama Error") and not answer.startswith("Failed to connect"):
            successful_queries += 1

        status_str = "RECALLED" if is_recalled else "MISSED"
        score_str = "2/2 (Correct)" if score == 2 else ("1/2 (Partial)" if score == 1 else "0/2 (Incorrect)")
        print(f"Test #{qid:02d} | Recall@5: [{status_str}] | Answer Score: {score_str} | E2E: {total_ms/1000.0:.2f}s")

    # Calculate Metrics
    recall_at_5_pct = (recall_at_5_hits / len(BENCHMARK_DATASET)) * 100.0
    answer_correctness_pct = (total_correctness_points / max_possible_points) * 100.0
    query_success_pct = (successful_queries / len(BENCHMARK_DATASET)) * 100.0

    avg_ret = statistics.mean(retrieval_latencies)
    avg_gen = statistics.mean(generation_latencies)
    avg_e2e = statistics.mean(e2e_latencies)

    sorted_e2e = sorted(e2e_latencies)
    min_e2e = sorted_e2e[0]
    max_e2e = sorted_e2e[-1]
    p50_e2e = statistics.median(sorted_e2e)
    p95_idx = max(0, int(0.95 * len(sorted_e2e)) - 1)
    p95_e2e = sorted_e2e[p95_idx]

    print("\n" + "=" * 80)
    print(f"  NAMMA-RAG EMPIRICAL BENCHMARK EVALUATION SUMMARY ({provider.upper()})  ")
    print("=" * 80)
    print(f"Total Test Questions     : {len(BENCHMARK_DATASET)}")
    print(f"Query Success Rate       : {successful_queries}/{len(BENCHMARK_DATASET)} ({query_success_pct:.1f}%)")
    print(f"Retrieval Recall@5       : {recall_at_5_hits}/{len(BENCHMARK_DATASET)} ({recall_at_5_pct:.1f}%)")
    print(f"Answer Correctness Score : {total_correctness_points}/{max_possible_points} ({answer_correctness_pct:.1f}%)")
    print("-" * 80)
    print(f"Avg Retrieval Latency    : {avg_ret:.2f} ms")
    print(f"Avg LLM Generation Time  : {avg_gen:.2f} ms ({avg_gen/1000.0:.2f} s)")
    print(f"Min End-to-End Latency   : {min_e2e:.2f} ms ({min_e2e/1000.0:.2f} s)")
    print(f"Avg End-to-End Latency   : {avg_e2e:.2f} ms ({avg_e2e/1000.0:.2f} s)")
    print(f"P50 End-to-End Latency   : {p50_e2e:.2f} ms ({p50_e2e/1000.0:.2f} s)")
    print(f"P95 End-to-End Latency   : {p95_e2e:.2f} ms ({p95_e2e/1000.0:.2f} s)")
    print(f"Max End-to-End Latency   : {max_e2e:.2f} ms ({max_e2e/1000.0:.2f} s)")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG accuracy with Ollama or Groq.")
    parser.add_argument("--provider", choices=["ollama", "groq"], default="ollama", help="LLM Provider to evaluate (default: ollama)")
    parser.add_argument("--model", type=str, default="qwen3:8b", help="Model name (default: qwen3:8b)")
    args = parser.parse_args()

    asyncio.run(run_accuracy_evaluation(provider=args.provider, model=args.model))

