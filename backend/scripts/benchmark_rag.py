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

async def run_benchmark(provider: str = "ollama", model: str = "qwen3:8b"):
    print("=" * 70)
    print(f"  NAMMA-RAG SYSTEM BENCHMARK ({provider.upper()} - {model})  ")
    print("=" * 70)

    # 1. Initialize components
    print("\n[1/4] Initializing Namma-RAG Core Components...")
    t0 = time.perf_counter()
    embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
    vector_store = QdrantVectorStore(collection_name="namma_rag_knowledge", dimension=384)
    retriever = RAGRetriever(embedding_service=embedding_service, vector_store=vector_store)

    if provider.lower() == "ollama":
        llm_client = OllamaClient(default_model=model)
    else:
        llm_client = GroqClient()

    init_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[+] Components Initialized in {init_ms:.2f} ms")

    # Verify vector store contents
    v_stats = vector_store.get_stats()
    print(f"[+] Qdrant Cloud Collection: '{v_stats.get('collection_name')}' | Total Chunks: {v_stats.get('total_chunks')}")

    # Benchmark Test Queries
    queries = [
        "What is Namma-RAG architecture?",
        "Where are vector chunks and embeddings stored in Namma-RAG?",
        "How is relational session data saved in MySQL?",
        "What LLM providers are supported in Namma-RAG?",
        "Explain vector similarity search using Qdrant Cloud."
    ]

    print(f"\n[2/4] Executing RAG Pipeline Benchmarks across {len(queries)} evaluation queries...")

    embedding_latencies: List[float] = []
    vector_search_latencies: List[float] = []
    retrieval_total_latencies: List[float] = []
    generation_latencies: List[float] = []
    e2e_total_latencies: List[float] = []
    successful_queries = 0

    for idx, q in enumerate(queries, 1):
        print(f"\n--- Query #{idx}: '{q}' ---")
        
        # Step A: Embedding generation latency
        t_emb_start = time.perf_counter()
        q_vec = embedding_service.embed_query(q)
        t_emb_ms = (time.perf_counter() - t_emb_start) * 1000.0
        embedding_latencies.append(t_emb_ms)

        # Step B: Qdrant Cloud retrieval latency
        t_qdrant_start = time.perf_counter()
        chunks = vector_store.search(q_vec, top_k=4)
        t_qdrant_ms = (time.perf_counter() - t_qdrant_start) * 1000.0
        vector_search_latencies.append(t_qdrant_ms)

        retrieval_total_ms = t_emb_ms + t_qdrant_ms
        retrieval_total_latencies.append(retrieval_total_ms)

        context = retriever.format_context(chunks)

        # Step C: LLM Inference latency
        if provider.lower() == "ollama":
            answer, t_gen_ms = await llm_client.generate_grounded_answer(q, context, model=model)
        else:
            answer, t_gen_ms = await llm_client.generate_grounded_answer(q, context, model="llama-3.3-70b-versatile")
        generation_latencies.append(t_gen_ms)

        t_e2e_ms = retrieval_total_ms + t_gen_ms
        e2e_total_latencies.append(t_e2e_ms)

        if answer and not answer.startswith("Groq Error") and not answer.startswith("Ollama Error") and not answer.startswith("Failed to connect"):
            successful_queries += 1

        print(f"  * Embedding Gen Latency : {t_emb_ms:.2f} ms")
        print(f"  * Qdrant Search Latency : {t_qdrant_ms:.2f} ms")
        print(f"  * Retrieval Total       : {retrieval_total_ms:.2f} ms")
        print(f"  * {provider.upper()} Generation    : {t_gen_ms:.2f} ms ({t_gen_ms/1000.0:.2f} s)")
        print(f"  * End-to-End Latency    : {t_e2e_ms/1000.0:.2f} s ({t_e2e_ms:.2f} ms)")
        print(f"  * Chunks Retrieved      : {len(chunks)}")

    # 3. Calculate Summary Statistics
    print("\n" + "=" * 70)
    print(f"  NAMMA-RAG EMPIRICAL BENCHMARK RESULTS SUMMARY ({provider.upper()})  ")
    print("=" * 70)
    
    avg_emb = statistics.mean(embedding_latencies)
    avg_search = statistics.mean(vector_search_latencies)
    avg_ret = statistics.mean(retrieval_total_latencies)
    avg_gen = statistics.mean(generation_latencies)
    avg_e2e = statistics.mean(e2e_total_latencies)

    print(f"Total Test Runs         : {len(queries)}")
    print(f"Query Success Rate      : {successful_queries}/{len(queries)} ({successful_queries/len(queries)*100:.1f}%)")
    print(f"Avg Embedding Latency   : {avg_emb:.2f} ms")
    print(f"Avg Qdrant Cloud Search : {avg_search:.2f} ms")
    print(f"Avg Total Retrieval     : {avg_ret:.2f} ms")
    print(f"Avg {provider.upper()} Generation  : {avg_gen:.2f} ms ({avg_gen/1000.0:.2f} s)")
    print(f"Avg End-to-End Latency  : {avg_e2e:.2f} ms ({avg_e2e/1000.0:.2f} s)")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark RAG pipeline latency and throughput.")
    parser.add_argument("--provider", choices=["ollama", "groq"], default="ollama", help="LLM provider (default: ollama)")
    parser.add_argument("--model", type=str, default="qwen3:8b", help="Model name (default: qwen3:8b)")
    args = parser.parse_args()

    asyncio.run(run_benchmark(provider=args.provider, model=args.model))

