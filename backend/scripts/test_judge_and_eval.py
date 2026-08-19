import asyncio
import os
import httpx
from app.core.embeddings import EmbeddingService
from app.core.vector_store import QdrantVectorStore
from app.core.retriever import RAGRetriever
from app.core.groq_client import GroqClient

async def test_single_q():
    e = EmbeddingService(model_name="all-MiniLM-L6-v2")
    v = QdrantVectorStore(collection_name="namma_rag_knowledge", dimension=384)
    r = RAGRetriever(embedding_service=e, vector_store=v)
    g = GroqClient()

    q = "Who is the author of the published research paper?"
    chunks, ret_ms = r.retrieve(q, top_k=5)

    print("Q:", q)
    print("Retrieved Chunks:")
    for idx, c in enumerate(chunks, 1):
        print(f" Chunk {idx}: [{c.get('filename')}] score={c.get('similarity_score')} text={c.get('text', '')[:100]}...")

    context = r.format_context(chunks)
    ans, gen_ms = await g.generate_grounded_answer(q, context, model="llama-3.3-70b-versatile")
    print("\nGenerated RAG Answer:")
    print(ans)

if __name__ == "__main__":
    asyncio.run(test_single_q())
