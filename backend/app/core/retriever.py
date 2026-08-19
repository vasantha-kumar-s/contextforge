import time
from typing import List, Dict, Any, Tuple
from app.core.embeddings import EmbeddingService
from app.core.vector_store import QdrantVectorStore

class RAGRetriever:
    """
    Retrieves top-k relevant context chunks for a user query using Qdrant Cloud dense vector similarity.
    Calculates exact retrieval latency telemetry.
    """

    def __init__(self, embedding_service: EmbeddingService, vector_store: QdrantVectorStore):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 4) -> Tuple[List[Dict[str, Any]], float]:
        start_time = time.perf_counter()
        query_vector = self.embedding_service.embed_query(query)
        chunks = self.vector_store.search(query_vector, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return chunks, elapsed_ms

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "No relevant document context found."

        context_blocks = []
        for idx, chunk in enumerate(chunks, 1):
            source_tag = f"[Source {idx}: {chunk.get('filename', 'Doc')} (Page {chunk.get('page_number', 1)})]"
            context_blocks.append(f"{source_tag}\n{chunk.get('text', '')}")

        return "\n\n".join(context_blocks)
