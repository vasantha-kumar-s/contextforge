from app.core.embeddings import EmbeddingService
from app.core.vector_store import QdrantVectorStore
from app.core.retriever import RAGRetriever

e = EmbeddingService(model_name="all-MiniLM-L6-v2")
v = QdrantVectorStore(collection_name="namma_rag_knowledge", dimension=384)
r = RAGRetriever(embedding_service=e, vector_store=v)

test_questions = [
    ("Who is the author of the published research paper?", "Sadik Lakhani"),
    ("What vector database is used in Namma-RAG?", "Qdrant"),
    ("What embedding model is used in Namma-RAG?", "MiniLM"),
    ("What is the DOI of the published paper?", "10.33545"),
    ("What is the capital city of Germany?", None),  # unanswerable
]

for q, snippet in test_questions:
    chunks, ms = r.retrieve(q, top_k=5)
    print(f"\n{'='*70}")
    print(f"Q: {q}")
    print(f"Retrieval: {ms:.0f}ms | Expected snippet: {snippet}")
    for i, c in enumerate(chunks):
        fn = c.get("filename", "?")
        score = c.get("similarity_score", 0)
        text_preview = c.get("text", "")[:120].replace("\n", " ")
        found = ""
        if snippet and snippet.lower() in c.get("text", "").lower():
            found = " <-- SNIPPET FOUND"
        print(f"  Rank {i+1}: [{fn}] sim={score:.4f} | {text_preview}...{found}")
