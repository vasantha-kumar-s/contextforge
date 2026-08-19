import os
import glob
from app.core.db import init_db, SessionLocal, DocumentModel
from app.core.embeddings import EmbeddingService
from app.core.vector_store import QdrantVectorStore
from app.core.extractor import DocumentExtractor
from app.core.chunker import RecursiveTextChunker

def reindex_all():
    print("=" * 70)
    print("  RE-INDEXING ALL DOCUMENTS INTO QDRANT CLOUD & MYSQL  ")
    print("=" * 70)

    init_db()
    embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
    vector_store = QdrantVectorStore(collection_name="namma_rag_knowledge", dimension=384)

    # Recreate Qdrant Collection fresh
    if vector_store.client:
        try:
            print("Recreating Qdrant Cloud Collection 'namma_rag_knowledge'...")
            vector_store.client.recreate_collection(
                collection_name="namma_rag_knowledge",
                vectors_config=vector_store.client.get_collection("namma_rag_knowledge").config.params.vectors
            )
            print("✓ Qdrant Cloud Collection recreated clean.")
        except Exception as e:
            print(f"Recreate warning: {e}")

    # Re-create payload index
    vector_store._init_qdrant()

    chunker = RecursiveTextChunker(chunk_size=800, chunk_overlap=150)
    upload_files = glob.glob("data/uploads/*.*")

    total_chunks = 0
    for file_path in upload_files:
        filename = os.path.basename(file_path)
        print(f"Ingesting '{filename}'...")
        try:
            pages = DocumentExtractor.extract_from_file(file_path, filename)
            if not pages:
                continue

            chunks = chunker.chunk_pages(pages)
            texts = [c["text"] for c in chunks]
            vectors = embedding_service.embed_texts(texts)

            vector_store.add_chunks(vectors, chunks)
            total_chunks += len(chunks)

            db = SessionLocal()
            try:
                doc = db.query(DocumentModel).filter(DocumentModel.filename == filename).first()
                if not doc:
                    doc = DocumentModel(filename=filename, page_count=len(pages), chunk_count=len(chunks))
                    db.add(doc)
                else:
                    doc.chunk_count = len(chunks)
                db.commit()
            finally:
                db.close()

            print(f"  -> Added {len(chunks)} chunks for '{filename}'")
        except Exception as e:
            print(f"Error processing '{filename}': {e}")

    print("=" * 70)
    print(f"SUCCESS: Total {total_chunks} Chunks Ingested into Qdrant Cloud & MySQL!")
    print("=" * 70)

if __name__ == "__main__":
    reindex_all()
