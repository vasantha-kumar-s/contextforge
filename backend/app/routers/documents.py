import os
import shutil
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from sqlalchemy.orm import Session
from app.core.extractor import DocumentExtractor
from app.core.chunker import RecursiveTextChunker
from app.core.db import SessionLocal, DocumentModel

router = APIRouter(prefix="/api/documents", tags=["Documents"])

_embedding_service = None
_vector_store = None

def init_document_router(embedding_service, vector_store):
    global _embedding_service, _vector_store
    _embedding_service = embedding_service
    _vector_store = vector_store

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.get("/stats")
def get_stats():
    """Returns vector store and document statistics."""
    if not _vector_store:
        return {"total_chunks": 0, "total_documents": 0}
    
    db: Session = SessionLocal()
    try:
        doc_count = db.query(DocumentModel).count()
        v_stats = _vector_store.get_stats()
        v_stats["total_documents"] = doc_count
        v_stats["filenames"] = [d.filename for d in db.query(DocumentModel).all()]
        return v_stats
    finally:
        db.close()

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(800),
    chunk_overlap: int = Form(150)
):
    """Uploads document, extracts text, generates vector embeddings, and stores in Qdrant Cloud & MySQL."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Invalid file name")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        pages = DocumentExtractor.extract_from_file(file_path, file.filename)
        if not pages:
            raise HTTPException(status_code=400, detail="No readable text extracted")

        chunker = RecursiveTextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_pages(pages)

        texts = [c["text"] for c in chunks]
        vectors = _embedding_service.embed_texts(texts)

        _vector_store.add_chunks(vectors, chunks)

        db: Session = SessionLocal()
        try:
            doc = db.query(DocumentModel).filter(DocumentModel.filename == file.filename).first()
            if not doc:
                doc = DocumentModel(filename=file.filename, page_count=len(pages), chunk_count=len(chunks))
                db.add(doc)
            else:
                doc.page_count = len(pages)
                doc.chunk_count = len(chunks)
            db.commit()
        finally:
            db.close()

        return {
            "message": f"Successfully processed '{file.filename}' into Qdrant Cloud & MySQL",
            "filename": file.filename,
            "pages_extracted": len(pages),
            "chunks_created": len(chunks)
        }
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@router.get("/chunks")
def list_chunks(limit: int = 50):
    if not _vector_store:
        return []
    return _vector_store.cached_chunks[:limit]

@router.delete("/{filename}")
def delete_document(filename: str):
    if not _vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        
    _vector_store.delete_document(filename)

    db: Session = SessionLocal()
    try:
        doc = db.query(DocumentModel).filter(DocumentModel.filename == filename).first()
        if doc:
            db.delete(doc)
            db.commit()
    finally:
        db.close()

    return {"message": f"Deleted document '{filename}'", "stats": get_stats()}

@router.post("/load-sample-data")
def load_sample_data():
    """Populates Qdrant Cloud and MySQL with a broad multi-domain technical document dataset."""
    sample_docs = [
        {
            "filename": "namma_rag_architecture.pdf",
            "content": """
================================================================================
NAMMA-RAG PLATFORM ARCHITECTURE SPECIFICATION
================================================================================
1. ARCHITECTURAL OVERVIEW
Namma-RAG is a high-throughput Retrieval-Augmented Generation platform. It combines PyMuPDF for document parsing, SentenceTransformers (384-dimensional dense embeddings) for semantic vector representations, Qdrant Cloud for vector similarity search, MySQL for relational session storage, and Groq/Ollama LLM inference engines.

2. QDRANT CLOUD EXCLUSIVE VECTOR DB
All extracted text chunks and 384-d normalized vector embeddings are stored directly in Qdrant Cloud (AWS EU-Central cluster).

3. MYSQL DATABASE STORAGE
Relational data, user chat sessions, conversation turns, and document metadata are stored in MySQL (mysql+pymysql://root:root@localhost:3306/namma_rag).

4. HYBRID LLM INFERENCE STRATEGY
Namma-RAG supports both cloud inference via Groq Cloud API (llama-3.3-70b-versatile / llama3-8b-8192) for sub-second generation (~0.96s) and local inference via Ollama for offline environments.
"""
        },
        {
            "filename": "system_design_concurrency.pdf",
            "content": """
================================================================================
SYSTEM DESIGN, CONCURRENCY & DATABASE INDEXING
================================================================================
1. DATABASE INDEXING AND B-TREES
B-Tree and B+Tree indexes reduce lookup time complexity from O(N) linear table scans to O(log N) tree traversals. In MySQL InnoDB, secondary indexes reference primary key values to retrieve row data.

2. PYTHON ASYNCHRONOUS CONCURRENCY
Python asyncio provides event-loop-driven non-blocking I/O operations. Using async HTTP clients like httpx allows high concurrency without thread overhead.

3. REST VS GRPC IPC
REST APIs rely on JSON over HTTP/1.1 with human-readable payloads. gRPC uses Protocol Buffers over HTTP/2 multiplexed streams, providing binary serialization for high-performance microservice communication.
"""
        },
        {
            "filename": "network_distributed_systems.pdf",
            "content": """
================================================================================
DISTRIBUTED SYSTEMS AND NETWORK PROTOCOLS
================================================================================
1. CAP THEOREM
The CAP Theorem states that a distributed data store can simultaneously provide at most two out of three guarantees: Consistency, Availability, and Partition Tolerance.

2. RAFT CONSENSUS ALGORITHM
Raft decomposes consensus into leader election, log replication, and safety guarantees. Leaders manage log entries across cluster nodes to maintain state machine consistency.

3. LOAD BALANCING ALGORITHMS
Round-robin distributes incoming requests sequentially across servers. Least-connections routes requests to the server with the fewest active connections, optimizing load distribution.
"""
        },
        {
            "filename": "transformer_attention_mechanisms.pdf",
            "content": """
================================================================================
TRANSFORMER ARCHITECTURES AND ATTENTION MATHEMATICS
================================================================================
1. SCALED DOT-PRODUCT ATTENTION
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V. Query, Key, and Value matrices transform input token embeddings into contextual representations.

2. MULTI-HEAD LATENT ATTENTION (MLA)
Multi-Head Latent Attention compresses Key and Value caches into low-rank latent vectors, drastically reducing memory footprint during generation.

3. POSITIONAL ENCODINGS
RoPE (Rotary Position Embedding) encodes absolute position with a rotation matrix, allowing transformers to generalize across flexible sequence lengths.
"""
        },
        {
            "filename": "database_storage_engines.pdf",
            "content": """
================================================================================
DATABASE STORAGE ENGINES AND ACID GUARANTEES
================================================================================
1. ACID TRANSACTIONS
Atomicity guarantees all-or-nothing execution. Consistency enforces schema constraints. Isolation prevents race conditions via MVCC (Multi-Version Concurrency Control). Durability uses Write-Ahead Logging (WAL) / redo logs.

2. LSM TREES VS B-TREES
LSM Trees (Log-Structured Merge Trees) optimize write throughput by appending writes to MemTables and SSTables. B-Trees optimize read operations at the expense of random write amplification.
"""
        },
        {
            "filename": "microservices_kubernetes_guide.pdf",
            "content": """
================================================================================
MICROSERVICES AND KUBERNETES CONTAINER ORCHESTRATION
================================================================================
1. KUBERNETES CONTROL PLANE
The Kubernetes Control Plane consists of the API Server, kube-scheduler, kube-controller-manager, and etcd distributed key-value store.

2. SERVICE MESH AND CIRCUIT BREAKING
Istio service mesh implements mTLS encryption, traffic splitting, and circuit breaking to prevent cascading failures across distributed microservices.
"""
        },
        {
            "filename": "web_security_oauth_tls.pdf",
            "content": """
================================================================================
WEB SECURITY, OAUTH 2.0 AND TLS 1.3 PROTOCOLS
================================================================================
1. OAUTH 2.0 AND OPENID CONNECT
OAuth 2.0 provides delegated authorization via JWT access tokens. OpenID Connect adds an Identity layer over OAuth 2.0 using ID tokens.

2. TLS 1.3 HANDSHAKE
TLS 1.3 reduces handshake latency from 2-RTT to 1-RTT by combining key exchange and cipher negotiation in the initial ClientHello message.
"""
        }
    ]

    total_chunks_added = 0
    chunker = RecursiveTextChunker(chunk_size=800, chunk_overlap=150)

    for doc_info in sample_docs:
        fname = doc_info["filename"]
        file_path = os.path.join(UPLOAD_DIR, fname)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc_info["content"])

        pages = [{"filename": fname, "page_number": 1, "text": doc_info["content"]}]
        chunks = chunker.chunk_pages(pages)
        texts = [c["text"] for c in chunks]
        vectors = _embedding_service.embed_texts(texts)

        _vector_store.delete_document(fname)
        _vector_store.add_chunks(vectors, chunks)
        total_chunks_added += len(chunks)

        db: Session = SessionLocal()
        try:
            doc = db.query(DocumentModel).filter(DocumentModel.filename == fname).first()
            if not doc:
                doc = DocumentModel(filename=fname, page_count=1, chunk_count=len(chunks))
                db.add(doc)
            else:
                doc.chunk_count = len(chunks)
            db.commit()
        finally:
            db.close()

    return {
        "message": f"Loaded expanded technical documents into Qdrant Cloud & MySQL! Created {total_chunks_added} chunks.",
        "stats": get_stats()
    }
