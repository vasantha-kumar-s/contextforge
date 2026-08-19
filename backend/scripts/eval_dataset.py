"""
eval_dataset.py
Ground-truth evaluation dataset for Namma-RAG.
50 questions across 2 indexed documents:
  - PROJECT_DOCUMENTATION.pdf    (74 chunks)
  - Published_Paper_6-2-9-272.pdf (85 chunks)

Types:
  "answerable"   – 30 questions with clear answers in the documents
  "multi_hop"    – 10 questions requiring evidence from multiple chunks
  "unanswerable" – 10 questions whose answers are NOT in the documents
"""

EVAL_DATASET = [

    # ============================================================
    # SECTION A - ANSWERABLE (30 questions)
    # ============================================================

    # Published_Paper questions (15)
    {
        "id": 1, "type": "answerable",
        "question": "Who is the author of the published research paper?",
        "expected_document": "Published_Paper_6-2-9-272.pdf",
        "expected_snippet": "Sadik Lakhani",
        "ground_truth": "Sadik Lakhani, affiliated with P P Savani University."
    },
    {
        "id": 2, "type": "answerable",
        "question": "What is the DOI of the published paper?",
        "expected_document": "Published_Paper_6-2-9-272.pdf",
        "expected_snippet": "10.33545",
        "ground_truth": "10.33545/27076636.2025.v6.i2b.127"
    },
    {
        "id": 3, "type": "answerable",
        "question": "How many times has the published paper been read?",
        "expected_document": "Published_Paper_6-2-9-272.pdf",
        "expected_snippet": "38",
        "ground_truth": "38 reads."
    },
    {
        "id": 4, "type": "answerable",
        "question": "How many citations does the published paper have?",
        "expected_document": "Published_Paper_6-2-9-272.pdf",
        "expected_snippet": "0",
        "ground_truth": "0 citations."
    },
    {
        "id": 5, "type": "answerable",
        "question": "What university is the paper author affiliated with?",
        "expected_document": "Published_Paper_6-2-9-272.pdf",
        "expected_snippet": "P P Savani University",
        "ground_truth": "P P Savani University."
    },
    {
        "id": 6, "type": "answerable",
        "question": "What plant diseases or leaf types are studied in the research paper?",
        "expected_document": "Published_Paper_6-2-9-272.pdf",
        "expected_snippet": "leaf",
        "ground_truth": "Plant leaf images including maize and potato leaves for disease detection."
    },
    {
        "id": 7, "type": "answerable",
        "question": "What image resolution is used in the published paper dataset?",
        "expected_document": "Published_Paper_6-2-9-272.pdf",
        "expected_snippet": "256",
        "ground_truth": "Images are resized to 256x256 pixels and normalized."
    },
    {
        "id": 8, "type": "answerable",
        "question": "Who are the references cited in the published research paper?",
        "expected_document": "Published_Paper_6-2-9-272.pdf",
        "expected_snippet": "Singh",
        "ground_truth": "Singh S, Solanki U, Vashi S (2025); Sharma et al (2025); Kriti et al (2021); Pathak et al (2021)."
    },
    {
        "id": 9, "type": "answerable",
        "question": "What journal volume and issue was the research paper published in?",
        "expected_document": "Published_Paper_6-2-9-272.pdf",
        "expected_snippet": "v6",
        "ground_truth": "Volume 6, Issue 2 (2025)."
    },
    {
        "id": 10, "type": "answerable",
        "question": "What preprocessing steps are applied to images in the paper?",
        "expected_document": "Published_Paper_6-2-9-272.pdf",
        "expected_snippet": "normalized",
        "ground_truth": "Images are resized and normalized before being fed into the model."
    },
    {
        "id": 11, "type": "answerable",
        "question": "What deep learning approach is used in the research paper?",
        "expected_document": "Published_Paper_6-2-9-272.pdf",
        "expected_snippet": "deep learning",
        "ground_truth": "Deep learning / CNN approach for plant disease detection."
    },
    {
        "id": 12, "type": "answerable",
        "question": "What is the subject area of the published research paper?",
        "expected_document": "Published_Paper_6-2-9-272.pdf",
        "expected_snippet": "disease",
        "ground_truth": "Plant disease detection using deep learning."
    },
    {
        "id": 13, "type": "answerable",
        "question": "What dataset is used in the published research paper?",
        "expected_document": "Published_Paper_6-2-9-272.pdf",
        "expected_snippet": "dataset",
        "ground_truth": "A plant leaf disease image dataset containing maize and potato leaf images."
    },
    {
        "id": 14, "type": "answerable",
        "question": "What year was the paper published?",
        "expected_document": "Published_Paper_6-2-9-272.pdf",
        "expected_snippet": "2025",
        "ground_truth": "2025."
    },
    {
        "id": 15, "type": "answerable",
        "question": "What metric does the paper report for its model performance?",
        "expected_document": "Published_Paper_6-2-9-272.pdf",
        "expected_snippet": "accuracy",
        "ground_truth": "The paper reports classification accuracy metrics for plant disease detection."
    },

    # PROJECT_DOCUMENTATION questions (15)
    {
        "id": 16, "type": "answerable",
        "question": "What is the name of this RAG project?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "Namma",
        "ground_truth": "Namma-RAG."
    },
    {
        "id": 17, "type": "answerable",
        "question": "What vector database is used in Namma-RAG?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "Qdrant",
        "ground_truth": "Qdrant Cloud (AWS EU-Central cluster)."
    },
    {
        "id": 18, "type": "answerable",
        "question": "What embedding model is used in Namma-RAG?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "MiniLM",
        "ground_truth": "all-MiniLM-L6-v2 from SentenceTransformers, producing 384-dimensional dense vectors."
    },
    {
        "id": 19, "type": "answerable",
        "question": "What relational database stores session and metadata in Namma-RAG?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "MySQL",
        "ground_truth": "MySQL at mysql+pymysql://root:root@localhost:3306/namma_rag."
    },
    {
        "id": 20, "type": "answerable",
        "question": "What LLM cloud provider is used in Namma-RAG?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "Groq",
        "ground_truth": "Groq Cloud API (llama-3.3-70b-versatile and llama3-8b-8192)."
    },
    {
        "id": 21, "type": "answerable",
        "question": "What local LLM runtime does Namma-RAG support?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "Ollama",
        "ground_truth": "Ollama local runtime."
    },
    {
        "id": 22, "type": "answerable",
        "question": "What chunk size and overlap does the Namma-RAG text chunker use?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "800",
        "ground_truth": "800 characters chunk size with 150 character overlap."
    },
    {
        "id": 23, "type": "answerable",
        "question": "What PDF parsing library does Namma-RAG use?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "PyMuPDF",
        "ground_truth": "PyMuPDF (fitz), using page.get_text('blocks') for block-level extraction."
    },
    {
        "id": 24, "type": "answerable",
        "question": "What MySQL tables are used in Namma-RAG?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "chat_sessions",
        "ground_truth": "documents, chat_sessions, and chat_messages."
    },
    {
        "id": 25, "type": "answerable",
        "question": "What is the vector dimension used in Namma-RAG?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "384",
        "ground_truth": "384-dimensional dense vectors."
    },
    {
        "id": 26, "type": "answerable",
        "question": "What frontend framework is used in Namma-RAG?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "React",
        "ground_truth": "React 18."
    },
    {
        "id": 27, "type": "answerable",
        "question": "What backend framework is used in Namma-RAG?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "FastAPI",
        "ground_truth": "FastAPI (Python)."
    },
    {
        "id": 28, "type": "answerable",
        "question": "What distance metric does Qdrant use in Namma-RAG?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "Cosine",
        "ground_truth": "Cosine similarity."
    },
    {
        "id": 29, "type": "answerable",
        "question": "What is the Qdrant Cloud network timeout configured in Namma-RAG?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "timeout",
        "ground_truth": "3.0 second socket timeout to prevent network hangs."
    },
    {
        "id": 30, "type": "answerable",
        "question": "What color theme does the Namma-RAG UI use?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "dark",
        "ground_truth": "Matte black and zinc dark theme inspired by Vercel."
    },

    # ============================================================
    # SECTION B - MULTI-HOP (10 questions)
    # ============================================================
    {
        "id": 31, "type": "multi_hop",
        "question": "How does Namma-RAG handle Ollama running out of memory and what is the fallback?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "Groq",
        "ground_truth": "Namma-RAG catches the Ollama 500 RAM error and suggests switching to Groq Cloud API."
    },
    {
        "id": 32, "type": "multi_hop",
        "question": "Why was chunk size increased from 500 to 800 characters in Namma-RAG?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "chunk",
        "ground_truth": "Smaller chunks fragmented research paper paragraphs; 800 chars keeps full paragraphs intact."
    },
    {
        "id": 33, "type": "multi_hop",
        "question": "What problem did two-column PDF parsing cause and how was it fixed in Namma-RAG?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "blocks",
        "ground_truth": "Naive extraction interleaved column text; fixed with PyMuPDF get_text('blocks') block sorting."
    },
    {
        "id": 34, "type": "multi_hop",
        "question": "Why was FAISS removed from Namma-RAG and what replaced it?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "Qdrant",
        "ground_truth": "FAISS was local-only; replaced with Qdrant Cloud for persistent cloud vector storage."
    },
    {
        "id": 35, "type": "multi_hop",
        "question": "How does Namma-RAG prevent stale browser script hash errors after a frontend rebuild?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "Cache-Control",
        "ground_truth": "FastAPI serves index.html with Cache-Control: no-cache, no-store headers."
    },
    {
        "id": 36, "type": "multi_hop",
        "question": "What caused high latency in Namma-RAG and how was it reduced from 6 seconds to 1.3 seconds?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "timeout",
        "ground_truth": "Qdrant Cloud socket read hangs caused 5s+ delays; fixed with 3s timeout and in-memory cache."
    },
    {
        "id": 37, "type": "multi_hop",
        "question": "What is the full end-to-end latency breakdown for a Namma-RAG query?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "retrieval",
        "ground_truth": "~15ms embedding + ~255ms Qdrant retrieval + ~1050ms Groq generation = ~1.3s total."
    },
    {
        "id": 38, "type": "multi_hop",
        "question": "How does the Namma-RAG in-memory cosine fallback work when Qdrant Cloud times out?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "cosine",
        "ground_truth": "Cached chunk vectors are searched locally using cosine distance when Qdrant times out."
    },
    {
        "id": 39, "type": "multi_hop",
        "question": "What are the MySQL table schemas for storing documents and sessions in Namma-RAG?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "documents",
        "ground_truth": "documents (id, filename, upload_date, page_count, chunk_count), chat_sessions, chat_messages."
    },
    {
        "id": 40, "type": "multi_hop",
        "question": "How does Namma-RAG render formatted markdown lists and bold text in the chat interface?",
        "expected_document": "PROJECT_DOCUMENTATION.pdf",
        "expected_snippet": "markdown",
        "ground_truth": "renderFormattedContent in ChatWorkbench.jsx parses **bold**, numbered/bullet lists, and source badges."
    },

    # ============================================================
    # SECTION C - UNANSWERABLE (10 questions)
    # These answers are NOT in any indexed document.
    # A good RAG system should say it cannot find the information.
    # ============================================================
    {
        "id": 41, "type": "unanswerable",
        "question": "What is the monthly price of Qdrant Cloud?",
        "expected_document": None, "expected_snippet": None,
        "ground_truth": "Not available in the provided documents."
    },
    {
        "id": 42, "type": "unanswerable",
        "question": "What is Groq's annual revenue as of 2024?",
        "expected_document": None, "expected_snippet": None,
        "ground_truth": "Not available in the provided documents."
    },
    {
        "id": 43, "type": "unanswerable",
        "question": "Who founded the company that makes Ollama?",
        "expected_document": None, "expected_snippet": None,
        "ground_truth": "Not available in the provided documents."
    },
    {
        "id": 44, "type": "unanswerable",
        "question": "What is the capital city of Germany?",
        "expected_document": None, "expected_snippet": None,
        "ground_truth": "Not available in the provided documents."
    },
    {
        "id": 45, "type": "unanswerable",
        "question": "What is NVIDIA's market capitalization in 2025?",
        "expected_document": None, "expected_snippet": None,
        "ground_truth": "Not available in the provided documents."
    },
    {
        "id": 46, "type": "unanswerable",
        "question": "What is the published paper author's email address?",
        "expected_document": None, "expected_snippet": None,
        "ground_truth": "Not available in the provided documents."
    },
    {
        "id": 47, "type": "unanswerable",
        "question": "What is the monthly AWS cost of running Namma-RAG in production?",
        "expected_document": None, "expected_snippet": None,
        "ground_truth": "Not available in the provided documents."
    },
    {
        "id": 48, "type": "unanswerable",
        "question": "Who are the founders of UANDWE?",
        "expected_document": None, "expected_snippet": None,
        "ground_truth": "Not available in the provided documents."
    },
    {
        "id": 49, "type": "unanswerable",
        "question": "What is the SRM University annual tuition fee?",
        "expected_document": None, "expected_snippet": None,
        "ground_truth": "Not available in the provided documents."
    },
    {
        "id": 50, "type": "unanswerable",
        "question": "What programming language is Qdrant itself written in?",
        "expected_document": None, "expected_snippet": None,
        "ground_truth": "Not available in the provided documents."
    },
]

ANSWERABLE   = [q for q in EVAL_DATASET if q["type"] == "answerable"]
MULTI_HOP    = [q for q in EVAL_DATASET if q["type"] == "multi_hop"]
UNANSWERABLE = [q for q in EVAL_DATASET if q["type"] == "unanswerable"]

if __name__ == "__main__":
    print(f"Total questions  : {len(EVAL_DATASET)}")
    print(f"  Answerable     : {len(ANSWERABLE)}")
    print(f"  Multi-hop      : {len(MULTI_HOP)}")
    print(f"  Unanswerable   : {len(UNANSWERABLE)}")
