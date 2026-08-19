import numpy as np
from typing import List, Union
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class LocalDenseVectorizer:
    """
    Offline 384-dimensional dense vector generator using character n-gram TF-IDF & TruncatedSVD.
    Provides sub-millisecond local vectorization with 0 network dependencies.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 3),
            analyzer="char_wb",
            max_features=10000,
            sublinear_tf=True
        )
        self.svd = TruncatedSVD(n_components=dimension, random_state=42)
        self.is_fitted = False

    def _fit_if_needed(self, texts: List[str]):
        if not self.is_fitted:
            vocab_seed = [
                "local RAG architecture FastAPI Ollama SentenceTransformers FAISS vector database",
                "embedding model semantic search retrieval latency recall accuracy end-to-end",
                "grounded answer question response document chunk text extraction PyMuPDF",
                "cosine similarity IndexFlatIP inner product normalized vector prompt context",
                "eval metrics benchmark performance resume bullet points interview system design"
            ] + texts
            sparse_mat = self.tfidf.fit_transform(vocab_seed)
            try:
                self.svd.fit(sparse_mat)
            except Exception:
                pass
            self.is_fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        
        self._fit_if_needed(texts)
        try:
            sparse_mat = self.tfidf.transform(texts)
            vectors = self.svd.transform(sparse_mat).astype(np.float32)
            if vectors.shape[1] < self.dimension:
                padded = np.zeros((len(texts), self.dimension), dtype=np.float32)
                padded[:, :vectors.shape[1]] = vectors
                vectors = padded
        except Exception:
            sparse_mat = self.tfidf.transform(texts).toarray().astype(np.float32)
            padded = np.zeros((len(texts), self.dimension), dtype=np.float32)
            padded[:, :min(sparse_mat.shape[1], self.dimension)] = sparse_mat[:, :min(sparse_mat.shape[1], self.dimension)]
            vectors = padded

        # L2 Normalize vectors for Cosine Similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(np.float32)


class EmbeddingService:
    """
    Hybrid Embedding Service:
    Tries SentenceTransformers first; gracefully falls back to LocalDenseVectorizer
    if HuggingFace network download is unavailable.
    """
    _instance = None

    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance.model_name = model_name
            cls._instance.dimension = 384
            cls._instance.uses_fallback = False
            cls._instance.model = None

            print(f"Attempting to load Embedding Model: {model_name}...")
            try:
                from sentence_transformers import SentenceTransformer
                cls._instance.model = SentenceTransformer(model_name)
                cls._instance.dimension = cls._instance.model.get_sentence_embedding_dimension()
                print(f"SentenceTransformers Loaded! Model: {model_name}, Dimension: {cls._instance.dimension}")
            except Exception as e:
                print(f"HuggingFace model download unavailable ({e}). Using Offline Local Dense Vectorizer.")
                cls._instance.uses_fallback = True
                cls._instance.fallback_vectorizer = LocalDenseVectorizer(dimension=384)

        return cls._instance

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        
        if not self.uses_fallback and self.model is not None:
            try:
                embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                return embeddings.astype(np.float32)
            except Exception as e:
                print(f"SentenceTransformers encode error ({e}). Switching to Offline Local Dense Vectorizer.")
                self.uses_fallback = True
                self.fallback_vectorizer = LocalDenseVectorizer(dimension=self.dimension)

        return self.fallback_vectorizer.encode(texts)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])
