import os
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class QdrantVectorStore:
    """
    Qdrant Cloud Vector Store implementation with 3s timeout and in-memory cosine fallback.
    Prevents 5-second cloud network socket delays.
    """

    def __init__(self, collection_name: str = "namma_rag_knowledge", dimension: int = 384):
        self.collection_name = collection_name
        self.dimension = dimension
        self.url = os.getenv("QDRANT_URL", "https://115caf20-7cb5-4c37-bc1d-fdcf162696ab.eu-central-1-0.aws.cloud.qdrant.io")
        self.api_key = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6MTQyMDM0MWEtYTMxMy00NWFkLTkxMmEtY2NkNzc5ODY1NjExIn0.DHzd-5MCF1yYnwMbB3fitkwUGZ2s1bJ9aksZhuizdgw")
        self.client = None
        self.cached_chunks: List[Dict[str, Any]] = []
        self._search_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._init_qdrant()

    def _init_qdrant(self):
        if not self.url or not self.api_key:
            return
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import VectorParams, Distance, PayloadSchemaType
            
            # 15.0s timeout for reliable cloud connection
            self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=15.0, check_compatibility=False)
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE)
                )
                print(f"Qdrant Cloud Collection Created/Connected: '{self.collection_name}'")

            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="filename",
                    field_schema=PayloadSchemaType.KEYWORD
                )
            except Exception:
                pass
        except Exception as e:
            print(f"Qdrant Cloud connection warning: {e}")
            self.client = None

    def add_chunks(self, vectors: np.ndarray, chunks_metadata: List[Dict[str, Any]]):
        if len(vectors) == 0:
            return
        for v, meta in zip(vectors, chunks_metadata):
            meta["_vector"] = v.tolist()
            self.cached_chunks.append(meta)

        if self.client:
            try:
                from qdrant_client.models import PointStruct
                import uuid

                points = []
                for v, meta in zip(vectors, chunks_metadata):
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, meta["chunk_id"]))
                    clean_meta = {k: v for k, v in meta.items() if k != "_vector"}
                    points.append(PointStruct(id=point_id, vector=v.tolist(), payload=clean_meta))

                self.client.upsert(collection_name=self.collection_name, points=points)
                self._search_cache.clear()
            except Exception as e:
                print(f"Qdrant Cloud upsert warning: {e}")

    def _in_memory_search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.cached_chunks:
            return []
        
        q_vec = query_vector.flatten()
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        scored = []
        for chunk in self.cached_chunks:
            vec = chunk.get("_vector")
            if vec is None:
                continue
            v_arr = np.array(vec)
            v_norm = np.linalg.norm(v_arr)
            if v_norm == 0:
                continue
            sim = float(np.dot(q_vec, v_arr) / (q_norm * v_norm))
            clean_meta = {k: v for k, v in chunk.items() if k != "_vector"}
            clean_meta["similarity_score"] = round(sim, 4)
            scored.append(clean_meta)

        scored.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored[:top_k]

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        vec_list = query_vector[0].tolist() if query_vector.ndim > 1 else query_vector.tolist()
        q_key = str([round(x, 4) for x in vec_list[:8]]) + f"_{top_k}"
        
        if q_key in self._search_cache:
            return self._search_cache[q_key]

        if not self.client:
            return self._deduplicate(self._in_memory_search(query_vector, top_k * 3), top_k)

        try:
            # Over-fetch 3x to allow deduplication of repeated PDF headers/footers
            fetch_limit = top_k * 3
            hits = []
            if hasattr(self.client, "search"):
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=vec_list,
                    limit=fetch_limit
                )
            elif hasattr(self.client, "query_points"):
                hits = self.client.query_points(
                    collection_name=self.collection_name,
                    query=vec_list,
                    limit=fetch_limit
                ).points

            results = []
            for hit in hits:
                meta = dict(hit.payload or {})
                meta["similarity_score"] = round(float(getattr(hit, "score", 0.0)), 4)
                results.append(meta)

            if not results and self.cached_chunks:
                results = self._in_memory_search(query_vector, fetch_limit)

            results = self._deduplicate(results, top_k)
            self._search_cache[q_key] = results
            return results
        except Exception as e:
            print(f"Qdrant Cloud search timeout/error: {e}. Falling back to fast local memory search...")
            results = self._in_memory_search(query_vector, top_k * 3)
            results = self._deduplicate(results, top_k)
            self._search_cache[q_key] = results
            return results

    @staticmethod
    def _deduplicate(chunks: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Remove chunks with identical text content, keeping the highest-scored one."""
        seen_texts = set()
        unique = []
        for chunk in chunks:
            # Normalize: first 200 chars of text as dedup key
            text_key = chunk.get("text", "")[:200].strip().lower()
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)
            unique.append(chunk)
            if len(unique) >= top_k:
                break
        return unique

    def delete_document(self, filename: str):
        self.cached_chunks = [c for c in self.cached_chunks if c.get("filename") != filename]
        self._search_cache.clear()
        if not self.client:
            return
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
                )
            )
        except Exception as e:
            print(f"Qdrant Cloud delete error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        if not self.client:
            return {"total_chunks": len(self.cached_chunks), "total_documents": 0, "store_type": "Qdrant Cloud (Offline)"}
        try:
            c_info = self.client.get_collection(self.collection_name)
            return {
                "total_chunks": c_info.points_count or len(self.cached_chunks),
                "vector_dimension": self.dimension,
                "store_type": "Qdrant Cloud (AWS EU-Central)",
                "collection_name": self.collection_name
            }
        except Exception:
            return {"total_chunks": len(self.cached_chunks), "store_type": "Qdrant Cloud"}
