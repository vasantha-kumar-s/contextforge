import os
import io
import sys
import zipfile
import pandas as pd
from typing import List, Dict, Any, Tuple
from app.core.embeddings import EmbeddingService
from app.core.vector_store import QdrantVectorStore

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

DEFAULT_ZIP_PATH = r"E:\Projects_TP\interview\data\archive (5).zip"
KAGGLE_COLLECTION = "namma_rag_kaggle_eval"

class KaggleDatasetLoader:
    def __init__(self, zip_path: str = DEFAULT_ZIP_PATH):
        self.zip_path = zip_path
        self.documents_df = None
        self.single_qa_df = None
        self.multi_qa_df = None
        self.no_answer_df = None
        self._load_csv_files()

    def _load_csv_files(self):
        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(f"Kaggle zip file not found at {self.zip_path}")
            
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            if 'documents.csv' in z.namelist():
                self.documents_df = pd.read_csv(io.BytesIO(z.read('documents.csv')))
            if 'single_passage_answer_questions.csv' in z.namelist():
                self.single_qa_df = pd.read_csv(io.BytesIO(z.read('single_passage_answer_questions.csv')))
            if 'multi_passage_answer_questions.csv' in z.namelist():
                self.multi_qa_df = pd.read_csv(io.BytesIO(z.read('multi_passage_answer_questions.csv')))
            if 'no_answer_questions.csv' in z.namelist():
                self.no_answer_df = pd.read_csv(io.BytesIO(z.read('no_answer_questions.csv')))

        print("✓ Loaded Kaggle CSVs from zip:")
        print(f"  - Documents: {len(self.documents_df)} passages")
        print(f"  - Single-Passage Questions: {len(self.single_qa_df)} rows")
        print(f"  - Multi-Passage Questions: {len(self.multi_qa_df)} rows")
        print(f"  - No-Answer Questions: {len(self.no_answer_df)} rows")

    def ingest_into_qdrant(self, collection_name: str = KAGGLE_COLLECTION) -> QdrantVectorStore:
        print(f"Vectorizing {len(self.documents_df)} Kaggle passages into Qdrant Cloud collection '{collection_name}'...")
        embedder = EmbeddingService()
        vector_store = QdrantVectorStore(collection_name=collection_name, dimension=384)

        texts = self.documents_df['text'].astype(str).tolist()
        vectors = embedder.embed_texts(texts)
        chunks_meta = []
        for idx, row in self.documents_df.iterrows():
            doc_idx = int(row['index']) if 'index' in row else idx
            source_url = str(row['source_url']) if 'source_url' in row else 'kaggle_doc'
            chunks_meta.append({
                'chunk_id': f"kaggle_doc_{doc_idx}",
                'text': str(row['text']),
                'doc_index': doc_idx,
                'source_url': source_url,
                'filename': f"kaggle_doc_{doc_idx}"
            })

        vector_store.add_chunks(vectors, chunks_meta)
        print(f"✓ Ingested {len(chunks_meta)} passage vectors into Qdrant Cloud collection '{collection_name}'.")
        return vector_store

    def get_evaluation_samples(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        single_samples = self.single_qa_df.to_dict('records') if self.single_qa_df is not None else []
        multi_samples = self.multi_qa_df.to_dict('records') if self.multi_qa_df is not None else []
        no_answer_samples = self.no_answer_df.to_dict('records') if self.no_answer_df is not None else []
        return single_samples, multi_samples, no_answer_samples

if __name__ == '__main__':
    loader = KaggleDatasetLoader()
    v_store = loader.ingest_into_qdrant()
