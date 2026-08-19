import os
import sys
import json
import time
import asyncio
import argparse
import pandas as pd
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from app.core.embeddings import EmbeddingService
from app.core.vector_store import QdrantVectorStore
from app.core.groq_client import GroqClient
from app.core.ollama_client import OllamaClient
from scripts.kaggle_dataset_loader import KaggleDatasetLoader, KAGGLE_COLLECTION

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

class KaggleRAGEvaluator:
    def __init__(self, provider: str = "groq", model: str = "llama-3.3-70b-versatile"):
        self.provider = provider
        self.model = model
        self.embedder = EmbeddingService()
        self.vector_store = QdrantVectorStore(collection_name=KAGGLE_COLLECTION, dimension=384)
        
        if provider == "groq":
            self.llm = GroqClient()
        else:
            self.llm = OllamaClient(model=model)

    async def _generate_answer(self, query: str, context: str) -> Tuple[str, float]:
        if self.provider == "groq":
            return await self.llm.generate_grounded_answer(query, context, model=self.model)
        else:
            prompt = f"DOCUMENT CONTEXT:\n{context}\n\nUSER QUESTION:\n{query}\n\nAnswer based ONLY on context above:"
            return self.llm.generate_response(prompt), 0.0

    async def evaluate_single_passage_questions(self, samples: List[Dict[str, Any]], limit: int = 10) -> Dict[str, Any]:
        print(f"\n--- Evaluating {min(limit, len(samples))} Single-Passage Questions ---")
        recalled_at_1 = 0
        recalled_at_5 = 0
        total_eval = 0

        for item in samples[:limit]:
            target_doc_idx = int(item['document_index'])
            question = str(item['question'])
            target_answer = str(item['answer'])

            q_vec = self.embedder.embed_query(question)
            hits = self.vector_store.search(q_vec, top_k=5)

            retrieved_doc_indices = [int(h.get('doc_index', -1)) for h in hits if 'doc_index' in h]
            if retrieved_doc_indices and retrieved_doc_indices[0] == target_doc_idx:
                recalled_at_1 += 1
            if target_doc_idx in retrieved_doc_indices:
                recalled_at_5 += 1

            total_eval += 1

        rec1_pct = round((recalled_at_1 / total_eval) * 100, 1) if total_eval else 0.0
        rec5_pct = round((recalled_at_5 / total_eval) * 100, 1) if total_eval else 0.0

        print(f"  ✓ Recall@1: {recalled_at_1}/{total_eval} ({rec1_pct}%)")
        print(f"  ✓ Recall@5: {recalled_at_5}/{total_eval} ({rec5_pct}%)")

        return {
            "total": total_eval,
            "recall_at_1": rec1_pct,
            "recall_at_5": rec5_pct
        }

    async def evaluate_no_answer_questions(self, samples: List[Dict[str, Any]], limit: int = 10) -> Dict[str, Any]:
        print(f"\n--- Evaluating {min(limit, len(samples))} No-Answer (Abstention) Questions ---")
        abstained_count = 0
        total_eval = 0

        for item in samples[:limit]:
            question = str(item['question'])
            q_vec = self.embedder.embed_query(question)
            hits = self.vector_store.search(q_vec, top_k=3)
            context = "\n\n".join([h.get('text', '') for h in hits])

            ans_text, _ = await self._generate_answer(question, context)
            
            # Check if model properly abstained
            lower_ans = ans_text.lower()
            if any(k in lower_ans for k in ["not mentioned", "cannot be deduced", "does not contain", "no information", "not state", "unclear"]):
                abstained_count += 1

            total_eval += 1

        abstention_pct = round((abstained_count / total_eval) * 100, 1) if total_eval else 0.0
        print(f"  ✓ Correct Abstention Rate: {abstained_count}/{total_eval} ({abstention_pct}%)")

        return {
            "total": total_eval,
            "abstention_rate": abstention_pct
        }

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default="groq", choices=["groq", "ollama"])
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    loader = KaggleDatasetLoader()
    evaluator = KaggleRAGEvaluator(provider=args.provider, model=args.model)
    single_q, multi_q, no_ans_q = loader.get_evaluation_samples()

    single_res = await evaluator.evaluate_single_passage_questions(single_q, limit=args.limit)
    no_ans_res = await evaluator.evaluate_no_answer_questions(no_ans_q, limit=args.limit)

    print(f"\n=======================================================")
    print(f"  KAGGLE RAG BENCHMARK COMPLETED SUCCESSFULLY ({args.provider.upper()})")
    print(f"=======================================================")

if __name__ == '__main__':
    asyncio.run(main())
