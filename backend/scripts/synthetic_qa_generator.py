import os
import re
import json
import random
import argparse
from typing import List, Dict, Any
from tqdm import tqdm

from app.core.groq_client import GroqClient
from app.core.ollama_client import OllamaClient
from scripts.kaggle_dataset_loader import KaggleDatasetLoader

QA_GENERATION_PROMPT = """Your task is to write a factoid question and an answer given a context. Your factoid question should be answerable with a specific, concise piece of factual information from the context. Your factoid question should be formulated in the same style as questions users could ask in a search engine. This means that your factoid question MUST NOT mention something like "according to the passage" or "context". Provide your answer as follows:
Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.
Context: {context}
Output:::"""

GROUNDEDNESS_CRITIQUE_PROMPT = """You will be given a context and a question. Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context. Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context. Provide your answer as follows:
Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.
Now here are the question and context.
Question: {question}
Context: {context}
Answer:::"""

RELEVANCE_CRITIQUE_PROMPT = """You will be given a question. Your task is to provide a 'total rating' representing how useful this question can be to developers or domain searchers. Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful. Provide your answer as follows:
Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.
Now here is the question.
Question: {question}
Answer:::"""

STANDALONE_CRITIQUE_PROMPT = """You will be given a question. Your task is to provide a 'total rating' representing how context-independent this question is. Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself. For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
Provide your answer as follows:
Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.
Now here is the question.
Question: {question}
Answer:::"""

class SyntheticQAGenerator:
    def __init__(self, provider: str = "ollama", model: str = "qwen3:8b"):
        self.provider = provider
        self.model = model
        if provider == "groq":
            self.client = GroqClient(model=model)
        else:
            self.client = OllamaClient(model=model)

    def generate(self, prompt: str) -> str:
        if self.provider == "groq":
            return self.client.generate_response(prompt=prompt)
        else:
            return self.client.generate_response(prompt=prompt)

    def parse_critique(self, text: str) -> Tuple[int, str]:
        score = 3
        eval_text = "No evaluation provided"
        try:
            if "Total rating:" in text:
                score_str = text.split("Total rating:")[-1].strip()
                match = re.search(r"\b([1-5])\b", score_str)
                if match:
                    score = int(match.group(1))
            if "Evaluation:" in text and "Total rating:" in text:
                eval_text = text.split("Evaluation:")[-1].split("Total rating:")[0].strip()
        except Exception:
            pass
        return score, eval_text

    def generate_synthetic_dataset(self, contexts: List[str], num_samples: int = 10) -> List[Dict[str, Any]]:
        sampled_contexts = random.sample(contexts, min(num_samples, len(contexts)))
        raw_qa_list = []

        print(f"Generating {len(sampled_contexts)} synthetic candidate QA couples...")
        for ctx in tqdm(sampled_contexts):
            prompt = QA_GENERATION_PROMPT.format(context=ctx)
            resp = self.generate(prompt)
            
            question, answer = None, None
            if "Factoid question:" in resp and "Answer:" in resp:
                try:
                    question = resp.split("Factoid question:")[1].split("Answer:")[0].strip()
                    answer = resp.split("Answer:")[1].strip()
                except Exception:
                    pass

            if question and answer and len(answer) < 400:
                raw_qa_list.append({
                    "context": ctx,
                    "question": question,
                    "answer": answer
                })

        print(f"Critiquing {len(raw_qa_list)} candidates across Groundedness, Relevance, and Standalone filters...")
        filtered_qa_list = []
        for item in tqdm(raw_qa_list):
            q, ctx = item["question"], item["context"]

            g_resp = self.generate(GROUNDEDNESS_CRITIQUE_PROMPT.format(question=q, context=ctx))
            g_score, g_eval = self.parse_critique(g_resp)

            r_resp = self.generate(RELEVANCE_CRITIQUE_PROMPT.format(question=q))
            r_score, r_eval = self.parse_critique(r_resp)

            s_resp = self.generate(STANDALONE_CRITIQUE_PROMPT.format(question=q))
            s_score, s_eval = self.parse_critique(s_resp)

            item.update({
                "groundedness_score": g_score,
                "groundedness_eval": g_eval,
                "relevance_score": r_score,
                "relevance_eval": r_eval,
                "standalone_score": s_score,
                "standalone_eval": s_eval,
            })

            if g_score >= 4 and r_score >= 4 and s_score >= 4:
                filtered_qa_list.append(item)

        print(f"[OK] Quality Filtering Complete: Retained {len(filtered_qa_list)} / {len(raw_qa_list)} high-quality QA couples.")
        return filtered_qa_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default="ollama", choices=["ollama", "groq"])
    parser.add_argument("--model", type=str, default="qwen3:8b")
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    loader = KaggleDatasetLoader()
    contexts = loader.documents_df["text"].tolist()

    generator = SyntheticQAGenerator(provider=args.provider, model=args.model)
    dataset = generator.generate_synthetic_dataset(contexts=contexts, num_samples=args.num_samples)

    out_file = r"e:\Projects_TP\interview\rag-naam\backend\data\synthetic_qa_dataset.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"[OK] Saved synthetic QA dataset to '{out_file}'")
