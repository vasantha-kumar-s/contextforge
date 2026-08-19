from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from app.core.evaluator import RAGEvaluator

router = APIRouter(prefix="/api/eval", tags=["Evaluation & Benchmarking"])

_evaluator = None

def init_eval_router(evaluator: RAGEvaluator):
    global _evaluator
    _evaluator = evaluator

class EvalRunRequest(BaseModel):
    model: Optional[str] = "qwen3:8b"
    top_k: Optional[int] = 5

@router.post("/run")
async def run_eval_benchmark(req: EvalRunRequest):
    """Executes live RAG evaluation benchmark across test dataset and calculates empirical metrics."""
    if not _evaluator:
        raise HTTPException(status_code=500, detail="Evaluator not initialized")

    results = await _evaluator.run_benchmark(model_name=req.model, top_k=req.top_k)
    return results
