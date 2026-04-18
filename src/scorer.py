"""
Ahoum Conversation Evaluation Engine
Scores conversation turns across 300+ facets using a local Ollama model via Langchain.
"""

import os
import time
from typing import List
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

# Highly optimized for Llama 3.1 8B context tracking abilities
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "150"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")  # Defaulting to llama3.1


class FacetScore(BaseModel):
    facet_id: int
    facet_name: str
    score: int = Field(description="1-5 integer score (1=Very Low/Absent, 5=Very High/Dominant)")
    confidence: float = Field(description="0.0-1.0 confidence")
    rationale: str = Field(description="One short sentence rationale based ONLY on text evidence")

class BatchScoreOutput(BaseModel):
    scores: list[FacetScore]

SYSTEM_PROMPT = """You are a precise conversation evaluation judge. Score a conversation turn on behavioral/linguistic/psychological facets.

Rules:
1. Base scores ONLY on evidence in the text. 
2. Unobservable facets (physiological, activity counts): score=1, confidence=0.1.
3. Use the full 1-5 range. Do NOT default everything to 3.
"""

def get_llm(model: str = OLLAMA_MODEL):
    return ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    ).with_structured_output(BatchScoreOutput)

def score_turn(
    conversation_turn: str,
    facets: list[dict],
    batch_size: int = BATCH_SIZE,
    model: str = OLLAMA_MODEL,
) -> list[dict]:
    all_scores = []
    batches = [facets[i:i+batch_size] for i in range(0, len(facets), batch_size)]
    llm = get_llm(model)

    for batch_idx, batch in enumerate(batches):
        facet_list = "\n".join(
            f"- [ID:{f['facet_id']}] {f['facet_name']} (category: {f.get('category','other')})"
            for f in batch
        )

        user_prompt = f"""{SYSTEM_PROMPT}

Conversation turn to evaluate:
---
{conversation_turn}
---

Score EXACTLY these {len(batch)} facets based on the text:
{facet_list}
"""
        try:
            structured_resp = llm.invoke(user_prompt)
            
            if structured_resp and structured_resp.scores:
                returned_ids = {s.facet_id: s for s in structured_resp.scores}
                for f in batch:
                    if f["facet_id"] in returned_ids:
                        s = returned_ids[f["facet_id"]]
                        all_scores.append({
                            "facet_id": s.facet_id,
                            "facet_name": s.facet_name,
                            "score": s.score,
                            "confidence": s.confidence,
                            "rationale": s.rationale,
                        })
                    else:
                        all_scores.append({
                            "facet_id": f["facet_id"],
                            "facet_name": f["facet_name"],
                            "score": 3,
                            "confidence": 0.0,
                            "rationale": "Missing from structured output array",
                        })
            else:
                raise ValueError("Empty or invalid structured response.")

        except Exception as e:
            for f in batch:
                all_scores.append({
                    "facet_id": f["facet_id"],
                    "facet_name": f["facet_name"],
                    "score": 3,
                    "confidence": 0.0,
                    "rationale": f"LLM parsing error: {str(e)[:60]}",
                })

        if batch_idx < len(batches) - 1:
            time.sleep(0.1)

    return all_scores

def score_conversation(
    conversation: list[dict],
    facets: list[dict],
    batch_size: int = BATCH_SIZE,
    model: str = OLLAMA_MODEL,
) -> list[dict]:
    results = []
    for turn in conversation:
        turn_text = f"{turn['speaker']}: {turn['text']}"
        scores = score_turn(turn_text, facets, batch_size=batch_size, model=model)
        results.append({
            "turn_id": turn["turn_id"],
            "speaker": turn["speaker"],
            "text": turn["text"],
            "scores": scores,
        })
    return results
