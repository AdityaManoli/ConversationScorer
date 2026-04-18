"""
Ahoum Evaluation API
FastAPI server exposing conversation scoring endpoints.
"""

import json
import os
import csv
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from scorer import score_turn, score_conversation
except ImportError:
    from src.scorer import score_turn, score_conversation

app = FastAPI(
    title="Ahoum Conversation Evaluation API",
    description="Score conversation turns on 300+ psychological/linguistic facets (powered by local Ollama)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load facets at startup
DATA_PATH = Path(__file__).parent.parent / "data" / "facets_cleaned.csv"

def load_facets(only_observable: bool = False) -> list[dict]:
    facets = []
    with open(DATA_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if only_observable and row.get("observable_in_text", "True") == "False":
                continue
            facets.append({
                "facet_id": int(row["facet_id"]),
                "facet_name": row["facet_name"],
                "category": row["category"],
                "observable_in_text": row.get("observable_in_text", "True") == "True",
            })
    return facets

FACETS = load_facets()


# ── Request / Response Models ──────────────────────────────────────────────

class TurnRequest(BaseModel):
    speaker: str
    text: str
    facet_ids: Optional[list[int]] = None  # None = all facets
    batch_size: int = 150

class ConversationRequest(BaseModel):
    turns: list[dict]  # [{turn_id, speaker, text}]
    facet_ids: Optional[list[int]] = None
    batch_size: int = 150

class FacetResponse(BaseModel):
    facet_id: int
    facet_name: str
    score: int
    confidence: float
    rationale: str

class TurnScoreResponse(BaseModel):
    speaker: str
    text: str
    scores: list[FacetResponse]

class ConversationScoreResponse(BaseModel):
    turns: list[dict]
    total_turns: int
    total_facets: int


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "facets_loaded": len(FACETS)}


@app.get("/facets")
def get_facets(category: Optional[str] = None, observable_only: bool = False):
    """List all available facets with optional filtering."""
    facets = FACETS
    if category:
        facets = [f for f in facets if f["category"] == category]
    if observable_only:
        facets = [f for f in facets if f["observable_in_text"]]
    return {"count": len(facets), "facets": facets}


@app.post("/score/turn", response_model=TurnScoreResponse)
def score_single_turn(req: TurnRequest):
    """Score a single conversation turn across all (or specified) facets."""
    facets = FACETS
    if req.facet_ids:
        id_set = set(req.facet_ids)
        facets = [f for f in facets if f["facet_id"] in id_set]
    if not facets:
        raise HTTPException(status_code=400, detail="No matching facets found")

    turn_text = f"{req.speaker}: {req.text}"
    scores = score_turn(turn_text, facets, batch_size=req.batch_size)
    return {"speaker": req.speaker, "text": req.text, "scores": scores}


@app.post("/score/conversation", response_model=ConversationScoreResponse)
def score_full_conversation(req: ConversationRequest):
    """Score every turn in a conversation."""
    facets = FACETS
    if req.facet_ids:
        id_set = set(req.facet_ids)
        facets = [f for f in facets if f["facet_id"] in id_set]
    if not facets:
        raise HTTPException(status_code=400, detail="No matching facets found")
    if not req.turns:
        raise HTTPException(status_code=400, detail="No turns provided")

    results = score_conversation(req.turns, facets, batch_size=req.batch_size)
    return {
        "turns": results,
        "total_turns": len(results),
        "total_facets": len(facets),
    }


@app.get("/health")
def health():
    return {"status": "healthy", "facets": len(FACETS)}
