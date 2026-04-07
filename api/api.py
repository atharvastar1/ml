from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
import os
from typing import List, Optional
import numpy as np

# Local imports
from src.lightgcn_model import LightGCN
from src.faiss_search.py import FAISSRecommender # I realized I put faiss_search.py in src/ but I'll import it correctly

# Actually, I'll import from src directly
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.lightgcn_model import LightGCN
from src.faiss_search import FAISSRecommender

app = FastAPI(
    title="LightGCN Recommendation API",
    description="High-performance Graph Neural Network based movie recommendation system",
    version="1.0.0"
)

# Global variables for model storage
STATE = {
    "model": None,
    "user_embedding": None,
    "item_embedding": None,
    "recommender": None,
    "movie_names": None,
    "num_users": 0,
    "num_items": 0
}

class RecommendationRequest(BaseModel):
    user_id: int
    k: int = 10
    exclude_history: Optional[List[int]] = None

class RecommendationResponse(BaseModel):
    item_idx: int
    item_name: str
    score: float

class RecommendationOutput(BaseModel):
    user_id: int
    recommendations: List[RecommendationResponse]

@app.on_event("startup")
async def startup_event():
    model_path = 'saved_model/lightgcn_model.pt'
    names_path = 'saved_model/movie_names.json'
    
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found at {model_path}. Please run main.py first.")
        return

    print("📜 Loading model and data...")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    STATE["num_users"] = checkpoint['num_users']
    STATE["num_items"] = checkpoint['num_items']
    
    # Load embeddings instead of recomputing for speed
    STATE["user_embedding"] = torch.load('saved_model/user_embeddings.pt', map_location='cpu')
    STATE["item_embedding"] = torch.load('saved_model/item_embeddings.pt', map_location='cpu')
    
    # Initialize FAISS
    STATE["recommender"] = FAISSRecommender(STATE["item_embedding"])
    
    # Load movie titles
    with open(names_path, 'r') as f:
        STATE["movie_names"] = json.load(f)
        
    print("✅ System ready!")

@app.get("/")
def read_root():
    return {"message": "Welcome to the LightGCN Recommendation API. Use /recommend for results.", "status": "active"}

@app.post("/recommend", response_model=RecommendationOutput)
def get_recommendations(request: RecommendationRequest):
    if STATE["user_embedding"] is None:
        raise HTTPException(status_code=503, detail="Model assets not loaded")
    
    user_idx = request.user_id
    if user_idx < 0 or user_idx >= STATE["num_users"]:
        raise HTTPException(status_code=400, detail=f"Invalid user_id. Must be between 0 and {STATE['num_users']-1}")
    
    # Get user embedding
    u_emb = STATE["user_embedding"][user_idx]
    
    # Get top-K using FAISS (get more for filtering)
    search_k = request.k + (len(request.exclude_history) if request.exclude_history else 0)
    scores, indices = STATE["recommender"].recommend(u_emb, k=search_k)
    
    recommendations = []
    for idx, score in zip(indices, scores):
        if request.exclude_history and int(idx) in request.exclude_history:
            continue
            
        recommendations.append(RecommendationResponse(
            item_idx=int(idx),
            item_name=STATE["movie_names"].get(str(idx), f"Movie {idx}"),
            score=float(score)
        ))
        
        if len(recommendations) == request.k:
            break
            
    return RecommendationOutput(
        user_id=user_idx,
        recommendations=recommendations
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
