from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
import os
from typing import List, Optional
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from src.lightgcn_model import LightGCN
from src.faiss_search import FAISSRecommender
from src.feature_engineering import FeatureEngineer
from src.ranking_layer import ReRanker

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
    "feature_engineer": None,
    "reranker": None,
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
    
    # Initialize Feature Engineer and ReRanker (System Design Integration)
    STATE["feature_engineer"] = FeatureEngineer()
    # Need Item Map to extract metadata
    with open('data/processed/item_mapping.json', 'r') as f:
        item_map = json.load(f)
    STATE["feature_engineer"].extract_features(item_map)
    
    STATE["reranker"] = ReRanker()
    
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
    
    # --- ANN RETRIEVAL STAGE (Top-100) ---
    search_k = 100 # Retrieve a larger candidate pool for re-ranking
    scores, indices = STATE["recommender"].recommend(u_emb, k=search_k)
    
    # Filter History
    if request.exclude_history:
        mask = ~np.isin(indices, request.exclude_history)
        indices = indices[mask]
        scores = scores[mask]
    
    # --- RANKING LAYER STAGE (MLP + Metadata) ---
    # Fetch embeddings and metadata for candidates
    candidate_indices = indices
    cand_item_embs = STATE["item_embedding"][candidate_indices]
    cand_metadata = STATE["feature_engineer"].get_feature_tensor(candidate_indices)
    
    # Re-rank using MLP Logic
    final_scores = STATE["reranker"].rank(u_emb, cand_item_embs, cand_metadata, scores)
    
    # Final Selection (Top-K)
    best_indices = np.argsort(-final_scores)[:request.k]
    
    recommendations = []
    for idx_in_top in best_indices:
        real_idx = candidate_indices[idx_in_top]
        recommendations.append(RecommendationResponse(
            item_idx=int(real_idx),
            item_name=STATE["movie_names"].get(str(real_idx), f"Movie {real_idx}"),
            score=float(final_scores[idx_in_top])
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
