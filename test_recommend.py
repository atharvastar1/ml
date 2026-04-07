import torch
import json
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.inference import RecommendationEngine

def test_run():
    print("🎬 Initializing Recommendation Engine Test...")
    
    # Load assets
    user_emb_path = 'saved_model/user_embeddings.pt'
    item_emb_path = 'saved_model/item_embeddings.pt'
    names_path = 'saved_model/movie_names.json'
    
    if not all(os.path.exists(p) for p in [user_emb_path, item_emb_path, names_path]):
        print("❌ Model assets missing. Please ensure training completed successfully.")
        return

    user_emb = torch.load(user_emb_path, map_location='cpu')
    item_emb = torch.load(item_emb_path, map_location='cpu')
    
    with open(names_path, 'r') as f:
        movie_names = json.load(f)
        
    engine = RecommendationEngine(user_emb, item_emb, movie_names)
    
    # Test for User 42
    user_id = 42
    print(f"\n✨ Top 10 Recommendations for User {user_id}:")
    print("-" * 50)
    
    recs = engine.recommend(user_id, k=10)
    
    for i, rec in enumerate(recs, 1):
        print(f"{i}. {rec['item_name']:<40} | Score: {rec['score']:.4f}")
    
    print("-" * 50)

if __name__ == "__main__":
    test_run()
