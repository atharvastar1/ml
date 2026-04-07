import torch
import numpy as np

def check_training_quality():
    print("🧠 Analyzing Model Training Quality...")
    
    # Load assets
    user_emb = torch.load('saved_model/user_embeddings.pt', map_location='cpu')
    item_emb = torch.load('saved_model/item_embeddings.pt', map_location='cpu')
    
    # 1. Check for collapse (all embeddings same)
    user_variance = user_emb.var(dim=0).mean().item()
    item_variance = item_emb.var(dim=0).mean().item()
    
    # 2. Check embedding norms
    user_norms = torch.norm(user_emb, p=2, dim=1).mean().item()
    item_norms = torch.norm(item_emb, p=2, dim=1).mean().item()
    
    # 3. Check for NaNs
    has_nan = torch.isnan(user_emb).any().item() or torch.isnan(item_emb).any().item()
    
    print(f"\n📊 Diagnostic Metrics:")
    print(f"- User Embedding Variance: {user_variance:.6f} {'✅' if user_variance > 0.001 else '⚠️ Low variance'}")
    print(f"- Item Embedding Variance: {item_variance:.6f} {'✅' if item_variance > 0.001 else '⚠️ Low variance'}")
    print(f"- Base Avg User Norm: {user_norms:.4f}")
    print(f"- Base Avg Item Norm: {item_norms:.4f}")
    print(f"- NaN presence: {'❌ FOUND!' if has_nan else '✅ Clean'}")

    # 4. Interaction Overlap Check
    # (High score for items user already liked = good training)
    print("\n🧐 Validating Implicit Logic...")
    # Get a random user's history
    import pandas as pd
    interactions = pd.read_csv('data/processed/implicit_interactions.csv')
    sample_user = interactions['user_idx'].iloc[0]
    user_history = interactions[interactions['user_idx'] == sample_user]['item_idx'].tolist()
    
    u_vec = user_emb[sample_user]
    pos_item_scores = torch.matmul(u_vec, item_emb[user_history].t()).mean().item()
    
    # Get random negative items
    all_indices = np.arange(item_emb.shape[0])
    neg_indices = np.random.choice([i for i in all_indices if i not in user_history], len(user_history))
    neg_item_scores = torch.matmul(u_vec, item_emb[neg_indices].t()).mean().item()
    
    print(f"- Mean Positive Score (Liked): {pos_item_scores:.4f}")
    print(f"- Mean Negative Score (Unseen): {neg_item_scores:.4f}")
    print(f"- Ratio (Pos/Neg): {pos_item_scores/neg_item_scores:.2f}x {'✅' if pos_item_scores > neg_item_scores else '❌ Training failed to separate'}")

if __name__ == "__main__":
    check_training_quality()
