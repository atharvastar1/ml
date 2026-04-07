import torch
import pandas as pd
import json
import os
import numpy as np
from pathlib import Path

# Import local modules
from src.download_data import download_movielens_small
from src.preprocess import convert_to_implicit, create_user_item_mapping, save_mappings
from src.graph_builder import build_edge_index, build_undirected_edge_index
from src.lightgcn_model import LightGCN
from src.training import train_lightgcn
from src.evaluation import evaluate_model

def run_pipeline():
    print("🚀 Starting LightGCN Recommendation Pipeline...")
    
    # 1. Download
    download_movielens_small(base_path="data")
    
    # 2. Preprocess
    data_path = "data/ml-100k/u.data"
    df = pd.read_csv(data_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    
    implicit_df = convert_to_implicit(df, rating_threshold=3.5)
    implicit_df, user_map, item_map = create_user_item_mapping(implicit_df)
    
    # Save processed data and mappings
    os.makedirs("data/processed", exist_ok=True)
    implicit_df.to_csv("data/processed/implicit_interactions.csv", index=False)
    save_mappings(user_map, item_map, path="data/processed")
    
    num_users = len(user_map)
    num_items = len(item_map)
    print(f"📊 Dataset Stats: {num_users} users, {num_items} items, {len(implicit_df)} interactions")
    
    # 3. Graph
    edge_index = build_edge_index(implicit_df, num_users, num_items)
    undirected_edge_index = build_undirected_edge_index(edge_index)
    
    # 4. Train/Test Split
    train_df = implicit_df.sample(frac=0.8, random_state=42)
    test_df = implicit_df.drop(train_df.index)
    
    # 5. Model
    model = LightGCN(num_users, num_items, embedding_dim=64, num_layers=3)
    
    # 6. Train
    model = train_lightgcn(
        model, train_df, undirected_edge_index,
        num_users, num_items, epochs=40, batch_size=2048, lr=0.001
    )
    
    # 7. Evaluate
    evaluate_model(model, test_df, undirected_edge_index, num_users, num_items, k=10)
    
    # 8. Save Model & Metadata
    print("💾 Saving model and artifacts...")
    os.makedirs("saved_model", exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_users': num_users,
        'num_items': num_items,
        'embedding_dim': 64,
        'num_layers': 3
    }, 'saved_model/lightgcn_model.pt')
    
    torch.save(undirected_edge_index, 'saved_model/edge_index.pt')
    
    # Generate and save final embeddings
    model.eval()
    with torch.no_grad():
        final_user_emb, final_item_emb = model(undirected_edge_index)
        torch.save(final_user_emb, 'saved_model/user_embeddings.pt')
        torch.save(final_item_emb, 'saved_model/item_embeddings.pt')
    
    # Map Item Indices to Movie Names
    items_path = "data/ml-100k/u.item"
    items_df = pd.read_csv(items_path, sep='|', header=None, encoding='latin-1')
    
    # movie_names[item_idx] = movie_title
    movie_names = {}
    for original_id, idx in item_map.items():
        # MovieLens IDs are 1-indexed
        title = items_df.iloc[original_id-1, 1]
        movie_names[int(idx)] = title
        
    with open('saved_model/movie_names.json', 'w') as f:
        json.dump(movie_names, f)
        
    print("✅ Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()
