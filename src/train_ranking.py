import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
import json

from src.ranking_layer import RankingMLP
from src.feature_engineering import FeatureEngineer

class InteractionDataset(Dataset):
    def __init__(self, user_ids, item_ids, labels, user_embs, item_embs, metadata_feats):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.labels = labels
        self.user_embs = user_embs
        self.item_embs = item_embs
        self.metadata_feats = metadata_feats
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        u_idx = self.user_ids[idx]
        i_idx = self.item_ids[idx]
        
        return (
            self.user_embs[u_idx], 
            self.item_embs[i_idx], 
            self.metadata_feats[i_idx], 
            torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        )

def train_ranking_layer():
    print("🧠 Starting Ranking Layer (MLP) Training...")
    
    # 1. Load data
    df = pd.read_csv('data/processed/implicit_interactions.csv')
    user_embs = torch.load('saved_model/user_embeddings.pt', map_location='cpu')
    item_embs = torch.load('saved_model/item_embeddings.pt', map_location='cpu')
    
    with open('data/processed/item_mapping.json', 'r') as f:
        item_mapping = json.load(f)
        
    # 2. Extract metadata features
    fe = FeatureEngineer()
    movie_features = fe.extract_features(item_mapping)
    metadata_tensor = fe.get_feature_tensor(range(len(item_embs)))
    
    # 3. Create training samples (Postive interactions)
    # Using 'user_idx' and 'item_idx' which match embeddings
    pos_users = df['user_idx'].values
    pos_items = df['item_idx'].values
    labels = np.ones(len(pos_users))
    
    # Simple negative sampling
    num_neg = len(pos_users)
    neg_users = np.random.randint(0, len(user_embs), num_neg)
    neg_items = np.random.randint(0, len(item_embs), num_neg)
    neg_labels = np.zeros(num_neg)
    
    all_users = np.concatenate([pos_users, neg_users])
    all_items = np.concatenate([pos_items, neg_items])
    all_labels = np.concatenate([labels, neg_labels])
    
    dataset = InteractionDataset(all_users, all_items, all_labels, user_embs, item_embs, metadata_tensor)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    # 4. Initialize MLP
    model = RankingMLP(user_dim=64, item_dim=64, feature_dim=19)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # 5. Training Loop
    model.train()
    for epoch in range(10):
        total_loss = 0
        for u_e, i_e, m_f, label in loader:
            optimizer.zero_grad()
            preds = model(u_e, i_e, m_f).squeeze()
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/10 | Loss: {total_loss/len(loader):.4f}")
        
    # 6. Save Model
    os.makedirs("saved_model", exist_ok=True)
    torch.save(model.state_dict(), 'saved_model/ranking_mlp.pt')
    print("✅ Ranking Layer Trained and Saved!")

if __name__ == "__main__":
    train_ranking_layer()
