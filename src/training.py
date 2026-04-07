import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm

class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking Loss"""
    def forward(self, pos_scores, neg_scores):
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8)
        return loss.mean()

def train_lightgcn(model, train_df, edge_index, num_users, num_items, 
                   epochs=50, batch_size=2048, lr=0.001):
    """
    Train LightGCN model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model.to(device)
    edge_index = edge_index.to(device)
    
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = BPRLoss()
    
    user_item_pairs = train_df[['user_idx', 'item_idx']].values
    all_items = np.arange(num_items)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Shuffle
        indices = np.random.permutation(len(user_item_pairs))
        
        for i in range(0, len(user_item_pairs), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_pairs = user_item_pairs[batch_indices]
            
            u_ids = torch.tensor(batch_pairs[:, 0], dtype=torch.long).to(device)
            p_ids = torch.tensor(batch_pairs[:, 1], dtype=torch.long).to(device)
            n_ids = torch.tensor(np.random.choice(all_items, len(u_ids)), dtype=torch.long).to(device)
            
            # Forward pass
            user_emb, item_emb = model(edge_index)
            
            pos_scores = (user_emb[u_ids] * item_emb[p_ids]).sum(dim=1)
            neg_scores = (user_emb[u_ids] * item_emb[n_ids]).sum(dim=1)
            
            loss = loss_fn(pos_scores, neg_scores)
            
            # L2 Regularization (optional but recommended)
            reg_loss = 1e-4 * (user_emb[u_ids].norm(2).pow(2) + 
                               item_emb[p_ids].norm(2).pow(2) + 
                               item_emb[n_ids].norm(2).pow(2)) / len(u_ids)
            
            total_loss_val = loss + reg_loss
            
            optimizer.zero_grad()
            total_loss_val.backward()
            optimizer.step()
            
            total_loss += total_loss_val.item()
            
        avg_loss = total_loss / (len(user_item_pairs) // batch_size + 1)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
    return model
