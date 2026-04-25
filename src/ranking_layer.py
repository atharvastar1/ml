import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class RankingMLP(nn.Module):
    """Refining Ranking Layer as per System Design Architecture."""
    
    def __init__(self, user_dim, item_dim, feature_dim):
        super(RankingMLP, self).__init__()
        
        # Concat user (64) + item (64) + metadata (19) = 147
        input_dim = user_dim + item_dim + feature_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, user_emb, item_emb, metadata_feat):
        # Concat features
        x = torch.cat([user_emb, item_emb, metadata_feat], dim=1)
        return self.net(x)

class ReRanker:
    """Business Logic layer that combines MLP scores with heuristics."""
    
    def __init__(self, model_state_path="saved_model/ranking_mlp.pt", user_dim=64, item_dim=64, feature_dim=19):
        self.model = RankingMLP(user_dim, item_dim, feature_dim)
        if torch.cuda.is_available(): self.model = self.model.cuda()
        
        if os.path.exists(model_state_path):
            self.model.load_state_dict(torch.load(model_state_path, map_location='cpu'))
        self.model.eval()
        
    def rank(self, user_emb, item_embs, metadata_feats, original_scores):
        """
        Combine GNN neural scores with MLP ranking and business rules.
        """
        with torch.no_grad():
            # Broadcast user_emb to match number of items
            u_batch = user_emb.repeat(len(item_embs), 1)
            
            # 1. MLP Prediction (Ranking Component)
            mlp_scores = self.model(u_batch, item_embs, metadata_feats).squeeze()
            
            # 2. Hybrid Calculation (70% MLP, 30% GNN Graph Score)
            # Normalizing graph scores for hybrid mix
            graph_scores = torch.tensor(original_scores)
            graph_scores = (graph_scores - graph_scores.min()) / (graph_scores.max() - graph_scores.min() + 1e-6)
            
            final_scores = 0.7 * mlp_scores + 0.3 * graph_scores
            
            return final_scores.cpu().numpy()
