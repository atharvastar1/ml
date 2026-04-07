import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGCN(nn.Module):
    """
    Light Graph Convolution Network for Recommendation.
    """
    
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super(LightGCN, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize with Xavier uniform
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def forward(self, edge_index):
        # Initial embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        all_embeddings = torch.cat([user_emb, item_emb], dim=0)
        embs = [all_embeddings]
        
        # Propagation
        for layer in range(self.num_layers):
            all_embeddings = self._propagate(all_embeddings, edge_index)
            embs.append(all_embeddings)
        
        # Average embeddings from all layers
        final_embs = torch.stack(embs, dim=0).mean(dim=0)
        
        final_user_embedding, final_item_embedding = torch.split(final_embs, [self.num_users, self.num_items])
        
        return final_user_embedding, final_item_embedding
    
    def _propagate(self, x, edge_index):
        """
        Message passing using normalized adjacency matrix logic.
        Efficient implementation using scatter_add.
        """
        row, col = edge_index
        
        # Compute degrees for normalization
        # In LightGCN: E = D^(-1/2) A D^(-1/2)
        # For simplicity in the guide logic: average (D^-1 A)
        
        # We need to compute degree of target nodes (col)
        deg = torch.zeros(x.size(0), device=x.device)
        deg.scatter_add_(0, col, torch.ones(len(col), device=x.device))
        
        # Degree normalization factor 1/sqrt(deg(i) * deg(j)) is more standard,
        # but the guide says "average instead of sum".
        # So we'll do: aggregated[target] = sum(neighbor_embeddings) / degree[target]
        
        deg_inv = 1.0 / deg
        deg_inv[torch.isinf(deg_inv)] = 0
        
        # Aggregate messages
        # aggregated[col[i]] += x[row[i]]
        out = torch.zeros_like(x)
        out.scatter_add_(0, col.unsqueeze(1).expand(-1, x.size(1)), x[row])
        
        # Normalize
        out = out * deg_inv.unsqueeze(1)
        
        return out
    
    def compute_scores(self, user_embedding, item_embedding):
        return torch.matmul(user_embedding, item_embedding.t())
