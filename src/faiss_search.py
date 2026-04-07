import faiss
import numpy as np
import torch

class FAISSRecommender:
    """Fast recommendation using FAISS (Inner Product for Dot Product)"""
    
    def __init__(self, item_embedding):
        # FAISS works with numpy float32
        self.item_embedding = item_embedding.cpu().detach().numpy().astype('float32')
        self.embedding_dim = item_embedding.shape[1]
        
        # IndexFlatIP is for Inner Product (useful for Maximum Inner Product Search / MIPS)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Normalize items if using Cosine Similarity instead of Dot Product
        # faiss.normalize_L2(self.item_embedding)
        
        self.index.add(self.item_embedding)
    
    def recommend(self, user_embedding, k=10):
        # Convert user embedding to numpy float32
        if isinstance(user_embedding, torch.Tensor):
            query = user_embedding.cpu().detach().numpy().reshape(1, -1).astype('float32')
        else:
            query = user_embedding.reshape(1, -1).astype('float32')
            
        distances, indices = self.index.search(query, k)
        
        return distances[0], indices[0].astype(int)
