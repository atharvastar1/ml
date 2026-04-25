import pandas as pd
import numpy as np
import torch
import os

class FeatureEngineer:
    """Extracts features for Ranking Layer alignment with System Design."""
    
    def __init__(self, items_path="data/ml-100k/u.item"):
        self.items_path = items_path
        self.movie_features = {}
        self.genre_list = [
            "unknown", "Action", "Adventure", "Animation", "Children's", 
            "Comedy", "Crime", "Documentary", "Drama", "Fantasy", 
            "Film-Noir", "Horror", "Musical", "Mystery", "Romance", 
            "Sci-Fi", "Thriller", "War", "Western"
        ]
        
    def extract_features(self, item_map):
        """Build genre-based feature vectors for all items."""
        if not os.path.exists(self.items_path):
            return None
            
        items_df = pd.read_csv(self.items_path, sep='|', header=None, encoding='latin-1')
        
        # Genre columns start from index 5
        genre_data = items_df.iloc[:, 5:].values
        
        features = {}
        for original_id, idx in item_map.items():
            # MovieLens IDs are 1-indexed
            col_idx = int(original_id) - 1
            if col_idx < len(genre_data):
                features[int(idx)] = genre_data[col_idx].astype(np.float32)
        
        self.movie_features = features
        return features

    def get_feature_tensor(self, item_indices):
        """Batch fetch feature tensors."""
        feature_list = []
        for idx in item_indices:
            feat = self.movie_features.get(int(idx), np.zeros(len(self.genre_list), dtype=np.float32))
            feature_list.append(feat)
            
        return torch.tensor(np.array(feature_list))
