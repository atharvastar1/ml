import torch
import numpy as np
import pandas as pd

def build_edge_index(df, num_users, num_items):
    """
    Convert interactions to edge_index format (PyTorch Geometric standard).
    Item indices shifted by num_users to distinguish from user indices.
    """
    # Get user and item indices
    user_indices = df['user_idx'].values
    item_indices = df['item_idx'].values + num_users  # Shift items
    
    # Create edge_index: [2, num_interactions]
    edge_index = torch.tensor(
        np.stack([user_indices, item_indices]),
        dtype=torch.long
    )
    
    print(f"Edge index shape: {edge_index.shape}")
    return edge_index

def build_undirected_edge_index(edge_index):
    """
    Make graph undirected (add reverse edges).
    """
    # Reverse edges (item -> user)
    reverse_edge_index = torch.stack([edge_index[1], edge_index[0]])
    
    # Concatenate forward and reverse
    undirected_edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
    
    print(f"Undirected edge index shape: {undirected_edge_index.shape}")
    return undirected_edge_index

if __name__ == "__main__":
    print("Graph builder module loaded.")
