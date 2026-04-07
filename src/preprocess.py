import pandas as pd
import numpy as np
import os
import json

def convert_to_implicit(df, rating_threshold=3.0):
    """
    Convert explicit ratings to implicit feedback.
    Logic: rating >= threshold -> implicit feedback of 1
    """
    print(f"Converting ratings (threshold={rating_threshold}) to implicit feedback...")
    implicit_df = df.copy()
    implicit_df['interaction'] = (implicit_df['rating'] >= rating_threshold).astype(int)
    
    # Keep only positive interactions
    implicit_df = implicit_df[implicit_df['interaction'] == 1]
    implicit_df = implicit_df[['user_id', 'item_id']]
    
    print(f"Implicit interactions: {len(implicit_df)}")
    print(f"Unique users: {implicit_df['user_id'].nunique()}")
    print(f"Unique items: {implicit_df['item_id'].nunique()}")
    
    return implicit_df

def create_user_item_mapping(df):
    """
    Create mappings from original IDs to sequential indices.
    """
    unique_users = df['user_id'].unique()
    unique_items = df['item_id'].unique()
    
    user_mapping = {int(u): int(idx) for idx, u in enumerate(sorted(unique_users))}
    item_mapping = {int(i): int(idx) for idx, i in enumerate(sorted(unique_items))}
    
    # Remap the dataframe
    df['user_idx'] = df['user_id'].map(user_mapping)
    df['item_idx'] = df['item_id'].map(item_mapping)
    
    return df, user_mapping, item_mapping

def save_mappings(user_map, item_map, path="data/processed"):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "user_mapping.json"), "w") as f:
        json.dump(user_map, f)
    with open(os.path.join(path, "item_mapping.json"), "w") as f:
        json.dump(item_map, f)
    print(f"Mappings saved to {path}")

if __name__ == "__main__":
    # Test loading
    data_path = "data/ml-100k/u.data"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
        implicit_df = convert_to_implicit(df)
        implicit_df, user_map, item_map = create_user_item_mapping(implicit_df)
        save_mappings(user_map, item_map)
    else:
        print(f"Data file {data_path} not found. Run download_data.py first.")
