import pandas as pd
import json
import os

def generate_profiles():
    # Load interactions
    interactions = pd.read_csv('data/processed/implicit_interactions.csv')
    
    # Load movie names and genres
    items_path = "data/ml-100k/u.item"
    items_df = pd.read_csv(items_path, sep='|', header=None, encoding='latin-1')
    genres = ['unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    
    # Map item_id to genres
    item_genres = {}
    for i, row in items_df.iterrows():
        active_genres = [genres[g] for g in range(19) if row[5+g] == 1]
        item_genres[i+1] = active_genres # Original ID is 1-indexed

    # Get User-ID map to translate back if needed, but here we use the processed user_id
    # Wait, interactions.csv already uses the re-indexed user_id (0-indexed)
    # We need to link re-indexed user_id to their original interactions to find genres.
    
    # Actually, let's just use the top 10 most active users to showcase.
    top_users = interactions['user_id'].value_counts().head(20).index.tolist()
    
    profiles = []
    for uid in top_users:
        user_items = interactions[interactions['user_id'] == uid]['item_id'].tolist()
        
        # This item_id is 0-indexed re-map. We need the original ID to get genres.
        # Let's load the map
        with open('data/processed/item_mapping.json', 'r') as f:
            item_map = json.load(f)
            # Reverse map: index -> original_id
            rev_item_map = {int(v): int(k) for k, v in item_map.items()}
        
        user_genres = []
        for iid in user_items:
            orig_id = rev_item_map.get(iid)
            if orig_id in item_genres:
                user_genres.extend(item_genres[orig_id])
        
        from collections import Counter
        top_genre = Counter(user_genres).most_common(1)[0][0] if user_genres else "General"
        
        profiles.append({
            "id": int(uid),
            "label": f"User #{uid}",
            "desc": f"{top_genre} Enthusiast",
            "count": len(user_items)
        })

    with open('saved_model/user_profiles.json', 'w') as f:
        json.dump(profiles, f)
    print("Generated user_profiles.json")

if __name__ == "__main__":
    generate_profiles()
