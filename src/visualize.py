import torch
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
import json
import os

def visualize_embeddings():
    print("🎨 Generating Embedding Visualization...")
    
    # Load assets
    item_emb = torch.load('saved_model/item_embeddings.pt', map_location='cpu').numpy()
    
    with open('saved_model/movie_names.json', 'r') as f:
        movie_names = json.load(f)
        
    # Load movie genres for coloring (if available)
    items_path = "data/ml-100k/u.item"
    if os.path.exists(items_path):
        genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                      'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                      'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        items_df = pd.read_csv(items_path, sep='|', header=None, encoding='latin-1', 
                               names=['id', 'title', 'date', 'v1', 'url'] + genre_cols)
        
        # Get primary genre for each item
        def get_primary_genre(row):
            for g in genre_cols:
                if row[g] == 1: return g
            return 'Other'
        
        items_df['primary_genre'] = items_df.apply(get_primary_genre, axis=1)
        
        # Map item indices to genres
        with open('data/processed/item_mapping.json', 'r') as f:
            item_map = json.load(f)
        
        # item_map: {orig_id: idx}
        idx_to_genre = {}
        for orig_id, idx in item_map.items():
            try:
                # pandas id is 1-indexed in file, and we loaded with names
                genre = items_df.loc[items_df['id'] == int(orig_id), 'primary_genre'].values[0]
                idx_to_genre[int(idx)] = genre
            except:
                idx_to_genre[int(idx)] = 'Unknown'
    else:
        idx_to_genre = {i: 'Unknown' for i in range(len(item_emb))}

    # Run T-SNE
    print("⏱️ Running T-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(item_emb)
    
    # Prepare DataFrame for Plotly
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'title': [movie_names.get(str(i), f"Movie {i}") for i in range(len(item_emb))],
        'genre': [idx_to_genre.get(i, 'Unknown') for i in range(len(item_emb))]
    })
    
    # Create Interactive Plot
    fig = px.scatter(df, x='x', y='y', color='genre', hover_name='title',
                     title='LightGCN Movie Embedding Space (T-SNE)',
                     template='plotly_dark',
                     color_discrete_sequence=px.colors.qualitative.Prism)
    
    fig.update_layout(
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        legend_title="Dominant Genre"
    )
    
    # Save as static PNG and interactive HTML
    os.makedirs('static', exist_ok=True)
    fig.write_html('static/embedding_map.html')
    print("✅ Visualization saved to static/embedding_map.html")
    
    return df

if __name__ == "__main__":
    visualize_embeddings()
