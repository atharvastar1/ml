import torch
import numpy as np

class RecommendationEngine:
    def __init__(self, user_embedding, item_embedding, item_names=None):
        self.user_embedding = user_embedding
        self.item_embedding = item_embedding
        self.item_names = item_names
    
    def recommend(self, user_idx, k=10, exclude_history=None):
        user_emb = self.user_embedding[user_idx]
        scores = torch.matmul(user_emb, self.item_embedding.t())
        
        scores = scores.cpu().detach().numpy()
        
        if exclude_history is not None:
            scores[list(exclude_history)] = -np.inf
            
        top_k_indices = np.argsort(-scores)[:k]
        top_k_scores = scores[top_k_indices]
        
        recommendations = []
        for idx, score in zip(top_k_indices, top_k_scores):
            item_name = self.item_names.get(str(idx), f"Item {idx}") if self.item_names else f"Item {idx}"
            recommendations.append({
                'item_idx': int(idx),
                'item_name': item_name,
                'score': float(score)
            })
            
        return recommendations

    def mmr_recommend(self, user_idx, k=10, lambda_param=0.5, history_indices=None):
        """
        Maximal Marginal Relevance (MMR) for diversity.
        Score = lambda * similarity(u, i) - (1-lambda) * max(similarity(i, j) for j in result_set)
        """
        # 1. Get initial scores
        user_emb = self.user_embedding[user_idx]
        scores = torch.matmul(user_emb, self.item_embedding.t())
        scores = scores.cpu().detach().numpy()
        
        # 2. Candidate pool (top 100)
        candidate_indices = np.argsort(-scores)[:100]
        
        # 3. Iterative Selection
        selected_indices = [candidate_indices[0]] # Start with best match
        remaining_indices = list(candidate_indices[1:])
        
        while len(selected_indices) < k and remaining_indices:
            mmr_scores = []
            for candidate in remaining_indices:
                relevance = scores[candidate]
                cand_emb = self.item_embedding[candidate]
                sel_embs = self.item_embedding[selected_indices]
                similarities = torch.matmul(sel_embs, cand_emb.t())
                redundancy = torch.max(similarities).item()
                mmr_val = lambda_param * relevance - (1 - lambda_param) * redundancy
                mmr_scores.append(mmr_val)
            
            best_idx = np.argmax(mmr_scores)
            selected_indices.append(remaining_indices.pop(best_idx))
            
        # Format output with Explainability
        recommendations = []
        for idx in selected_indices:
            item_name = self.item_names.get(str(idx), f"Item {idx}") if self.item_names else f"Item {idx}"
            # ADD XAI EXPLANATION
            explanation = ""
            if history_indices is not None:
                explanation = self.explain_recommendation(user_idx, idx, history_indices)
            
            recommendations.append({
                'item_idx': int(idx),
                'item_name': item_name,
                'score': float(scores[idx]),
                'explanation': explanation
            })
            
        return recommendations

    def explain_recommendation(self, user_idx, item_idx, history_indices, top_n=2):
        """
        Explain why an item was recommended by finding sources in user history.
        """
        if not history_indices:
            return "Based on global cinema trends."
            
        target_emb = self.item_embedding[item_idx]
        history_embs = self.item_embedding[list(history_indices)]
        
        # Dot product for similarity
        similarities = torch.matmul(history_embs, target_emb.t())
        
        # Get top-N
        vals, indices = torch.topk(similarities, min(top_n, len(history_indices)))
        
        source_titles = []
        for idx in indices:
            orig_item_idx = list(history_indices)[idx.item()]
            title = self.item_names.get(str(orig_item_idx), f"Movie {orig_item_idx}")
            # Clean title
            title = title.split('(')[0].strip()
            source_titles.append(f"'{title}'")
            
        return f"Because you loved {', '.join(source_titles)}."

    def get_popular_items(self, k=10):
        """
        Return the most interacted items (fallback for Cold Start).
        For this demo, we use a curated list of ML-100k hits.
        """
        popular_indices = [49, 257, 99, 180, 293, 285, 287, 0, 120, 241]
        recommendations = []
        for idx in popular_indices[:k]:
            item_name = self.item_names.get(str(idx), f"Popular Movie {idx}")
            recommendations.append({
                'item_idx': int(idx),
                'item_name': item_name,
                'score': 1.0,
                'is_popular': True
            })
        return recommendations

    def update_user_embedding_from_likes(self, liked_item_indices):
        """
        Compute a temporary user embedding based on a set of liked items.
        In LightGCN, a user's embedding is effectively the average of their neighbors' embeddings.
        """
        if not liked_item_indices:
            return None
            
        # Extract embeddings of liked items
        liked_embs = self.item_embedding[list(liked_item_indices)]
        
        # Average them to create a new user vector
        new_user_emb = torch.mean(liked_embs, dim=0)
        
        return new_user_emb

    def get_similar_items(self, item_idx, k=10):
        """
        Find items with similar embeddings to the query item (Neural Similarity).
        """
        query_emb = self.item_embedding[item_idx]
        scores = torch.matmul(query_emb, self.item_embedding.t())
        scores = scores.cpu().detach().numpy()
        
        # Exclude the query item itself
        scores[item_idx] = -np.inf
        
        top_k_indices = np.argsort(-scores)[:k]
        top_k_scores = scores[top_k_indices]
        
        results = []
        for idx, score in zip(top_k_indices, top_k_scores):
            item_name = self.item_names.get(str(idx), f"Item {idx}") if self.item_names else f"Item {idx}"
            results.append({
                'item_idx': int(idx),
                'item_name': item_name,
                'score': float(score),
                'explanation': f"Because it shares a latent neighborhood with '{self.item_names.get(str(item_idx), 'Query')}'."
            })
        return results
