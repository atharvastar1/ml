import torch
import numpy as np

def precision_at_k(recommendations, ground_truth, k=10):
    top_k = recommendations[:k]
    correct = len(set(top_k) & set(ground_truth))
    return correct / k if k > 0 else 0

def recall_at_k(recommendations, ground_truth, k=10):
    top_k = recommendations[:k]
    correct = [1 if item in ground_truth else 0 for item in top_k]
    return sum(correct) / len(ground_truth) if len(ground_truth) > 0 else 0

def ndcg_at_k(recommendations, ground_truth, k=10):
    top_k = recommendations[:k]
    dcg = 0
    for i, item in enumerate(top_k):
        if item in ground_truth:
            dcg += 1 / np.log2(i + 2)
    
    # Ideal DCG
    idcg = 0
    for i in range(min(len(ground_truth), k)):
        idcg += 1 / np.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0

def evaluate_model(model, test_df, edge_index, num_users, num_items, k=10):
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        user_emb, item_emb = model(edge_index.to(device))
    
    # Ground truth mapping
    user_gt = test_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
    
    precisions = []
    recalls = []
    ndcgs = []
    
    # For speed, we evaluate a subset if too many users
    eval_users = list(user_gt.keys())
    if len(eval_users) > 500:
        np.random.shuffle(eval_users)
        eval_users = eval_users[:500]
        
    for u_idx in eval_users:
        u_emb = user_emb[u_idx]
        scores = torch.matmul(u_emb, item_emb.t())
        
        _, top_k_indices = torch.topk(scores, k + 100) # Get more to allow filtering historical items if needed
        rec_items = top_k_indices.cpu().numpy()
        
        gt = user_gt[u_idx]
        precisions.append(precision_at_k(rec_items, gt, k))
        recalls.append(recall_at_k(rec_items, gt, k))
        ndcgs.append(ndcg_at_k(rec_items, gt, k))
        
    avg_p = np.mean(precisions)
    avg_r = np.mean(recalls)
    avg_n = np.mean(ndcgs)
    
    print(f"Evaluation (k={k}): Precision: {avg_p:.4f}, Recall: {avg_r:.4f}, NDCG: {avg_n:.4f}")
    return avg_p, avg_r, avg_n
