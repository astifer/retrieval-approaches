import numpy as np
from typing import List, Union

def recall_at_k(target: Union[int, List[int], np.integer], predict: List[int], k: int) -> float:
    """
    Calculate Recall@K metric.
    
    Args:
        target: ID of the correct document or list of correct document IDs
        predict: List of predicted document IDs sorted by relevance
        k: Number of top documents to consider
        
    Returns:
        float: Recall@K score
    """
    # Convert numpy integer to Python int if needed
    if isinstance(target, np.integer):
        target = int(target)
    
    if isinstance(target, int):
        target = [target]
    
    # Convert to sets for easier intersection calculation
    target_set = set(target)
    predict_set = set(predict[:k])
    
    # Calculate intersection size
    intersection = len(target_set.intersection(predict_set))
    
    # Calculate recall
    recall = intersection / len(target_set) if target_set else 0.0
    
    return recall

def mrr(target: Union[int, List[int], np.integer], predict: List[int]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) metric.
    
    Args:
        target: ID of the correct document or list of correct document IDs
        predict: List of predicted document IDs sorted by relevance
        
    Returns:
        float: MRR score
    """
    # Convert numpy integer to Python int if needed
    if isinstance(target, np.integer):
        target = int(target)
    
    if isinstance(target, int):
        target = [target]
    
    # Find the rank of the first correct document
    for rank, doc_id in enumerate(predict, 1):
        if doc_id in target:
            return 1.0 / rank
    
    return 0.0

def batch_recall_at_k(targets: List[Union[int, List[int], np.integer]], 
                     predicts: List[List[int]], 
                     k: int) -> float:
    """
    Calculate Recall@K metric for a batch of predictions.
    
    Args:
        targets: List of correct document IDs for each query
        predicts: List of predicted document IDs for each query
        k: Number of top documents to consider
        
    Returns:
        float: Average Recall@K score
    """
    recalls = [recall_at_k(t, p, k) for t, p in zip(targets, predicts)]
    return np.mean(recalls)

def batch_mrr(targets: List[Union[int, List[int], np.integer]], 
             predicts: List[List[int]]) -> float:
    """
    Calculate MRR metric for a batch of predictions.
    
    Args:
        targets: List of correct document IDs for each query
        predicts: List of predicted document IDs for each query
        
    Returns:
        float: Average MRR score
    """
    mrrs = [mrr(t, p) for t, p in zip(targets, predicts)]
    return np.mean(mrrs) 