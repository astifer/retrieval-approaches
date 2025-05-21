import torch
from sentence_transformers import SentenceTransformer
from utils import load_and_split_data, compute_cosine_similarity, get_top_k_predictions
from metrics import batch_recall_at_k, batch_mrr
import json
from tqdm import tqdm
import numpy as np
import argparse

def encode_texts(model: SentenceTransformer, texts: list, batch_size: int = 32, device: str = 'cuda') -> torch.Tensor:
    """
    Encode texts using the E5 model.
    
    Args:
        model: E5 model
        texts: List of texts to encode
        batch_size: Batch size for encoding
        device: Device to use for encoding ('cuda' or 'cpu')
        
    Returns:
        Tensor of shape (n_texts, embedding_dim)
    """
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            batch_embeddings = model.encode(batch, convert_to_tensor=True, device=device)
        embeddings.append(batch_embeddings)
    
    return torch.cat(embeddings, dim=0)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run E5 model baseline')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for encoding (cuda or cpu)')
    parser.add_argument('--max_samples', type=int, default=100000,
                      help='Maximum number of samples to use')
    args = parser.parse_args()
    
    # Load and split data
    print("Loading and splitting data...")
    data = load_and_split_data(max_samples=args.max_samples)
    
    # Prepare data
    train_questions = [item["question"] for item in data["train"]]
    train_answers = [item["answer"] for item in data["train"]]
    test_questions = [item["question"] for item in data["test"]]
    test_answers = [item["answer"] for item in data["test"]]
    
    # Load E5 model
    print("Loading E5 model...")
    model = SentenceTransformer('intfloat/multilingual-e5-base')
    
    # Encode questions and answers
    print(f"Encoding questions and answers on {args.device}...")
    test_q_embeddings = encode_texts(model, test_questions, device=args.device)
    test_a_embeddings = encode_texts(model, test_answers, device=args.device)
    
    # Convert to numpy for similarity computation
    test_q_embeddings = test_q_embeddings.cpu().numpy()
    test_a_embeddings = test_a_embeddings.cpu().numpy()
    
    # Compute similarity matrix
    print("Computing similarity matrix...")
    similarity_matrix = compute_cosine_similarity(test_q_embeddings, test_a_embeddings)
    
    # Get predictions
    print("Getting predictions...")
    predictions = get_top_k_predictions(similarity_matrix, k=10)
    
    # Create ground truth indices (each question is matched with its corresponding answer)
    ground_truth = np.arange(len(test_questions))
    
    # Calculate metrics
    print("Calculating metrics...")
    recall_1 = batch_recall_at_k(ground_truth, predictions, k=1)
    recall_3 = batch_recall_at_k(ground_truth, predictions, k=3)
    recall_10 = batch_recall_at_k(ground_truth, predictions, k=10)
    mrr_score = batch_mrr(ground_truth, predictions)
    
    # Print results
    print("\nResults:")
    print(f"Recall@1: {recall_1:.4f}")
    print(f"Recall@3: {recall_3:.4f}")
    print(f"Recall@10: {recall_10:.4f}")
    print(f"MRR: {mrr_score:.4f}")
    
    # Save results
    results = {
        "recall@1": float(recall_1),
        "recall@3": float(recall_3),
        "recall@10": float(recall_10),
        "mrr": float(mrr_score)
    }
    
    with open("e5_baseline_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main() 