import torch
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
from utils import (
    load_and_split_data,
    create_contrastive_pairs,
    create_triplet_pairs,
    compute_cosine_similarity,
    get_top_k_predictions
)
from metrics import batch_recall_at_k, batch_mrr
import json
from tqdm import tqdm
import numpy as np
import argparse

def train_contrastive_model(
    model: SentenceTransformer,
    train_questions: list,
    train_answers: list,
    n_pairs: int = 10000,
    batch_size: int = 32,
    epochs: int = 3,
    device: str = 'cuda'
) -> SentenceTransformer:
    """
    Train E5 model using Contrastive Loss.
    
    Args:
        model: E5 model
        train_questions: List of training questions
        train_answers: List of training answers
        n_pairs: Number of pairs to create
        batch_size: Batch size for training
        epochs: Number of training epochs
        device: Device to use for training ('cuda' or 'cpu')
        
    Returns:
        Trained model
    """
    # Create contrastive pairs
    print("Creating contrastive pairs...")
    pairs_questions, pairs_answers, labels = create_contrastive_pairs(
        train_questions, train_answers, n_pairs
    )
    
    # Create training examples
    train_examples = [
        InputExample(texts=[q, a], label=float(l))
        for q, a, l in zip(pairs_questions, pairs_answers, labels)
    ]
    
    # Create data loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Define loss function
    train_loss = losses.ContrastiveLoss(model)
    
    # Train model
    print(f"Training model with Contrastive Loss on {device}...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        show_progress_bar=True,
        device=device
    )
    
    return model

def train_triplet_model(
    model: SentenceTransformer,
    train_questions: list,
    train_answers: list,
    n_triplets: int = 10000,
    batch_size: int = 32,
    epochs: int = 3,
    device: str = 'cuda'
) -> SentenceTransformer:
    """
    Train E5 model using Triplet Loss.
    
    Args:
        model: E5 model
        train_questions: List of training questions
        train_answers: List of training answers
        n_triplets: Number of triplets to create
        batch_size: Batch size for training
        epochs: Number of training epochs
        device: Device to use for training ('cuda' or 'cpu')
        
    Returns:
        Trained model
    """
    # Create triplet pairs
    print("Creating triplet pairs...")
    anchors, positives, negatives = create_triplet_pairs(
        train_questions, train_answers, n_triplets
    )
    
    # Create training examples
    train_examples = [
        InputExample(texts=[a, p, n])
        for a, p, n in zip(anchors, positives, negatives)
    ]
    
    # Create data loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Define loss function
    train_loss = losses.TripletLoss(model)
    
    # Train model
    print(f"Training model with Triplet Loss on {device}...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        show_progress_bar=True,
        device=device
    )
    
    return model

def evaluate_model(
    model: SentenceTransformer,
    test_questions: list,
    test_answers: list,
    batch_size: int = 32,
    device: str = 'cuda'
) -> dict:
    """
    Evaluate model on test data.
    
    Args:
        model: Trained model
        test_questions: List of test questions
        test_answers: List of test answers
        batch_size: Batch size for encoding
        device: Device to use for evaluation ('cuda' or 'cpu')
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Encode questions and answers
    print(f"Encoding test data on {device}...")
    test_q_embeddings = model.encode(test_questions, batch_size=batch_size, show_progress_bar=True, device=device)
    test_a_embeddings = model.encode(test_answers, batch_size=batch_size, show_progress_bar=True, device=device)
    
    # Compute similarity matrix
    print("Computing similarity matrix...")
    similarity_matrix = compute_cosine_similarity(test_q_embeddings, test_a_embeddings)
    
    # Get predictions
    print("Getting predictions...")
    predictions = get_top_k_predictions(similarity_matrix, k=10)
    
    # Create ground truth indices
    ground_truth = np.arange(len(test_questions))
    
    # Calculate metrics
    print("Calculating metrics...")
    recall_1 = batch_recall_at_k(ground_truth, predictions, k=1)
    recall_3 = batch_recall_at_k(ground_truth, predictions, k=3)
    recall_10 = batch_recall_at_k(ground_truth, predictions, k=10)
    mrr_score = batch_mrr(ground_truth, predictions)
    
    return {
        "recall@1": float(recall_1),
        "recall@3": float(recall_3),
        "recall@10": float(recall_10),
        "mrr": float(mrr_score)
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train E5 model with different loss functions')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for training (cuda or cpu)')
    args = parser.parse_args()
    
    # Load and split data
    print("Loading and splitting data...")
    data = load_and_split_data()
    
    # Prepare data
    train_questions = [item["question"] for item in data["train"]]
    train_answers = [item["answer"] for item in data["train"]]
    test_questions = [item["question"] for item in data["test"]]
    test_answers = [item["answer"] for item in data["test"]]
    
    # Train and evaluate Contrastive Loss model
    print("\nTraining with Contrastive Loss...")
    contrastive_model = SentenceTransformer('intfloat/multilingual-e5-base')
    contrastive_model = train_contrastive_model(
        contrastive_model,
        train_questions,
        train_answers,
        device=args.device
    )
    contrastive_results = evaluate_model(
        contrastive_model,
        test_questions,
        test_answers,
        device=args.device
    )
    
    # Save Contrastive Loss results
    with open("e5_contrastive_results.json", "w") as f:
        json.dump(contrastive_results, f, indent=4)
    
    # Train and evaluate Triplet Loss model
    print("\nTraining with Triplet Loss...")
    triplet_model = SentenceTransformer('intfloat/multilingual-e5-base')
    triplet_model = train_triplet_model(
        triplet_model,
        train_questions,
        train_answers,
        device=args.device
    )
    triplet_results = evaluate_model(
        triplet_model,
        test_questions,
        test_answers,
        device=args.device
    )
    
    # Save Triplet Loss results
    with open("e5_triplet_results.json", "w") as f:
        json.dump(triplet_results, f, indent=4)
    
    # Print results
    print("\nContrastive Loss Results:")
    for metric, value in contrastive_results.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nTriplet Loss Results:")
    for metric, value in triplet_results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 