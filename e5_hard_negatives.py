import torch
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
from utils import (
    load_and_split_data,
    compute_cosine_similarity,
    get_top_k_predictions
)
from metrics import batch_recall_at_k, batch_mrr
import json
from tqdm import tqdm
import numpy as np
import argparse

def create_hard_negative_pairs(
    model: SentenceTransformer,
    questions: list,
    answers: list,
    n_triplets: int = 10000,
    k: int = 5,
    batch_size: int = 32,
    device: str = 'cuda'
) -> tuple:
    """
    Create triplet pairs with hard negatives.
    
    Args:
        model: E5 model
        questions: List of questions
        answers: List of answers
        n_triplets: Number of triplets to create
        k: Number of top similar documents to consider for hard negatives
        batch_size: Batch size for encoding
        device: Device to use for encoding ('cuda' or 'cpu')
        
    Returns:
        Tuple of (anchors, positives, negatives)
    """
    # Encode all questions and answers
    print(f"Encoding questions and answers on {device}...")
    q_embeddings = model.encode(questions, batch_size=batch_size, show_progress_bar=True, device=device)
    a_embeddings = model.encode(answers, batch_size=batch_size, show_progress_bar=True, device=device)
    
    # Compute similarity matrix
    print("Computing similarity matrix...")
    similarity_matrix = compute_cosine_similarity(q_embeddings, a_embeddings)
    
    # Get top-k similar documents for each question
    print("Getting top-k similar documents...")
    top_k_indices = get_top_k_predictions(similarity_matrix, k=k+1)  # +1 because the correct answer is included
    
    anchors = []
    positives = []
    negatives = []
    
    print("Creating hard negative triplets...")
    for i in tqdm(range(n_triplets)):
        # Select random question
        q_idx = np.random.randint(0, len(questions))
        
        # Get top-k similar answers (excluding the correct one)
        similar_answers = top_k_indices[q_idx]
        correct_answer_idx = q_idx
        similar_answers = similar_answers[similar_answers != correct_answer_idx][:k]
        
        # Select random hard negative
        neg_idx = np.random.choice(similar_answers)
        
        anchors.append(questions[q_idx])
        positives.append(answers[q_idx])
        negatives.append(answers[neg_idx])
    
    return anchors, positives, negatives

def train_model_with_hard_negatives(
    model: SentenceTransformer,
    train_questions: list,
    train_answers: list,
    n_triplets: int = 10000,
    batch_size: int = 32,
    epochs: int = 3,
    device: str = 'cuda'
) -> SentenceTransformer:
    """
    Train E5 model using Triplet Loss with hard negatives.
    
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
    # Create triplet pairs with hard negatives
    print("Creating triplet pairs with hard negatives...")
    anchors, positives, negatives = create_hard_negative_pairs(
        model,
        train_questions,
        train_answers,
        n_triplets,
        batch_size=batch_size,
        device=device
    )
    
    # Create training examples
    train_examples = [
        InputExample(texts=[a, p, n])
        for a, p, n in zip(anchors, positives, negatives)
    ]
    
    # Create data loader with pin_memory for faster GPU transfer
    train_dataloader = DataLoader(
        train_examples, 
        shuffle=True, 
        batch_size=batch_size,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Define loss function
    train_loss = losses.TripletLoss(model)
    
    # Train model
    print(f"Training model with Triplet Loss and hard negatives on {device}...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        show_progress_bar=True,
        report_to=None
    )
    
    return model

@torch.no_grad()
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
    parser = argparse.ArgumentParser(description='Train E5 model with hard negatives')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for training (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=2,
                      help='Number of training epochs')
    parser.add_argument('--n_triplets', type=int, default=50000,
                      help='Number of training triplets to create')
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
    
    # Train and evaluate model with hard negatives
    print("\nTraining with hard negatives...")
    model = SentenceTransformer('intfloat/multilingual-e5-base')
    model.to(args.device)
    model = train_model_with_hard_negatives(
        model,
        train_questions,
        train_answers,
        n_triplets=args.n_triplets,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device
    )
    model.eval()
    results = evaluate_model(
        model,
        test_questions,
        test_answers,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Save results
    with open("e5_hard_negatives_results.json", "w") as f:
        json.dump(results, f, indent=4)
    del model
    
    # Print results
    print("\nResults:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 