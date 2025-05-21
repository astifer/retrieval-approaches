import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_and_split_data, compute_cosine_similarity, get_top_k_predictions
from metrics import batch_recall_at_k, batch_mrr
import json
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run TF-IDF baseline')
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
    
    # Initialize TF-IDF vectorizer
    print("Training TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    # Fit vectorizer on training data
    vectorizer.fit(train_questions + train_answers)
    
    # Transform questions and answers
    print("Vectorizing questions and answers...")
    train_q_vectors = vectorizer.transform(train_questions)
    train_a_vectors = vectorizer.transform(train_answers)
    test_q_vectors = vectorizer.transform(test_questions)
    test_a_vectors = vectorizer.transform(test_answers)
    
    # Compute similarity matrix
    print("Computing similarity matrix...")
    similarity_matrix = cosine_similarity(test_q_vectors, test_a_vectors)
    
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
    
    with open("tfidf_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main() 