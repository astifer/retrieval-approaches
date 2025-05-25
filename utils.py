import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
import numpy as np
from tqdm import tqdm

def load_and_split_data(test_size: float = 0.2, seed: int = 42, max_samples: int = 100000) -> Tuple[Dict, Dict]:
    """
    Load Natural Questions dataset and split into train/test sets.
    
    Args:
        test_size: Proportion of data to use for testing
        seed: Random seed for reproducibility
        max_samples: Maximum number of samples to use
        
    Returns:
        Tuple of (train_data, test_data) dictionaries
    """
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("sentence-transformers/natural-questions")
    
    # Convert to list of dictionaries for easier processing
    print("Processing data...")
    data = [{"question": item["query"], "answer": item["answer"]} 
            for item in dataset["train"]]
    
    # Take a subset if max_samples is specified
    if max_samples and max_samples < len(data):
        np.random.seed(seed)
        indices = np.random.choice(len(data), max_samples, replace=False)
        data = [data[i] for i in indices]
    
    # Split into train/test
    print("Splitting data...")
    train_data, test_data = train_test_split(
        data, 
        test_size=test_size, 
        random_state=seed
    )
    
    print(f"Using {len(train_data)} training samples and {len(test_data)} test samples")
    return {"train": train_data, "test": test_data}

def compute_cosine_similarity(query_embeddings: np.ndarray, 
                            doc_embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and document embeddings.
    
    Args:
        query_embeddings: Query embeddings of shape (n_queries, embedding_dim)
        doc_embeddings: Document embeddings of shape (n_docs, embedding_dim)
        
    Returns:
        Similarity matrix of shape (n_queries, n_docs)
    """
    # Normalize embeddings
    query_norm = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    doc_norm = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    
    # Compute cosine similarity
    similarity = np.dot(query_norm, doc_norm.T)
    
    return similarity

def get_top_k_predictions(similarity_matrix: np.ndarray, k: int) -> np.ndarray:
    """
    Get top-k predictions for each query based on similarity scores.
    
    Args:
        similarity_matrix: Similarity matrix of shape (n_queries, n_docs)
        k: Number of top predictions to return
        
    Returns:
        Array of shape (n_queries, k) containing indices of top-k documents
    """
    return np.argsort(similarity_matrix, axis=1)[:, ::-1][:, :k]

def create_contrastive_pairs(questions: List[str], 
                           answers: List[str], 
                           n_pairs: int) -> Tuple[List[str], List[str], List[int]]:
    """
    Create contrastive pairs for training.
    
    Args:
        questions: List of questions
        answers: List of answers
        n_pairs: Number of pairs to create
        
    Returns:
        Tuple of (questions, answers, labels) where labels are 1 for positive pairs
        and 0 for negative pairs
    """
    n_samples = len(questions)
    pairs_questions = []
    pairs_answers = []
    labels = []
    
    for _ in tqdm(range(n_pairs)):
        # Create positive pair
        idx = np.random.randint(0, n_samples)
        pairs_questions.append(questions[idx])
        pairs_answers.append(answers[idx])
        labels.append(1)
        
        # Create negative pair
        idx1 = np.random.randint(0, n_samples)
        idx2 = np.random.randint(0, n_samples)
        while idx2 == idx1:  # Ensure different indices
            idx2 = np.random.randint(0, n_samples)
        
        pairs_questions.append(questions[idx1])
        pairs_answers.append(answers[idx2])
        labels.append(0)
    
    return pairs_questions, pairs_answers, labels

def create_triplet_pairs(questions: List[str], 
                        answers: List[str], 
                        n_triplets: int) -> Tuple[List[str], List[str], List[str]]:
    """
    Create triplet pairs for training.
    
    Args:
        questions: List of questions
        answers: List of answers
        n_triplets: Number of triplets to create
        
    Returns:
        Tuple of (anchors, positives, negatives)
    """
    n_samples = len(questions)
    anchors = []
    positives = []
    negatives = []
    
    for _ in tqdm(range(n_triplets)):
        # Select anchor and positive
        idx = np.random.randint(0, n_samples)
        anchors.append(questions[idx])
        positives.append(answers[idx])
        
        # Select negative
        neg_idx = np.random.randint(0, n_samples)
        while neg_idx == idx:  # Ensure different indices
            neg_idx = np.random.randint(0, n_samples)
        negatives.append(answers[neg_idx])
    
    return anchors, positives, negatives 


from datasets import Dataset
import torch

def create_contrastive_dataset(questions, answers, labels, tokenizer):
    def tokenize(example):
        tokens_1 = tokenizer(example['text1'], truncation=True, padding="max_length", max_length=128)
        tokens_2 = tokenizer(example['text2'], truncation=True, padding="max_length", max_length=128)
        return {
            'input_ids': tokens_1['input_ids'],
            'attention_mask': tokens_1['attention_mask'],
            'input_ids_2': tokens_2['input_ids'],
            'attention_mask_2': tokens_2['attention_mask'],
            'labels': example['label']
        }

    data = [{'text1': q, 'text2': a, 'label': float(l)} for q, a, l in zip(questions, answers, labels)]
    dataset = Dataset.from_list(data)
    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'input_ids_2', 'attention_mask_2', 'labels'])
    return dataset

def create_triplet_dataset(anchors, positives, negatives, tokenizer):
    def tokenize(example):
        anc = tokenizer(example['anchor'], truncation=True, padding="max_length", max_length=128)
        pos = tokenizer(example['positive'], truncation=True, padding="max_length", max_length=128)
        neg = tokenizer(example['negative'], truncation=True, padding="max_length", max_length=128)
        return {
            'input_ids': anc['input_ids'],
            'attention_mask': anc['attention_mask'],
            'input_ids_pos': pos['input_ids'],
            'attention_mask_pos': pos['attention_mask'],
            'input_ids_neg': neg['input_ids'],
            'attention_mask_neg': neg['attention_mask'],
            'labels': 0.0  # Dummy label for Trainer compatibility
        }

    data = [{'anchor': a, 'positive': p, 'negative': n} for a, p, n in zip(anchors, positives, negatives)]
    dataset = Dataset.from_list(data)
    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type='torch', columns=[
        'input_ids', 'attention_mask',
        'input_ids_pos', 'attention_mask_pos',
        'input_ids_neg', 'attention_mask_neg',
        'labels'
    ])
    return dataset
