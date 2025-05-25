import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
import numpy as np
from tqdm import tqdm

def load_and_split_data(test_size: float = 0.2, seed: int = 42, max_samples: int = 100000) -> Tuple[Dict, Dict]:
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
    # Normalize embeddings
    query_norm = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    doc_norm = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    
    # Compute cosine similarity
    similarity = np.dot(query_norm, doc_norm.T)
    
    return similarity

def get_top_k_predictions(similarity_matrix: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(similarity_matrix, axis=1)[:, ::-1][:, :k]

def create_contrastive_pairs(questions: List[str], 
                           answers: List[str], 
                           n_pairs: int) -> Tuple[List[str], List[str], List[int]]:
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
    print('Creating triplet dataset...')
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

@torch.no_grad()
def evaluate_model(model, test_questions, test_answers, device='cuda', batch_size=8):
    from metrics import batch_recall_at_k, batch_mrr
    print(f"Evaluationg model...")
    model.eval()
    model = model.to(device)
    
    q_embeddings = []
    for i in range(0, len(test_questions), batch_size):
        q_batch = test_questions[i:i+batch_size]
        q_emb = model.encode(q_batch, device=device)
        q_embeddings.append(q_emb.cpu())
    q_embeddings = torch.cat(q_embeddings).numpy()
    
    a_embeddings = []
    for i in range(0, len(test_answers), batch_size):
        a_batch = test_answers[i:i+batch_size]
        a_emb = model.encode(a_batch, device=device)
        a_embeddings.append(a_emb.cpu())
    a_embeddings = torch.cat(a_embeddings).numpy()

    print("Computing cosine similarity..")
    sim_matrix = compute_cosine_similarity(q_embeddings, a_embeddings)
    print("Getting top k predictions...")
    predictions = get_top_k_predictions(sim_matrix, k=10)
    ground_truth = np.arange(len(test_questions))

    return {
        "recall@1": float(batch_recall_at_k(ground_truth, predictions, k=1)),
        "recall@3": float(batch_recall_at_k(ground_truth, predictions, k=3)),
        "recall@10": float(batch_recall_at_k(ground_truth, predictions, k=10)),
        "mrr": float(batch_mrr(ground_truth, predictions)),
    }