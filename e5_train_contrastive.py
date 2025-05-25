from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, PreTrainedModel, PretrainedConfig
from torch import nn
import torch
from typing import Optional, Union
import numpy as np
import argparse
import json

from metrics import batch_recall_at_k, batch_mrr
from models import TripletModel, ContrastiveModel

from utils import (
    load_and_split_data,
    create_contrastive_pairs,
    create_triplet_pairs,
    compute_cosine_similarity,
    get_top_k_predictions,
    create_contrastive_dataset
)


@torch.no_grad()
def evaluate_model(model, test_questions, test_answers, device='cuda', batch_size=8):
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


def train_with_trainer(train_dataset, model_name='intfloat/multilingual-e5-small', output_dir='./contrastive_model', batch_size=8, epochs=3, device='cuda'):
    model = ContrastiveModel(model_name=model_name)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_dir='./logs',
        logging_steps=50,
        save_total_limit=1,
        remove_unused_columns=False,
        device=device
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    print(f"Train model...")
    trainer.train()
    return model

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--n_pairs', type=int, default=50000)
    parser.add_argument('--max_samples', type=int, default=100000)
    args = parser.parse_args()

    data = load_and_split_data(max_samples=args.max_samples)
    train_q = [d['question'] for d in data['train']]
    train_a = [d['answer'] for d in data['train']]
    test_q = [d['question'] for d in data['test']]
    test_a = [d['answer'] for d in data['test']]

    q_pairs, a_pairs, labels = create_contrastive_pairs(train_q, train_a, args.n_pairs)
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
    dataset = create_contrastive_dataset(q_pairs, a_pairs, labels, tokenizer)
    
    device = device
    if not torch.cuda.is_available():
        device = 'cpu'

    model = train_with_trainer(dataset, batch_size=args.batch_size, epochs=args.epochs, device=device)
    results = evaluate_model(model, test_q, test_a, device=device, batch_size=args.batch_size)
    print(results)
    with open("contrastive_trainer_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
