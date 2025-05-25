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
    create_triplet_pairs,
    create_triplet_dataset,
    evaluate_model
)

def train_triplet_model(train_dataset, model_name='intfloat/multilingual-e5-base', output_dir='./triplet_model', batch_size=8, epochs=1, device='cuda'):
    print(f"Train model {model_name}...")
    model = TripletModel(model_name=model_name)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_dir='./logs',
        logging_steps=50,
        save_total_limit=1,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()
    return model

def main_triplet():
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--n_triples', type=int, default=50000)
    parser.add_argument('--max_samples', type=int, default=100000)
    args = parser.parse_args()

    data = load_and_split_data(max_samples=args.max_samples)
    train_q = [d['question'] for d in data['train']]
    train_a = [d['answer'] for d in data['train']]
    test_q = [d['question'] for d in data['test']]
    test_a = [d['answer'] for d in data['test']]

    anchors, positives, negatives = create_triplet_pairs(train_q, train_a, args.n_triples)
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
    dataset = create_triplet_dataset(anchors, positives, negatives, tokenizer)

    device = args.device
    if not torch.cuda.is_available():
        device = 'cpu'
    model = train_triplet_model(dataset, batch_size=args.batch_size, epochs=args.epochs, device=device)

    results = evaluate_model(model, test_q, test_a, device=device, batch_size=args.batch_size)
    print(results)
    with open("triplet_trainer_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main_triplet()
