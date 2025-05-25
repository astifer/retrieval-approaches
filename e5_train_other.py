from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, PreTrainedModel, PretrainedConfig
from torch import nn
import torch
from typing import Optional, Union
import numpy as np

from metrics import batch_recall_at_k, batch_mrr

from utils import (
    load_and_split_data,
    create_contrastive_pairs,
    create_triplet_pairs,
    compute_cosine_similarity,
    get_top_k_predictions,
    create_contrastive_dataset
)

class SentenceEmbeddingConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class ContrastiveModel(PreTrainedModel):
    def __init__(self, model_name: str, config: Optional[PretrainedConfig] = None):
        if config is None:
            config = SentenceEmbeddingConfig()
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.similarity = nn.CosineSimilarity(dim=1)

    def encode(self, texts, device='cuda'):
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = self.encoder(**tokens)
        embeddings = outputs.last_hidden_state[:, 0]  # CLS token
        return embeddings

    def forward(self, input_ids=None, attention_mask=None, labels=None, input_ids_2=None, attention_mask_2=None):
        outputs_1 = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        outputs_2 = self.encoder(input_ids=input_ids_2, attention_mask=attention_mask_2)
        emb_1 = outputs_1.last_hidden_state[:, 0]
        emb_2 = outputs_2.last_hidden_state[:, 0]

        similarities = self.similarity(emb_1, emb_2)
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(similarities, labels)
            return {'loss': loss, 'similarity': similarities}
        return {'similarity': similarities}

@torch.no_grad()
def evaluate_model(model, test_questions, test_answers, device='cuda', batch_size=32):
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

    sim_matrix = compute_cosine_similarity(q_embeddings, a_embeddings)
    predictions = get_top_k_predictions(sim_matrix, k=10)
    ground_truth = np.arange(len(test_questions))

    return {
        "recall@1": float(batch_recall_at_k(ground_truth, predictions, k=1)),
        "recall@3": float(batch_recall_at_k(ground_truth, predictions, k=3)),
        "recall@10": float(batch_recall_at_k(ground_truth, predictions, k=10)),
        "mrr": float(batch_mrr(ground_truth, predictions)),
    }


def train_with_trainer(train_dataset, model_name='intfloat/multilingual-e5-base', output_dir='./contrastive_model', batch_size=32, epochs=3):
    model = ContrastiveModel(model_name=model_name)
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

def main():
    from utils import load_and_split_data, create_contrastive_pairs
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=64)
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
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
    dataset = create_contrastive_dataset(q_pairs, a_pairs, labels, tokenizer)

    model = train_with_trainer(dataset, batch_size=args.batch_size, epochs=args.epochs)

    results = evaluate_model(model, test_q, test_a, device=args.device, batch_size=args.batch_size)
    with open("contrastive_trainer_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
