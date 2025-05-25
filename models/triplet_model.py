from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, PreTrainedModel, PretrainedConfig
from torch import nn
import torch
from typing import Optional, Union
import numpy as np


class SentenceEmbeddingConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TripletModel(PreTrainedModel):
    def __init__(self, model_name: str, config: Optional[PretrainedConfig] = None):
        if config is None:
            config = SentenceEmbeddingConfig()
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

    def encode(self, texts, device='cuda'):
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = self.encoder(**tokens)
        return outputs.last_hidden_state[:, 0]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_ids_pos=None,
        attention_mask_pos=None,
        input_ids_neg=None,
        attention_mask_neg=None,
        labels=None  # unused, needed to comply with Trainer
    ):
        anchor = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        positive = self.encoder(input_ids=input_ids_pos, attention_mask=attention_mask_pos).last_hidden_state[:, 0]
        negative = self.encoder(input_ids=input_ids_neg, attention_mask=attention_mask_neg).last_hidden_state[:, 0]
        loss = self.loss_fn(anchor, positive, negative)
        return {'loss': loss}
