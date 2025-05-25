from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PretrainedConfig
from torch import nn
from typing import Optional


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