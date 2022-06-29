from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
import torch
class PoolerClassificationHead(RobertaClassificationHead):
    """Head for sentence-level classification tasks."""
    def __init__(self, config, pooler_type="cls"):
        super().__init__(config)
        self.pooler_type = pooler_type

    def forward(self, features, attention_mask=None):
        if self.pooler_type == "cls":
            features = features[:, 0]
        elif self.pooler_type == "avg" and attention_mask is not None:
            features = (features * attention_mask.unsqueeze(-1)).sum(axis=-2) / attention_mask.sum(axis=-1).unsqueeze(-1)
        else:
            raise NotImplementedError
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x