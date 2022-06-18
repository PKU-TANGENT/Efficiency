import torch.nn as nn
import torch
class SWAM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.output_size = 2 # a score for first and last layer of each word
        self.embedding_transform = nn.Linear(self.hidden_size, self.output_size)
        self.last_hidden_transform = nn.Linear(self.hidden_size, self.output_size)
        self.activation = nn.Tanh()
        self.get_prob = nn.Softmax(dim=-1)
    def forward(self, word_embeddings, last_hidden_state, attention_masks):
        batch, seq_len, _ = last_hidden_state.shape
        weights = self.embedding_transform(word_embeddings) + self.last_hidden_transform(last_hidden_state)
        weights = self.activation(weights)
        # weights (batch, seq_len, output_size)
        # attention_masks (batch, seq_len)
        weights = weights - ((attention_masks==0).clone().detach()*10000).unsqueeze(-1)
        weights = weights.view(batch, seq_len * self.output_size)
        weights = self.get_prob(weights)# get probs along seq_len dimension
        weights = weights.view(batch, seq_len, self.output_size)
        swam_outputs = last_hidden_state * weights[:,:,0].unsqueeze(2) + word_embeddings * weights[:,:,1].unsqueeze(2)
        swam_outputs = torch.sum(swam_outputs, dim=1)
        return (swam_outputs, weights)

class SWAMClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
