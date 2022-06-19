import torch.nn as nn
import torch
class SWAM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.output_size = 1 # a score for last layer output of each word
        self.embedding_transform = nn.Linear(self.hidden_size, self.output_size)
        self.last_hidden_transform = nn.Linear(self.hidden_size, self.output_size)
        self.activation = nn.Tanh()
        self.get_prob = nn.Softmax(dim=-1)
    def forward(self, word_embedding, last_hidden_state, attention_mask, force_swam_weight=None):
        batch, seq_len, _ = last_hidden_state.shape
        weights = self.embedding_transform(word_embedding) + self.last_hidden_transform(last_hidden_state)
        weights = self.activation(weights)
        weights = weights.view(batch, seq_len) 
        weights = weights - (attention_mask==0).clone().detach()*10000
        weights = self.get_prob(weights)# get probs along seq_len dimension
        swam_outputs = last_hidden_state * weights.unsqueeze(2) if force_swam_weight is None else last_hidden_state * force_swam_weight.unsqueeze(2)
        swam_outputs = torch.sum(swam_outputs, dim=1)
        return (swam_outputs, weights)