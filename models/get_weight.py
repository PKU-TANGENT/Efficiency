import torch.nn as nn
import torch
class SelfWeighted(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.output_size = 1 # a probability for each word
        self.layer = nn.Linear(self.hidden_size*2, self.output_size)
        self.activation = nn.Tanh()
        self.get_prob = nn.Softmax(dim=-1)
    def forward(self, word_embeddings, last_hidden_state, attention_masks):
        batch, seq_len, _ = last_hidden_state.shape
        concated_inputs = torch.cat([word_embeddings, last_hidden_state], dim=-1)
        weights = self.layer(concated_inputs)
        weights = self.activation(weights)
        weights = weights.view(batch, seq_len) 
        weights = weights - (attention_masks==0).clone().detach()*10000
        weights = self.get_prob(weights)# get probs along seq_len dimension
        self_weighted_outputs = last_hidden_state * weights.unsqueeze(2)
        self_weighted_outputs = torch.sum(self_weighted_outputs, dim=1)
        return (self_weighted_outputs, weights)
