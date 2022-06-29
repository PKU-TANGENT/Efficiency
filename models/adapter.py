import torch
import torch.nn as nn
class Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.down_project = nn.Linear(config.hidden_size, config.project_dim)
        self.activation = nn.Tanh()
        self.up_project = nn.Linear(config.project_dim, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        outputs = self.down_project(hidden_states)
        outputs = self.activation(outputs)
        outputs = self.up_project(outputs)
        outputs = self.dropout(outputs)
        outputs = self.LayerNorm(hidden_states + outputs)
        return outputs