import torch
import torch.nn as nn
import math
class Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.down_project = nn.Linear(config.hidden_size, config.project_dim) if not config.identity_init else AdapterDownProject(config.hidden_size, config.project_dim)
        self.activation = nn.Tanh()
        self.up_project = nn.Linear(config.project_dim, config.hidden_size) if not config.identity_init else AdapterUpProject(config.project_dim, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=config.elementwise_affine)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        outputs = self.down_project(hidden_states)
        outputs = self.activation(outputs)
        outputs = self.up_project(outputs)
        outputs = self.dropout(outputs)
        outputs = self.LayerNorm(hidden_states + outputs)
        return outputs
        
class AdapterDownProject(nn.Linear):
    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
class AdapterUpProject(nn.Linear):
    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)