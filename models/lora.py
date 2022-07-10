import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class Lora(nn.Linear):
    __constants__ = ['in_features', 'out_features', 'lora_rank']
    lora_down: torch.Tensor
    lora_up: torch.Tensor
    lora_rank: int
    def __init__(self, in_features: int, out_features: int, lora_rank: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        self.lora_rank = lora_rank
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(nn.Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.lora_down = nn.Parameter(torch.empty((lora_rank, in_features), **factory_kwargs))
        self.lora_up = nn.Parameter(torch.empty((out_features, lora_rank), **factory_kwargs))        
        self.reset_parameters()
    def reset_parameters(self) -> None:
        super().reset_parameters()
        nn.init.normal_(self.lora_down)
        nn.init.zeros_(self.lora_up)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tmp_weight = self.lora_up@self.lora_down + self.weight
        return F.linear(input, tmp_weight, self.bias)
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, lora_rank={}, bias={}'.format(
            self.in_features, self.out_features, self.lora_rank, self.bias is not None
        )

def modify_lora_layer(layer, config):
    layer.attention.self.query = Lora(config.hidden_size, layer.attention.self.all_head_size, config.lora_rank)
    layer.attention.self.key = Lora(config.hidden_size, layer.attention.self.all_head_size, config.lora_rank)
    # layer.attention.self.value = Lora(config.hidden_size, layer.attention.self.all_head_size, config.lora_rank)
    # layer.attention.output.dense = Lora(config.hidden_size, config.hidden_size, config.lora_rank)
    return layer