import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, Size
from typing import Union, List, Tuple
_shape_t = Union[int, List[int], Size]
class BitfitLayerNorm(nn.LayerNorm):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine', 'tune']
    tune: bool
    def __init__(self, normalized_shape: _shape_t, tune: bool=True, eps: float = 0.00001, elementwise_affine: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(nn.LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.tune = tune
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            if tune:
                self.bitfit = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()
    def reset_parameters(self) -> None:
        super().reset_parameters()
        nn.init.zeros_(self.bitfit)
    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input, 
            self.normalized_shape, 
            self.weight, 
            self.bias + self.bitfit, 
            self.eps
            ) if self.tune else F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)

        
class BitfitLinear(nn.Linear):
    __constants__ = ['in_features', 'out_features', 'tune']
    tune: bool
    def __init__(self, in_features: int, out_features: int, tune: bool=True, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(nn.Linear, self).__init__()
        self.tune = tune
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            if tune:
                self.bitfit = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self) -> None:
        super().reset_parameters()
        if self.tune:
            nn.init.zeros_(self.bitfit)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias + self.bitfit) if self.tune else F.linear(input, self.weight, self.bias)