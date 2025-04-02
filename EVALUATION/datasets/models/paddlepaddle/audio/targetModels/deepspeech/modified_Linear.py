import torch 
import math
from typing import Any
import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules.module import Module
from torch.nn.modules.lazy import LazyModuleMixin


class Modified_Linear(torch.nn.Linear):
	def __init__(self, in_features: int, out_features: int, bias: bool=True,
	    device=None, dtype=None) ->None:
	    factory_kwargs = {'device': device, 'dtype': dtype}
	    super().__init__()
	    self.in_features = in_features
	    self.out_features = out_features
	    self.weight = Parameter(torch.empty((out_features, in_features), **
	        factory_kwargs))
	    if bias:
	        self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
	    else:
	        self.register_parameter('bias', None)
	    self.reset_parameters()
	    (True)(self.bias)

