import torch 
import math
import warnings
import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch._torch_docs import reproducibility_notes
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union


class Modified_Conv2d(torch.nn.Conv2d):
	def __init__(self, in_channels: int, out_channels: int, kernel_size:
	    _size_2_t, stride: _size_2_t=1, padding: Union[str, _size_2_t]=0,
	    dilation: _size_2_t=1, groups: int=1, bias: bool=True, padding_mode:
	    str='zeros', device=None, dtype=None) ->None:
	    factory_kwargs = {'device': device, 'dtype': dtype}
	    kernel_size_ = _pair(kernel_size)
	    stride_ = _pair(stride)
	    padding_ = padding if isinstance(padding, str) else _pair(padding)
	    dilation_ = _pair(dilation)
	    super().__init__(in_channels, out_channels, kernel_size_, stride_,
	        padding_, dilation_, False, _pair(0), groups, bias, padding_mode,
	        **factory_kwargs)
	    (lambda tensor: torch.nn.init.uniform_(tensor, a=-1.0 / math.sqrt(384 *
	        3 * 3), b=1.0 / math.sqrt(384 * 3 * 3)))(self.weight)
	    (lambda tensor: torch.nn.init.uniform_(tensor, a=-1.0 / math.sqrt(384 *
	        3 * 3), b=1.0 / math.sqrt(384 * 3 * 3)))(self.bias)

