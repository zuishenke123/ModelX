from .modified_KaimingNormal import Modified_KaimingNormal
import paddle
import re
from collections import OrderedDict
from functools import partial
from typing import Any, List, Optional, Tuple
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor
__all__ = ['DenseNet']


class _DenseLayer(paddle.nn.Layer):

    def __init__(self, num_input_features: int, growth_rate: int, bn_size:
        int, drop_rate: float, memory_efficient: bool=False) ->None:
        super().__init__()
        self.norm1 = paddle.nn.BatchNorm2D(num_features=num_input_features)
        self.relu1 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(in_channels=num_input_features,
            out_channels=bn_size * growth_rate, kernel_size=1, stride=1,
            bias_attr=False)
        self.norm2 = paddle.nn.BatchNorm2D(num_features=bn_size * growth_rate)
        self.relu2 = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(in_channels=bn_size * growth_rate,
            out_channels=growth_rate, kernel_size=3, stride=1, padding=1,
            bias_attr=False)
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[paddle.Tensor]) ->paddle.Tensor:
        concated_features = paddle.concat(x=inputs, axis=1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(
            concated_features)))
        return bottleneck_output

    def any_requires_grad(self, input: List[paddle.Tensor]) ->bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    def forward(self, input: paddle.Tensor) ->paddle.Tensor:
        if isinstance(input, paddle.Tensor):
            prev_features = [input]
        else:
            prev_features = input
        if self.memory_efficient and self.any_requires_grad(prev_features):
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = paddle.nn.functional.dropout(x=new_features, p=
                self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(paddle.nn.LayerDict):
    _version = 2

    def __init__(self, num_layers: int, num_input_features: int, bn_size:
        int, growth_rate: int, drop_rate: float, memory_efficient: bool=False
        ) ->None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                growth_rate=growth_rate, bn_size=bn_size, drop_rate=
                drop_rate, memory_efficient=memory_efficient)
            self.add_sublayer(name='denselayer%d' % (i + 1), sublayer=layer)

    def forward(self, init_features: paddle.Tensor) ->paddle.Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return paddle.concat(x=features, axis=1)


class _Transition(paddle.nn.Sequential):

    def __init__(self, num_input_features: int, num_output_features: int
        ) ->None:
        super().__init__()
        self.norm = paddle.nn.BatchNorm2D(num_features=num_input_features)
        self.relu = paddle.nn.ReLU()
        self.conv = paddle.nn.Conv2D(in_channels=num_input_features,
            out_channels=num_output_features, kernel_size=1, stride=1,
            bias_attr=False)
        self.pool = paddle.nn.AvgPool2D(kernel_size=2, stride=2)


class DenseNet(paddle.nn.Layer):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(self, growth_rate: int=32, block_config: Tuple[int, int,
        int, int]=(6, 12, 24, 16), num_init_features: int=64, bn_size: int=
        4, drop_rate: float=0, num_classes: int=1000, memory_efficient:
        bool=False) ->None:
        super().__init__()
        self.features = paddle.nn.Sequential(
            *[('conv0',
            paddle.nn.Conv2D( in_channels=3, out_channels=num_init_features, kernel_size=7, stride=2, padding=3, bias_attr=False)
        ), ('norm0', paddle.nn.
            BatchNorm2D(num_features=num_init_features)), ('relu0', paddle.
            nn.ReLU()), ('pool0', paddle.nn.MaxPool2D(kernel_size=3, stride
            =2, padding=1))])
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=
                num_features, bn_size=bn_size, growth_rate=growth_rate,
                drop_rate=drop_rate, memory_efficient=memory_efficient)
            self.features.add_sublayer(name='denseblock%d' % (i + 1),
                sublayer=block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                    num_output_features=num_features // 2)
                self.features.add_sublayer(name='transition%d' % (i + 1),
                    sublayer=trans)
                num_features = num_features // 2
        self.features.add_sublayer(name='norm5', sublayer=paddle.nn.
            BatchNorm2D(num_features=num_features))
        self.classifier = paddle.nn.Linear(in_features=num_features,
            out_features=num_classes)
        for m in self.sublayers():
            if isinstance(m, paddle.nn.Conv2D):
                tmp_initializer = Modified_KaimingNormal()
                m.weight.set_value(paddle.create_parameter(shape=m.weight.
                    shape, dtype=m.weight.dtype, default_initializer=
                    tmp_initializer))
            elif isinstance(m, paddle.nn.BatchNorm2D):
                tmp_initializer = paddle.nn.initializer.Constant(value=1)
                m.weight.set_value(paddle.create_parameter(shape=m.weight.
                    shape, dtype=m.weight.dtype, default_initializer=
                    tmp_initializer))
                tmp_initializer = paddle.nn.initializer.Constant(value=0)
                m.bias.set_value(paddle.create_parameter(shape=m.bias.shape,
                    dtype=m.bias.dtype, default_initializer=tmp_initializer))
            elif isinstance(m, paddle.nn.Linear):
                tmp_initializer = paddle.nn.initializer.Constant(value=0)
                m.bias.set_value(paddle.create_parameter(shape=m.bias.shape,
                    dtype=m.bias.dtype, default_initializer=tmp_initializer))

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        features = self.features(x)
        out = paddle.nn.functional.relu(x=features)
        out = paddle.nn.functional.adaptive_avg_pool2d(x=out, output_size=(
            1, 1))
        out = paddle.flatten(x=out, start_axis=1)
        out = self.classifier(out)
        return out
