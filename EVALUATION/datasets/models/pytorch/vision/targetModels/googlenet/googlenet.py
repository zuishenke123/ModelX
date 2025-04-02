from .modified_TruncatedNormal import Modified_TruncatedNormal
import paddle
import warnings
from collections import namedtuple
from functools import partial
from typing import Any, Callable, List, Optional, Tuple
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor
__all__ = ['GoogLeNet']
GoogLeNetOutputs = namedtuple('GoogLeNetOutputs', ['logits', 'aux_logits2',
    'aux_logits1'])
GoogLeNetOutputs.__annotations__ = {'logits': paddle.Tensor, 'aux_logits2':
    Optional[paddle.Tensor], 'aux_logits1': Optional[paddle.Tensor]}
_GoogLeNetOutputs = GoogLeNetOutputs


class GoogLeNet(paddle.nn.Layer):
    __constants__ = ['aux_logits', 'transform_input']

    def __init__(self, num_classes: int=1000, aux_logits: bool=True,
        transform_input: bool=False, init_weights: Optional[bool]=None,
        blocks: Optional[List[Callable[..., paddle.nn.Layer]]]=None,
        dropout: float=0.2, dropout_aux: float=0.7) ->None:
        super().__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        if init_weights is None:
            warnings.warn(
                'The default weight initialization of GoogleNet will be changed in future releases of torchvision. If you wish to keep the old behavior (which leads to long initialization times due to scipy/scipy#11299), please set init_weights=True.'
                , FutureWarning)
            init_weights = True
        if len(blocks) != 3:
            raise ValueError(
                f'blocks length should be 3 instead of {len(blocks)}')
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = paddle.nn.MaxPool2D(kernel_size=3, stride=2,
            ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = paddle.nn.MaxPool2D(kernel_size=3, stride=2,
            ceil_mode=True)
        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = paddle.nn.MaxPool2D(kernel_size=3, stride=2,
            ceil_mode=True)
        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = paddle.nn.MaxPool2D(kernel_size=2, stride=2,
            ceil_mode=True)
        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)
        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes, dropout=
                dropout_aux)
            self.aux2 = inception_aux_block(528, num_classes, dropout=
                dropout_aux)
        else:
            self.aux1 = None
            self.aux2 = None
        self.avgpool = paddle.nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.fc = paddle.nn.Linear(in_features=1024, out_features=num_classes)
        if init_weights:
            for m in self.sublayers():
                if isinstance(m, paddle.nn.Conv2D) or isinstance(m, paddle.
                    nn.Linear):
                    tmp_initializer = Modified_TruncatedNormal(mean=0.0,
                        std=0.01)
                    m.weight.set_value(paddle.create_parameter(shape=m.
                        weight.shape, dtype=m.weight.dtype,
                        default_initializer=tmp_initializer))
                elif isinstance(m, paddle.nn.BatchNorm2D):
                    tmp_initializer = paddle.nn.initializer.Constant(value=1)
                    m.weight.set_value(paddle.create_parameter(shape=m.
                        weight.shape, dtype=m.weight.dtype,
                        default_initializer=tmp_initializer))
                    tmp_initializer = paddle.nn.initializer.Constant(value=0)
                    m.bias.set_value(paddle.create_parameter(shape=m.bias.
                        shape, dtype=m.bias.dtype, default_initializer=
                        tmp_initializer))

    def _transform_input(self, x: paddle.Tensor) ->paddle.Tensor:
        if self.transform_input:
            x_ch0 = paddle.unsqueeze(x=x[:, 0], axis=1) * (0.229 / 0.5) + (
                0.485 - 0.5) / 0.5
            x_ch1 = paddle.unsqueeze(x=x[:, 1], axis=1) * (0.224 / 0.5) + (
                0.456 - 0.5) / 0.5
            x_ch2 = paddle.unsqueeze(x=x[:, 2], axis=1) * (0.225 / 0.5) + (
                0.406 - 0.5) / 0.5
            x = paddle.concat(x=(x_ch0, x_ch1, x_ch2), axis=1)
        return x

    def _forward(self, x: paddle.Tensor) ->Tuple[paddle.Tensor, Optional[
        paddle.Tensor], Optional[paddle.Tensor]]:
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        aux1: Optional[paddle.Tensor] = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2: Optional[paddle.Tensor] = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = paddle.flatten(x=x, start_axis=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x, aux2, aux1

    def forward(self, x: paddle.Tensor) ->GoogLeNetOutputs:
        x = self._transform_input(x)
        x, aux1, aux2 = self._forward(x)
        aux_defined = self.training and self.aux_logits
        return self.eager_outputs(x, aux2, aux1)


class Inception(paddle.nn.Layer):

    def __init__(self, in_channels: int, ch1x1: int, ch3x3red: int, ch3x3:
        int, ch5x5red: int, ch5x5: int, pool_proj: int, conv_block:
        Optional[Callable[..., paddle.nn.Layer]]=None) ->None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)
        self.branch2 = paddle.nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        self.branch3 = paddle.nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )
        self.branch4 = paddle.nn.Sequential(
            paddle.nn.MaxPool2D(kernel_size =3, stride=1, padding=1, ceil_mode=True),
            conv_block( in_channels, pool_proj, kernel_size=1)
        )

    def _forward(self, x: paddle.Tensor) ->List[paddle.Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        outputs = self._forward(x)
        return paddle.concat(x=outputs, axis=1)


class InceptionAux(paddle.nn.Layer):

    def __init__(self, in_channels: int, num_classes: int, conv_block:
        Optional[Callable[..., paddle.nn.Layer]]=None, dropout: float=0.7
        ) ->None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)
        self.fc1 = paddle.nn.Linear(in_features=2048, out_features=1024)
        self.fc2 = paddle.nn.Linear(in_features=1024, out_features=num_classes)
        self.dropout = paddle.nn.Dropout(p=dropout)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = paddle.nn.functional.adaptive_avg_pool2d(x=x, output_size=(4, 4))
        x = self.conv(x)
        x = paddle.flatten(x=x, start_axis=1)
        x = paddle.nn.functional.relu(x=self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class BasicConv2d(paddle.nn.Layer):

    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any
        ) ->None:
        super().__init__()
        self.conv = paddle.nn.Conv2D(in_channels=in_channels, out_channels=
            out_channels, bias_attr=False, **kwargs)
        self.bn = paddle.nn.BatchNorm2D(num_features=out_channels, epsilon=
            0.001)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return paddle.nn.functional.relu(x=x)
