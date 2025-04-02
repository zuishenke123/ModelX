from .modified_TruncatedNormal import Modified_TruncatedNormal
import paddle
import warnings
from collections import namedtuple
from functools import partial
from typing import Any, Callable, List, Optional, Tuple
import paddle.nn.functional as F
from paddle import nn, Tensor
__all__ = ['Inception3']
InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])
InceptionOutputs.__annotations__ = {'logits': paddle.Tensor, 'aux_logits':
    Optional[paddle.Tensor]}
_InceptionOutputs = InceptionOutputs


class Inception3(paddle.nn.Layer):

    def __init__(self, num_classes: int=1000, aux_logits: bool=True,
        transform_input: bool=False, inception_blocks: Optional[List[
        Callable[..., paddle.nn.Layer]]]=None, init_weights: Optional[bool]
        =None, dropout: float=0.5) ->None:
        super().__init__()
        if inception_blocks is None:
            inception_blocks = [BasicConv2d, InceptionA, InceptionB,
                InceptionC, InceptionD, InceptionE, InceptionAux]
        if init_weights is None:
            warnings.warn(
                'The default weight initialization of inception_v3 will be changed in future releases of torchvision. If you wish to keep the old behavior (which leads to long initialization times due to scipy/scipy#11299), please set init_weights=True.'
                , FutureWarning)
            init_weights = True
        if len(inception_blocks) != 7:
            raise ValueError(
                f'length of inception_blocks should be 7 instead of {len(inception_blocks)}'
                )
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = paddle.nn.MaxPool2D(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.maxpool2 = paddle.nn.MaxPool2D(kernel_size=3, stride=2)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        self.AuxLogits: Optional[paddle.nn.Layer] = None
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.avgpool = paddle.nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.fc = paddle.nn.Linear(in_features=2048, out_features=num_classes)
        if init_weights:
            for m in self.sublayers():
                if isinstance(m, paddle.nn.Conv2D) or isinstance(m, paddle.
                    nn.Linear):
                    stddev = float(m.stddev) if hasattr(m, 'stddev') else 0.1
                    tmp_initializer = Modified_TruncatedNormal(mean=0.0,
                        std=stddev)
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
        paddle.Tensor]]:
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        aux: Optional[paddle.Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = paddle.flatten(x=x, start_axis=1)
        x = self.fc(x)
        return x, aux

    def forward(self, x: paddle.Tensor) ->InceptionOutputs:
        x = self._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        return self.eager_outputs(x, aux)


class InceptionA(paddle.nn.Layer):

    def __init__(self, in_channels: int, pool_features: int, conv_block:
        Optional[Callable[..., paddle.nn.Layer]]=None) ->None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)
        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1
            )

    def _forward(self, x: paddle.Tensor) ->List[paddle.Tensor]:
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = paddle.nn.functional.avg_pool2d(x=x, kernel_size=3,
            stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        outputs = self._forward(x)
        return paddle.concat(x=outputs, axis=1)


class InceptionB(paddle.nn.Layer):

    def __init__(self, in_channels: int, conv_block: Optional[Callable[...,
        paddle.nn.Layer]]=None) ->None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

    def _forward(self, x: paddle.Tensor) ->List[paddle.Tensor]:
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = paddle.nn.functional.max_pool2d(x=x, kernel_size=3,
            stride=2)
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        outputs = self._forward(x)
        return paddle.concat(x=outputs, axis=1)


class InceptionC(paddle.nn.Layer):

    def __init__(self, in_channels: int, channels_7x7: int, conv_block:
        Optional[Callable[..., paddle.nn.Layer]]=None) ->None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)
        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(
            0, 3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=
            (3, 0))
        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1),
            padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7),
            padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1),
            padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7),
            padding=(0, 3))
        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x: paddle.Tensor) ->List[paddle.Tensor]:
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = paddle.nn.functional.avg_pool2d(x=x, kernel_size=3,
            stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        outputs = self._forward(x)
        return paddle.concat(x=outputs, axis=1)


class InceptionD(paddle.nn.Layer):

    def __init__(self, in_channels: int, conv_block: Optional[Callable[...,
        paddle.nn.Layer]]=None) ->None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)
        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7),
            padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1),
            padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

    def _forward(self, x: paddle.Tensor) ->List[paddle.Tensor]:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = paddle.nn.functional.max_pool2d(x=x, kernel_size=3,
            stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        outputs = self._forward(x)
        return paddle.concat(x=outputs, axis=1)


class InceptionE(paddle.nn.Layer):

    def __init__(self, in_channels: int, conv_block: Optional[Callable[...,
        paddle.nn.Layer]]=None) ->None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)
        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3),
            padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1),
            padding=(1, 0))
        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3),
            padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1),
            padding=(1, 0))
        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x: paddle.Tensor) ->List[paddle.Tensor]:
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)
            ]
        branch3x3 = paddle.concat(x=branch3x3, axis=1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.
            branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = paddle.concat(x=branch3x3dbl, axis=1)
        branch_pool = paddle.nn.functional.avg_pool2d(x=x, kernel_size=3,
            stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        outputs = self._forward(x)
        return paddle.concat(x=outputs, axis=1)


class InceptionAux(paddle.nn.Layer):

    def __init__(self, in_channels: int, num_classes: int, conv_block:
        Optional[Callable[..., paddle.nn.Layer]]=None) ->None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = paddle.nn.Linear(in_features=768, out_features=num_classes)
        self.fc.stddev = 0.001

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = paddle.nn.functional.avg_pool2d(x=x, kernel_size=5, stride=3)
        x = self.conv0(x)
        x = self.conv1(x)
        x = paddle.nn.functional.adaptive_avg_pool2d(x=x, output_size=(1, 1))
        x = paddle.flatten(x=x, start_axis=1)
        x = self.fc(x)
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
