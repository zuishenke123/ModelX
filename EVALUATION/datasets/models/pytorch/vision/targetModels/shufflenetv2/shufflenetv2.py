import paddle
from functools import partial
from typing import Any, Callable, List, Optional
import paddle.nn as nn
from paddle import Tensor
__all__ = ['ShuffleNetV2']


def channel_shuffle(x: paddle.Tensor, groups: int) ->paddle.Tensor:
    batchsize, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups
    x = x.reshape([batchsize, groups, channels_per_group, height, width])
    perm1 = list(range(len(x.shape)))
    perm1[1], perm1[2] = perm1[2], perm1[1]
    x = paddle.transpose(x, perm=perm1)
    x = x.reshape([batchsize, num_channels, height, width])
    return x


class InvertedResidual(paddle.nn.Layer):

    def __init__(self, inp: int, oup: int, stride: int) ->None:
        super().__init__()
        if not 1 <= stride <= 3:
            raise ValueError('illegal stride value')
        self.stride = stride
        branch_features = oup // 2
        if self.stride == 1 and inp != branch_features << 1:
            raise ValueError(
                f'Invalid combination of stride {stride}, inp {inp} and oup {oup} values. If stride == 1 then inp should be equal to oup // 2 << 1.'
                )
        if self.stride > 1:
            self.branch1 = paddle.nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                paddle.nn.BatchNorm2D(num_features=inp),
                paddle.nn.Conv2D( in_channels=inp, out_channels=branch_features, kernel_size= 1, stride=1, padding=0, bias_attr=False),
                paddle.nn.BatchNorm2D(num_features=branch_features),
                paddle.nn.ReLU()
            )
        else:
            self.branch1 = paddle.nn.Sequential()
        self.branch2 = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels= inp if self._stride > 1 else branch_features, out_channels= branch_features, kernel_size=1, stride=1, padding=0, bias_attr= False),
            paddle.nn.BatchNorm2D(num_features=branch_features),
            paddle.nn.ReLU(),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            paddle.nn.BatchNorm2D(num_features=branch_features),
            paddle.nn.Conv2D(in_channels=branch_features, out_channels= branch_features, kernel_size=1, stride=1, padding=0, bias_attr= False),
            paddle.nn.BatchNorm2D(num_features=branch_features),
            paddle.nn.ReLU()
        )

    @staticmethod
    def depthwise_conv(i: int, o: int, kernel_size: int, stride: int=1,
        padding: int=0, bias: bool=False) ->paddle.nn.Conv2D:
        return paddle.nn.Conv2D(in_channels=i, out_channels=o, kernel_size=
            kernel_size, stride=stride, padding=padding, bias_attr=bias,
            groups=i)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(chunks=2, axis=1)
            out = paddle.concat(x=(x1, self.branch2(x2)), axis=1)
        else:
            out = paddle.concat(x=(self.branch1(x), self.branch2(x)), axis=1)
        out = channel_shuffle(out, 2)
        return out


class ShuffleNetV2(paddle.nn.Layer):

    def __init__(self, stages_repeats: List[int], stages_out_channels: List
        [int], num_classes: int=1000, inverted_residual: Callable[...,
        paddle.nn.Layer]=InvertedResidual) ->None:
        super().__init__()
        if len(stages_repeats) != 3:
            raise ValueError(
                'expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError(
                'expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels= input_channels, out_channels=output_channels, kernel_size=3, stride=2, padding=1, bias_attr=False),
            paddle.nn.BatchNorm2D( num_features=output_channels),
            paddle.nn.ReLU()
        )
        input_channels = output_channels
        self.maxpool = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.stage2: paddle.nn.Sequential
        self.stage3: paddle.nn.Sequential
        self.stage4: paddle.nn.Sequential
        stage_names = [f'stage{i}' for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names,
            stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels,
                    output_channels, 1))
            setattr(
                self,
                name,
                paddle.nn.Sequential(*seq
            ))
            input_channels = output_channels
        output_channels = self._stage_out_channels[-1]
        self.conv5 = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels= input_channels, out_channels=output_channels, kernel_size=1, stride=1, padding=0, bias_attr=False),
            paddle.nn.BatchNorm2D( num_features=output_channels),
            paddle.nn.ReLU()
        )
        self.fc = paddle.nn.Linear(in_features=output_channels,
            out_features=num_classes)

    def _forward_impl(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean(axis=[2, 3])
        x = self.fc(x)
        return x

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        return self._forward_impl(x)
