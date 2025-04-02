import paddle
from typing import cast, Dict, List, Union
import paddle.nn as nn
__all__ = ['VGG']


class VGG(paddle.nn.Layer):

    def __init__(self, features: paddle.nn.Layer, num_classes: int=1000,
        init_weights: bool=True, dropout: float=0.5) ->None:
        super().__init__()
        self.features = features
        self.avgpool = paddle.nn.AdaptiveAvgPool2D(output_size=(7, 7))
        self.classifier = paddle.nn.Sequential(
            paddle.nn.Linear(in_features =512 * 7 * 7, out_features=4096),
            paddle.nn.ReLU(name=True),
            paddle.nn.Dropout(p=dropout),
            paddle.nn.Linear(in_features=4096, out_features=4096),
            paddle.nn.ReLU(name=True),
            paddle.nn.Dropout(p=dropout),
            paddle.nn.Linear(in_features=4096, out_features=num_classes)
        )
        if init_weights:
            for m in self.sublayers():
                if isinstance(m, paddle.nn.Conv2D):
                    tmp_initializer = paddle.nn.initializer.KaimingNormal(
                        nonlinearity='relu')
                    m.weight.set_value(paddle.create_parameter(shape=m.
                        weight.shape, dtype=m.weight.dtype,
                        default_initializer=tmp_initializer))
                    if m.bias is not None:
                        tmp_initializer = paddle.nn.initializer.Constant(value
                            =0)
                        m.bias.set_value(paddle.create_parameter(shape=m.
                            bias.shape, dtype=m.bias.dtype,
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
                elif isinstance(m, paddle.nn.Linear):
                    tmp_initializer = paddle.nn.initializer.Normal(mean=0,
                        std=0.01)
                    m.weight.set_value(paddle.create_parameter(shape=m.
                        weight.shape, dtype=m.weight.dtype,
                        default_initializer=tmp_initializer))
                    tmp_initializer = paddle.nn.initializer.Constant(value=0)
                    m.bias.set_value(paddle.create_parameter(shape=m.bias.
                        shape, dtype=m.bias.dtype, default_initializer=
                        tmp_initializer))

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = paddle.flatten(x=x, start_axis=1)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool=False
    ) ->paddle.nn.Sequential:
    layers: List[paddle.nn.Layer] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [paddle.nn.MaxPool2D(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = paddle.nn.Conv2D(in_channels=in_channels, out_channels
                =v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, paddle.nn.BatchNorm2D(num_features=v),
                    paddle.nn.ReLU()]
            else:
                layers += [conv2d, paddle.nn.ReLU()]
            in_channels = v
    return paddle.nn.Sequential(
        *layers
    )


cfgs: Dict[str, List[Union[str, int]]] = {'A': [64, 'M', 128, 'M', 256, 256,
    'M', 512, 512, 'M', 512, 512, 'M'], 'B': [64, 64, 'M', 128, 128, 'M', 
    256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 'D': [64, 64, 'M', 128, 
    128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 
    512, 512, 'M', 512, 512, 512, 512, 'M']}
