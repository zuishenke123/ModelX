# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle import ParamAttr
from paddle.nn.initializer import Uniform

__all__ = []


class ConvPoolLayer(nn.Layer):
    def __init__(
        self,
        input_channels,
        output_channels,
        filter_size,
        stride,
        padding,
        stdv,
        groups=1,
        act=None,
    ):
        super().__init__()

        self.relu = nn.ReLU() if act == "relu" else None

        self._conv = nn.Conv2D(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=None,
            bias_attr=None,
        )
        self._pool = nn.MaxPool2D(kernel_size=3, stride=2, padding=0)

    def forward(self, inputs):
        x = self._conv(inputs)
        if self.relu is not None:
            x = self.relu(x)
        x = self._pool(x)
        return x


class AlexNet(nn.Layer):
    """AlexNet model from
    `"ImageNet Classification with Deep Convolutional Neural Networks"
    <https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf>`_.

    Args:
        num_classes (int, optional): Output dim of last fc layer. If num_classes <= 0, last fc layer
            will not be defined. Default: 1000.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of AlexNet model.

    """

    def __init__(self, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes
        stdv = 1.0 / math.sqrt(3 * 11 * 11)
        self._conv1 = ConvPoolLayer(3, 64, 11, 4, 2, stdv, act="relu")
        stdv = 1.0 / math.sqrt(64 * 5 * 5)
        self._conv2 = ConvPoolLayer(64, 192, 5, 1, 2, stdv, act="relu")
        stdv = 1.0 / math.sqrt(192 * 3 * 3)
        self._conv3 = nn.Conv2D(
            192,
            384,
            3,
            stride=1,
            padding=1,
            weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(initializer=Uniform(-stdv, stdv)),
        )
        self._conv4 = nn.Conv2D(
            384,
            256,
            3,
            stride=1,
            padding=1,
            weight_attr=ParamAttr(initializer=Uniform(- 1.0 / math.sqrt(384 * 3 * 3), 1.0 / math.sqrt(384 * 3 * 3))),
            bias_attr=ParamAttr(initializer=Uniform(- 1.0 / math.sqrt(384 * 3 * 3), 1.0 / math.sqrt(384 * 3 * 3))),
        )
        stdv = 1.0 / math.sqrt(256 * 3 * 3)
        self._conv5 = ConvPoolLayer(256, 256, 3, 1, 1, stdv, act="relu")

        if self.num_classes > 0:
            self._drop1 = nn.Dropout(p=0.5, mode="downscale_in_infer")
            self._fc6 = nn.Linear(
                in_features=256 * 6 * 6,
                out_features=4096,
                weight_attr=ParamAttr(initializer=Uniform(- 1.0 / math.sqrt(256 * 6 * 6), 1.0 / math.sqrt(256 * 6 * 6))),
                bias_attr=ParamAttr(initializer=Uniform(- 1.0 / math.sqrt(256 * 6 * 6), 1.0 / math.sqrt(256 * 6 * 6))),
            )

            self._drop2 = nn.Dropout(p=0.5, mode="downscale_in_infer")
            self._fc7 = nn.Linear(
                in_features=4096,
                out_features=4096,
                weight_attr=ParamAttr(initializer=Uniform(- 1.0 / math.sqrt(256 * 6 * 6), 1.0 / math.sqrt(256 * 6 * 6))),
                bias_attr=ParamAttr(initializer=Uniform(- 1.0 / math.sqrt(256 * 6 * 6), 1.0 / math.sqrt(256 * 6 * 6))),
            )
            self._fc8 = nn.Linear(
                in_features=4096,
                out_features=num_classes,
                weight_attr=ParamAttr(initializer=Uniform(- 1.0 / math.sqrt(256 * 6 * 6), 1.0 / math.sqrt(256 * 6 * 6))),
                bias_attr=ParamAttr(initializer=Uniform(- 1.0 / math.sqrt(256 * 6 * 6), 1.0 / math.sqrt(256 * 6 * 6))),
            )

    def forward(self, inputs):
        x = self._conv1(inputs)
        x = self._conv2(x)
        x = self._conv3(x)
        x = F.relu(x)
        x = self._conv4(x)
        x = F.relu(x)
        x = self._conv5(x)

        if self.num_classes > 0:
            x = paddle.flatten(x, start_axis=1, stop_axis=-1)
            x = self._drop1(x)
            x = self._fc6(x)
            x = F.relu(x)
            x = self._drop2(x)
            x = self._fc7(x)
            x = F.relu(x)
            x = self._fc8(x)

        return x

