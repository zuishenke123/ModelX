from .modified_Conv2d import Modified_Conv2d
from .modified_Linear import Modified_Linear
import torch
import math
__all__ = []


class ConvPoolLayer(torch.nn.Module):

    def __init__(self, input_channels, output_channels, filter_size, stride,
        padding, stdv, groups=1, act=None):
        super().__init__()
        self.relu = torch.nn.ReLU() if act == 'relu' else None
        self._conv = torch.nn.Conv2d(in_channels=input_channels,
            out_channels=output_channels, kernel_size=filter_size, stride=
            stride, padding=padding, groups=groups)
        self._pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, inputs):
        x = self._conv(inputs)
        if self.relu is not None:
            x = self.relu(x)
        x = self._pool(x)
        return x


class AlexNet(torch.nn.Module):
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
        self._conv1 = ConvPoolLayer(3, 64, 11, 4, 2, stdv, act='relu')
        stdv = 1.0 / math.sqrt(64 * 5 * 5)
        self._conv2 = ConvPoolLayer(64, 192, 5, 1, 2, stdv, act='relu')
        stdv = 1.0 / math.sqrt(192 * 3 * 3)
        self._conv3 = Modified_Conv2d(in_channels=192, out_channels=384,
            kernel_size=3, stride=1, padding=1)
        self._conv4 = Modified_Conv2d(in_channels=384, out_channels=256,
            kernel_size=3, stride=1, padding=1)
        stdv = 1.0 / math.sqrt(256 * 3 * 3)
        self._conv5 = ConvPoolLayer(256, 256, 3, 1, 1, stdv, act='relu')
        if self.num_classes > 0:
            self._drop1 = torch.nn.Dropout(p=0.5)
            self._fc6 = Modified_Linear(in_features=256 * 6 * 6,
                out_features=4096)
            self._drop2 = torch.nn.Dropout(p=0.5)
            self._fc7 = Modified_Linear(in_features=4096, out_features=4096)
            self._fc8 = Modified_Linear(in_features=4096, out_features=
                num_classes)

    def forward(self, inputs):
        x = self._conv1(inputs)
        x = self._conv2(x)
        x = self._conv3(x)
        x = torch.nn.functional.relu(input=x)
        x = self._conv4(x)
        x = torch.nn.functional.relu(input=x)
        x = self._conv5(x)
        if self.num_classes > 0:
            x = torch.flatten(input=x, start_dim=1, end_dim=-1)
            x = self._drop1(x)
            x = self._fc6(x)
            x = torch.nn.functional.relu(input=x)
            x = self._drop2(x)
            x = self._fc7(x)
            x = torch.nn.functional.relu(input=x)
            x = self._fc8(x)
        return x
