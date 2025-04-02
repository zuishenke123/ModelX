# -*- coding: utf-8 -*-
import numpy as np


def generate_test_cases():
    # 生成num_cases组固定的3D测试用例数据，保留4位小数
    test_cases = []
    for _ in range(10):
        input_array = np.random.randn(1, 3, 5, 5)
        test_cases.append(input_array)
    return test_cases


def test_operator(test_cases):
    import torch
    import torch.nn as nn
    # 定义卷积层
    conv_transpose = nn.ConvTranspose2d(in_channels=3, out_channels=6, kernel_size=4, stride=2, padding=1,
                                        output_padding=1, dilation=1)
    ret = []
    for i, input_array in enumerate(test_cases):
        # 将numpy数组转换为torch张量
        input_tensor = torch.tensor(input_array, dtype=torch.float32)

        # 进行反卷积操作
        output_tensor = conv_transpose(input_tensor)
        ret.append(output_tensor.detach().numpy())
    return ret
