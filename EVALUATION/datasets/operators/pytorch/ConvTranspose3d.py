# -*- coding: utf-8 -*-
import numpy as np


def generate_test_cases():
    # 生成num_cases组固定的3D测试用例数据，保留4位小数
    test_cases = []
    for _ in range(10):
        input_array = np.random.randn(1, 3, 5, 5, 5)
        test_cases.append(input_array)
    return test_cases


def test_operator(test_cases):
    import torch
    import torch.nn as nn
    # 定义一个包含多种参数的卷积层，确保能够处理 (1, 3, 5, 5, 5) 的输入
    conv = nn.Conv3d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1, dilation=1, groups=1)
    ret = []

    for i, input_array in enumerate(test_cases):
        # 将numpy数组转换为torch张量
        input_tensor = torch.tensor(input_array, dtype=torch.float32)

        # 进行卷积操作
        output_tensor = conv(input_tensor)
        ret.append(output_tensor.detach().numpy())
    return ret
