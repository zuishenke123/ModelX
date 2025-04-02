# -*- coding: utf-8 -*-
import numpy as np


def generate_test_cases():
    test_cases = []
    for _ in range(10):
        # 随机生成形状为 (1, C * kernel_size[0] * kernel_size[1], L) 的输入数据
        input_array = np.random.randn(1, 3 * 3 * 3, 16)  # 假设C=3, kernel_size=(3, 3), L=16
        test_cases.append(input_array)
    return test_cases


def test_operator(test_cases):
    import torch
    import torch.nn as nn
    # 定义一个包含多种参数的Fold层，确保能够处理 (1, C * kernel_size[0] * kernel_size[1], L) 的输入
    fold = nn.Fold(output_size=(8, 8), kernel_size=(3, 3), dilation=1, padding=1, stride=2)
    ret = []

    for i, input_array in enumerate(test_cases):
        # 将numpy数组转换为torch张量
        input_tensor = torch.tensor(input_array, dtype=torch.float32)

        # 进行Fold操作
        output_tensor = fold(input_tensor)
        ret.append(output_tensor.detach().numpy())
    return ret
