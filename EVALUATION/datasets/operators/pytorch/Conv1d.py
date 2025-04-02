# -*- coding: utf-8 -*-
import numpy as np


def generate_test_cases():
    # 生成num_cases组固定的3D测试用例数据，保留4位小数
    test_cases = []
    # 生成固定的权重和偏置
    # 对于 nn.Conv1d，权重的形状应该是 [out_channels, in_channels, kernel_size]
    weights = np.random.randn(6, 3, 3).astype(np.float32)  # 6 个输出通道，3 个输入通道，3 的核大小
    biases = np.random.randn(6).astype(np.float32)  # 偏置的形状应该是 [out_channels]

    for _ in range(10):
        # 输入数据的形状应该是 [batch_size, in_channels, length]
        # 假设 length 是 5, in_channels 是 3, 这里只有一个 batch
        input_array = np.random.randn(1, 3, 5).astype(np.float32)
        test_cases.append((input_array, weights, biases))
    return test_cases

def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # 定义卷积层
    conv = nn.Conv1d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1, dilation=1, groups=1)
    ret = []
    # 使用 torch.no_grad() 来禁用梯度计算
    with torch.no_grad():
        for i, (input_array, weights, biases) in enumerate(test_cases):
            # 将numpy数组转换为torch张量
            input_tensor = torch.from_numpy(input_array)
            weights_tensor = torch.from_numpy(weights)
            biases_tensor = torch.from_numpy(biases)

            conv.weight = nn.Parameter(weights_tensor)
            conv.bias = nn.Parameter(biases_tensor)

            output_tensor = conv(input_tensor)
            ret.append(output_tensor.detach().numpy())
    return ret
