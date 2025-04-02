# -*- coding: utf-8 -*-
import numpy as np


def generate_test_cases():
    np.random.seed(42)  # 设置随机种子以保证每次生成的数据是一致的
    test_cases = []
    # 生成固定的卷积权重和偏置
    # 对于 nn.Conv2d，权重的形状为 [out_channels, in_channels, kernel_height, kernel_width]
    weights = np.random.randn(6, 3, 3, 3).astype(np.float32)  # 例如：6个输出通道，3个输入通道，3x3卷积核
    biases = np.random.randn(6).astype(np.float32)  # 偏置的形状为 [out_channels]
    for _ in range(10):
        # 输入数据的形状为 [batch_size, in_channels, height, width]
        # 假设 height 和 width 分别为 32
        input_array = np.random.randn(1, 3, 32, 32).astype(np.float32)
        test_cases.append((input_array, weights, biases))
    return test_cases


def test_operator(test_cases):
    import torch
    import torch.nn as nn
    # 定义卷积层
    conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)

    ret = []
    with torch.no_grad():
        for input_array, weights, biases in test_cases:
            # 将numpy数组转换为torch张量
            input_tensor = torch.from_numpy(input_array)
            weights_tensor = torch.from_numpy(weights)
            biases_tensor = torch.from_numpy(biases)

            # 设置卷积层的权重和偏置
            conv.weight = nn.Parameter(weights_tensor)
            conv.bias = nn.Parameter(biases_tensor)

            # 应用卷积层
            output_tensor = conv(input_tensor)
            ret.append(output_tensor.detach().numpy())  # 将输出转换回numpy数组
    return ret

