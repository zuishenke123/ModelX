# -*- coding: utf-8 -*-
import numpy as np


def generate_test_cases():
    # 生成固定权重和偏置
    weights = np.random.randn(3, 6, 4).astype(np.float32)  # 注意这里的维度设置，3是输出通道数，6是输入通道数
    biases = np.random.randn(6).astype(np.float32)  # 偏置的大小应与输出通道数相同
    test_cases = []
    for _ in range(10):
        # 输入数据的形状 [batch_size, out_channels, length]
        input_array = np.random.randn(1, 3, 8).astype(np.float32)  # 注意这里输入的通道数应与权重的 out_channels 匹配
        test_cases.append((input_array, weights, biases))
    return test_cases

def test_operator(test_cases):
    import torch
    import torch.nn as nn
    # 定义反卷积层
    conv_transpose = nn.ConvTranspose1d(in_channels=3, out_channels=6, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)

    ret = []
    with torch.no_grad():
        for input_array, weights, biases in test_cases:
            # 将numpy数组转换为torch张量
            input_tensor = torch.from_numpy(input_array)
            weights_tensor = torch.from_numpy(weights)
            biases_tensor = torch.from_numpy(biases)

            # 设置卷积层的权重和偏置
            conv_transpose.weight = nn.Parameter(weights_tensor)
            conv_transpose.bias = nn.Parameter(biases_tensor)

            # 应用反卷积层
            output_tensor = conv_transpose(input_tensor)
            ret.append(output_tensor.detach().numpy())  # 将输出转换回numpy数组
    return ret
