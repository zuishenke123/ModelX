# -*- coding: utf-8 -*-
import numpy as np


def generate_test_cases():
    test_cases = []
    for _ in range(10):
        # 随机生成形状为 (1, 3, 8, 8) 的输入数据
        input_array = np.random.randn(1, 3, 8, 8).round(4)
        test_cases.append(input_array)
    return test_cases


def test_operator(test_cases):
    import torch
    import torch.nn as nn
    # 定义一个包含多种参数的Unfold层
    unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=2)
    ret = []

    for i, input_array in enumerate(test_cases):
        # 将numpy数组转换为torch张量
        input_tensor = torch.tensor(input_array, dtype=torch.float32)

        # 进行Unfold操作
        output_tensor = unfold(input_tensor)
        ret.append(output_tensor.detach().numpy())

    return ret
