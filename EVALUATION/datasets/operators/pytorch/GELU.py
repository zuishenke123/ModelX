# -*- coding: utf-8 -*-
import numpy as np


def generate_test_cases():
    test_cases = []
    for _ in range(10):
        # 随机生成形状为 (5, 4) 的输入数据
        input_array = np.random.randn(5, 4).round(4)
        test_cases.append(input_array)
    return test_cases


def test_operator(test_cases):
    import torch.nn as nn
    import torch
    # 定义一个GELU层
    gelu = nn.GELU()  # GELU没有参数
    ret = []
    for i, input_array in enumerate(test_cases):
        # 将numpy数组转换为torch张量
        input_tensor = torch.tensor(input_array, dtype=torch.float32)

        # 计算GELU
        output_tensor = gelu(input_tensor)
        ret.append(output_tensor.detach().numpy())
    return ret