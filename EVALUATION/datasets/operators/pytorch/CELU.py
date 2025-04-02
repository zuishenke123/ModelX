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
    import torch
    import torch.nn as nn
    # 定义一个包含参数的CELU层
    celu = nn.CELU(alpha=1.0)  # 默认alpha=1.0
    ret = []

    for i, input_array in enumerate(test_cases):
        # 将numpy数组转换为torch张量
        input_tensor = torch.tensor(input_array)

        # 计算CELU
        output_tensor = celu(input_tensor)
        ret.append(output_tensor.detach().numpy())
    return ret

