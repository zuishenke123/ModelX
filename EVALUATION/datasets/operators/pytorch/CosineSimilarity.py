# -*- coding: utf-8 -*-
import numpy as np


def generate_test_cases():
    test_cases = []
    for _ in range(10):
        # 随机生成形状为 (5, 4) 的两个输入数据
        input1_array = np.random.randn(5, 4)
        input2_array = np.random.randn(5, 4)
        test_cases.append((input1_array, input2_array))
    return test_cases

def test_operator(test_cases):
    import torch
    import torch.nn as nn
    # 定义一个包含多种参数的CosineSimilarity层
    cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
    ret = []
    for i, (input1_array, input2_array) in enumerate(test_cases):
        # 将numpy数组转换为torch张量
        input1_tensor = torch.tensor(input1_array, dtype=torch.float32)
        input2_tensor = torch.tensor(input2_array, dtype=torch.float32)

        # 计算CosineSimilarity
        output_tensor = cosine_similarity(input1_tensor, input2_tensor)
        ret.append(output_tensor.detach().numpy())
    return ret

