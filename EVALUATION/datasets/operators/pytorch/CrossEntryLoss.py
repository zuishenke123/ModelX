# -*- coding: utf-8 -*-
import numpy as np


def generate_test_cases():
    test_cases = []
    for _ in range(10):
        input_array = np.random.randn(5, 4) # 随机生成形状为 (5, 4) 的输入数据
        labels = np.random.randint(0, 4, size=(5,))  # 随机生成标签，范围在 0 到 3 之间

        test_cases.append((input_array, labels))

    return test_cases


def test_operator(test_cases):
    import torch
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss(label_smoothing=0.5)
    ret = []
    # 计算并打印每组测试用例的损失
    for i, (input_array, labels) in enumerate(test_cases):
        input_tensor = torch.tensor(input_array, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        output = criterion(input_tensor, labels_tensor)
        ret.append(output.item())
    return ret
