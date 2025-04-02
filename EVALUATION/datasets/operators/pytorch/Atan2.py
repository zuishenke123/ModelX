# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.atan2 using NumPy
def generate_test_cases():
    test_cases = [
        (np.random.uniform(-10, 10, size=10), np.random.uniform(-10, 10, size=10)),  # 随机1D数组
        (np.random.uniform(-10, 10, size=100), np.random.uniform(-10, 10, size=100)),  # 更大的1D数组
        (np.random.uniform(-10, 10, size=(10, 10)), np.random.uniform(-10, 10, size=(10, 10))),  # 2D数组
        (np.random.uniform(-10, 10, size=(5, 5, 5)), np.random.uniform(-10, 10, size=(5, 5, 5))),  # 3D数组
        (np.random.uniform(-1, 1, size=50), np.random.uniform(-1, 1, size=50)),  # 较小范围的随机值
        (np.random.uniform(-100, 100, size=20), np.random.uniform(-100, 100, size=20))  # 较大范围的随机值
    ]
    return test_cases


# Function to test torch.atan2 with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.atan2 on multiple test cases:")
    ret = []
    for i, (y, x) in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor_y = torch.from_numpy(y)
        tensor_x = torch.from_numpy(x)

        # Apply torch.atan2
        atan2_results = torch.atan2(tensor_y, tensor_x)
        ret.append(atan2_results.numpy())

    return ret
