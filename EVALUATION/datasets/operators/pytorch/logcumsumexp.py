# -*- coding: utf-8 -*-
import torch
import numpy as np


# Function to generate test cases for torch.logcumsumexp using NumPy
def generate_test_cases():
    # Generate a list of test cases with varying characteristics
    test_cases = [
        (np.array([1, 2, 3], dtype=np.float32), 0),  # Simple 1D array
        (np.linspace(-10, 10, num=10, dtype=np.float32), 0),  # Linearly spaced values
        (np.random.rand(10, 5) * 10, 1),  # 2D array with random values, dimension 1
        (np.array([-1, -2, -3, -4, -5], dtype=np.float32), 0),  # Negative values
        (np.full((10,), 10, dtype=np.float32), 0),  # Large constant values
        (np.random.randn(5, 5, 5), 2),  # 3D array of normal values, dimension 2
        (np.array([[np.inf, -np.inf, np.nan], [1, 2, 3]], dtype=np.float32), 0),  # Special values
        (np.array([], dtype=np.float32).reshape(0,10), 1),  # Empty array
        (np.full((10, 10), -20, dtype=np.float32), 0),  # 2D array with very small negative values
        (np.full((3, 3, 3), 0.001, dtype=np.float32), 2)  # 3D array with small values, dimension 2
    ]
    return test_cases


# Function to test torch.logcumsumexp with generated test cases
def test_torch_logcumsumexp(test_cases):
    import torch
    print("Testing torch.logcumsumexp on multiple test cases:")
    ret = []
    for i, (array, dim) in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(array)

        # Apply torch.logcumsumexp
        logcumsumexp_result = torch.logcumsumexp(tensor, dim)
        ret.append(logcumsumexp_result.numpy())

    return ret
