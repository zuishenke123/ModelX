# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.log2 using NumPy
def generate_test_cases():
    # Generate a list of test cases with varying characteristics
    test_cases = [
        np.array([1, 2, 4, 8, 16], dtype=np.float32),  # Powers of two
        np.linspace(0.1, 32, num=10, dtype=np.float32),  # Linearly spaced values from 0.1 to 32
        np.random.rand(10) * 64,  # Random positive values up to 64
        np.array([0.001, 0.01, 0.1], dtype=np.float32),  # Small positive values near zero
        np.full((10,), 1024, dtype=np.float32),  # Large constant value
        np.random.uniform(0.001, 10, size=(5, 5)),  # 2D array of small random floats
        np.array([np.inf, -np.inf, np.nan], dtype=np.float32),  # Special values: Infinity, negative infinity, NaN
        np.array([], dtype=np.float32),  # Empty array
        np.full((10, 10), 0.125, dtype=np.float32),  # 2D array with constant value less than 1
        np.array([-1, -10, -100], dtype=np.float32)  # Negative values
    ]
    return test_cases


# Function to test torch.log2 with generated test cases
def test_torch_log2(test_cases):
    import torch
    print("Testing torch.log2 on multiple test cases:")
    ret = []
    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(case)

        # Apply torch.log2, safely handle negative values and zero
        with torch.no_grad():  # To prevent tracking history in autograd
            log2_results = torch.log2(torch.clamp(tensor, min=1e-9))
            ret.append(log2_results.numpy())

    return ret
