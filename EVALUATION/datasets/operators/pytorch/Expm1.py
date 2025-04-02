# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.expm1 using NumPy
def generate_test_cases():
    # Generate a list of test cases with varying characteristics
    test_cases = [
        np.array([0, 0.001, -0.001, 1, -1], dtype=np.float32),  # Small values around zero
        np.linspace(-1, 1, num=10, dtype=np.float32),  # Linearly spaced values within a common range
        np.random.randn(10),  # Random normal values
        np.zeros(10, dtype=np.float32),  # All zeros
        np.full((10,), 5, dtype=np.float32),  # High constant values
        np.full((10,), -5, dtype=np.float32),  # Low constant negative values
        np.random.uniform(-10, 10, size=(5, 5)),  # 2D array of random floats
        np.array([np.inf, -np.inf, np.nan], dtype=np.float32),  # Special values: Infinity and NaN
        np.array([], dtype=np.float32),  # Empty array
        np.full((10, 10), 0.1, dtype=np.float32)  # 2D array with moderate constant
    ]
    return test_cases


# Function to test torch.expm1 with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.expm1 on multiple test cases:")
    ret = []
    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(case)

        # Apply torch.expm1
        expm1_results = torch.expm1(tensor)
        ret.append(expm1_results.numpy())

    return ret
