# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.exp using NumPy
def generate_test_cases():
    # Generate a list of test cases with various characteristics
    test_cases = [
        np.array([0, 1, -1, 10, -10], dtype=np.float32),  # Typical small and large values
        np.linspace(-2, 2, num=10, dtype=np.float32),  # Linearly spaced values in a moderate range
        np.random.randn(10),  # Random normal values
        np.zeros(10, dtype=np.float32),  # All zeros
        np.full((10,), 20, dtype=np.float32),  # High constant values
        np.full((10,), -20, dtype=np.float32),  # Low negative values
        np.random.uniform(-5, 5, size=(5, 5)),  # 2D array of random floats
        np.array([np.inf, -np.inf, np.nan], dtype=np.float32),  # Special values: Infinity and NaN
        np.array([], dtype=np.float32),  # Empty array
        np.full((10, 10), 0.5, dtype=np.float32)  # 2D array with a moderate constant
    ]
    return test_cases


# Function to test torch.exp with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.exp on multiple test cases:")
    ret = []
    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(case)

        # Apply torch.exp
        exp_results = torch.exp(tensor)
        ret.append(exp_results.numpy())

    return ret
