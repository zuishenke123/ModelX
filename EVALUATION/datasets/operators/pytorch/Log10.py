# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.log10 using NumPy
def generate_test_cases():
    # Generate a list of test cases with varying characteristics
    test_cases = [
        np.array([1, 10, 100, 1000], dtype=np.float32),  # Powers of ten
        np.linspace(0.1, 100, num=10, dtype=np.float32),  # Linearly spaced values from 0.1 to 100
        np.random.rand(10) * 1000,  # Random positive values up to 1000
        np.array([0.001, 0.01, 0.1], dtype=np.float32),  # Small positive values near zero
        np.full((10,), 10000, dtype=np.float32),  # Large constant value
        np.random.uniform(0.001, 1, size=(5, 5)),  # 2D array of small random floats
        np.array([np.inf, -np.inf, np.nan], dtype=np.float32),  # Special values: Infinity, negative infinity, NaN
        np.array([], dtype=np.float32),  # Empty array
        np.full((10, 10), 0.5, dtype=np.float32),  # 2D array with constant value less than 10
        np.array([-1, -10, -100], dtype=np.float32)  # Negative values
    ]
    return test_cases



# Function to test torch.log10 with generated test cases
def test_torch_log10(test_cases):
    import torch
    print("Testing torch.log10 on multiple test cases:")
    ret = []
    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(case)

        # Apply torch.log10, safely handle negative values and zero
        with torch.no_grad():  # To prevent tracking history in autograd
            log10_results = torch.log10(torch.clamp(tensor, min=1e-9))
            ret.append(log10_results.numpy())

    return ret
