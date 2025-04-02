# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.log using NumPy
def generate_test_cases():
    # Generate a list of test cases with varying characteristics
    test_cases = [
        np.array([1, np.e, np.e**2, 10], dtype=np.float32),  # Typical values for natural log
        np.linspace(0.1, 10, num=10, dtype=np.float32),  # Linearly spaced values from 0.1 to 10
        np.random.rand(10) * 100,  # Random positive values up to 100
        np.array([0.001, 0.01, 0.1], dtype=np.float32),  # Small positive values
        np.full((10,), 1000, dtype=np.float32),  # Large constant value
        np.random.uniform(0.001, 1, size=(5, 5)),  # 2D array of small random floats
        np.array([np.inf, -np.inf, np.nan], dtype=np.float32),  # Special values: Infinity, negative infinity, NaN
        np.array([], dtype=np.float32),  # Empty array
        np.full((10, 10), 0.5, dtype=np.float32),  # 2D array with constant value less than 1
        np.array([-1, -10, -100], dtype=np.float32)  # Negative values
    ]
    return test_cases


# Function to test torch.log with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.log on multiple test cases:")
    ret = []
    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(case)

        # Apply torch.log, safely handle negative values and zero
        with torch.no_grad():  # To prevent tracking history in autograd
            log_results = torch.log(torch.clamp(tensor, min=1e-9))
            ret.append(log_results.numpy())

    return ret
