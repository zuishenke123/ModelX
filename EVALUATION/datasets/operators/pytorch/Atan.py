# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.atan using NumPy
def generate_test_cases():
    # Generate a list of test cases with various characteristics
    test_cases = [
        np.array([-1.0, 0.0, 1.0], dtype=np.float32),  # Simple values including zero
        np.linspace(-10, 10, num=10, dtype=np.float32),  # Linearly spaced values from negative to positive
        np.random.uniform(-5, 5, size=(10,)),            # Random values within a typical range
        np.full((10,), 0, dtype=np.float32),             # Constant array of zeros
        np.full((10,), 100, dtype=np.float32),           # Constant array of a large positive value
        np.full((10,), -100, dtype=np.float32),          # Constant array of a large negative value
        np.random.uniform(-100, 100, size=(5, 5)),       # 2D array of random values with large range
        np.array([np.finfo(np.float32).max, -np.finfo(np.float32).max, 1e-10], dtype=np.float32),  # Array including infinity and NaN
        np.zeros((10, 10), dtype=np.float32),            # 2D array of zeros
        np.array([], dtype=np.float32)                   # Empty array
    ]
    return test_cases


# Function to test torch.atan with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.atan on multiple test cases:")
    ret = []

    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(case)

        # Apply torch.atan
        atan_results = torch.atan(tensor)
        ret.append(atan_results.numpy())
    return ret
