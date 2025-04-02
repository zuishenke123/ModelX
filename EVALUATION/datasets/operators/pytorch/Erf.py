# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.erf using NumPy
def generate_test_cases():
    # Generate a list of test cases with varying characteristics
    test_cases = [
        np.array([0, 0.5, -0.5, 1.0, -1.0, 2.0], dtype=np.float32),  # Common values around the mean
        np.linspace(-3, 3, num=10, dtype=np.float32),  # Linearly spaced values across the range
        np.random.randn(10),  # Random normal values
        np.zeros(10, dtype=np.float32),  # Zero values
        np.full((10,), 5, dtype=np.float32),  # Constant high values
        np.full((10,), -5, dtype=np.float32),  # Constant low values
        np.random.uniform(-10, 10, size=(5, 5)),  # 2D array of random floats
        np.array([np.inf, -np.inf, np.nan], dtype=np.float32),  # Special values: Infinity and NaN
        np.array([], dtype=np.float32),  # Empty array
        np.full((10, 10), 3, dtype=np.float32)  # 2D array with constant value
    ]
    return test_cases


# Function to test torch.erf with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.erf on multiple test cases:")
    ret = []
    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(case)

        # Apply torch.erf
        erf_results = torch.erf(tensor)
        ret.append(erf_results.numpy())

    return ret
