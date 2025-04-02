# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.digamma using NumPy
def generate_test_cases():
    # Generate a list of test cases with various characteristics
    test_cases = [
        np.array([0.5, 1.0, 1.5, -0.5, -1.5, 2.5], dtype=np.float32),  # Typical values and near poles
        np.linspace(0.1, 2, num=10, dtype=np.float32),  # Linearly spaced values in a low range
        np.random.uniform(1, 5, size=(10,)),             # Random positive values
        np.full((10,), 0.5, dtype=np.float32),           # Constant half-integer values
        np.array([-2.5, -3.5, -4.5], dtype=np.float32),  # Negative half-integers
        np.random.uniform(-10, -1, size=(5, 5)),         # 2D array of negative values
        np.array([np.inf, -np.inf, np.nan], dtype=np.float32),  # Special values: Infinity and NaN
        np.array([], dtype=np.float32),                  # Empty array
        np.full((10, 10), 1.0, dtype=np.float32),        # 2D array with a constant value of 1
        np.linspace(-0.9, 0.9, 10, dtype=np.float32)     # Values around zero
    ]
    return test_cases


# Function to test torch.digamma with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.digamma on multiple test cases:")
    ret = []

    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(case)

        # Apply torch.digamma
        digamma_results = torch.digamma(tensor)
        ret.append(digamma_results.numpy())

    return ret


