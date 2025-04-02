# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.lgamma using NumPy
def generate_test_cases():
    # Generate a list of test cases with varying characteristics
    test_cases = [
        np.array([0.5, 1.0, 2.0, 5.0, 10.0], dtype=np.float32),  # Positive values where gamma is well-defined
        np.linspace(0.1, 2, num=10, dtype=np.float32),  # Linearly spaced positive values
        np.random.uniform(1, 10, size=(10,)),  # Random positive values
        np.array([-0.5, -1.5, -2.5, -3.5], dtype=np.float32),  # Negative non-integers with poles at negative integers
        np.array([-10, -20, -30], dtype=np.float32),  # Large negative values
        np.array([np.inf, -np.inf], dtype=np.float32),  # Infinity values
        np.array([np.nan], dtype=np.float32),  # NaN value
        np.array([], dtype=np.float32),  # Empty array
        np.full((10,), 3.141592, dtype=np.float32),  # Constant value
        np.full((10, 10), -1.23, dtype=np.float32)  # 2D array with negative non-integer values
    ]
    return test_cases


# Function to test torch.lgamma with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.lgamma on multiple test cases:")
    ret = []
    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(case)

        # Apply torch.lgamma
        lgamma_results = torch.lgamma(tensor)
        ret.append(lgamma_results.numpy())

    return ret

