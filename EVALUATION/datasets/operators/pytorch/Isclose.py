# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.isclose using NumPy
def generate_test_cases():
    # Generate a list of test cases with varying precision and tolerance levels
    test_cases = [
        (np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([1.0, 2.0, 3.0], dtype=np.float32)),  # Exact match
        (np.array([1.0, 2.0, 3.0001], dtype=np.float32), np.array([1.0, 2.0001, 3.0], dtype=np.float32)),  # Close match
        (np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([1.1, 2.1, 3.1], dtype=np.float32)),  # Not close
        (np.array([1000, 2000, 3000], dtype=np.float32), np.array([1000.1, 2000.1, 3000.1], dtype=np.float32)),  # Larger values, close
        (np.random.rand(5, 5), np.random.rand(5, 5) + 0.0001),  # Random values with small differences
        (np.zeros((10, 10)), np.full((10, 10), 0.0001)),  # Zeros and very small numbers
        (np.ones((10, 10)), np.ones((10, 10)) + 0.0001),  # Ones and ones plus a tiny bit
        (np.linspace(0, 1, 10), np.linspace(0, 1, 10) + 0.0001),  # Linearly spaced values with slight offset
        (np.full((10, 10), np.inf), np.full((10, 10), np.inf)),  # Both tensors are infinite
        (np.full((10, 10), np.nan), np.full((10, 10), np.nan))  # Both tensors are NaNs
    ]
    return test_cases


# Function to test torch.isclose with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.isclose on multiple test cases:")
    ret = []

    for i, (a, b) in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor_a = torch.from_numpy(a)
        tensor_b = torch.from_numpy(b)

        # Check if the tensors are close
        closeness_result = torch.isclose(tensor_a, tensor_b, atol=1e-4, rtol=1e-5)
        ret.append(closeness_result.numpy())
    return ret
