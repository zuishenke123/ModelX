# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.equal using NumPy
def generate_test_cases():
    # Generate a list of test cases with varying characteristics
    test_cases = [
        (np.array([1, 2, 3], dtype=np.float32), np.array([1, 2, 3], dtype=np.float32)),  # Exactly the same
        (np.array([1, 2, 3], dtype=np.float32), np.array([1, 2, 4], dtype=np.float32)),  # Different values
        (np.array([1, 2, 3], dtype=np.float32), np.array([1.0, 2.0, 3.0], dtype=np.float64)),  # Different types
        (np.array([1, 2, 3], dtype=np.float32), np.array([1, 2, 3, 4], dtype=np.float32)),  # Different shapes
        (np.zeros((10, 10), dtype=np.float32), np.zeros((10, 10), dtype=np.float32)),  # All zeros
        (np.ones((5, 5), dtype=np.float32), np.ones((5, 5), dtype=np.float32)),  # All ones
        (np.array([], dtype=np.float32), np.array([], dtype=np.float32)),  # Empty arrays
        (np.array([np.inf, -np.inf, np.nan], dtype=np.float32), np.array([np.inf, -np.inf, np.nan], dtype=np.float32)),  # Special values
        (np.random.rand(10, 10), np.random.rand(10, 10)),  # Random values, unlikely to be equal
        (np.full((3, 3), 7, dtype=np.float32), np.full((3, 3), 7, dtype=np.float32))  # Constant arrays
    ]
    return test_cases


# Function to test torch.equal with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.equal on multiple test cases:")
    ret = []
    for i, (a, b) in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor_a = torch.from_numpy(a)
        tensor_b = torch.from_numpy(b)

        # Apply torch.equal
        are_equal = torch.equal(tensor_a, tensor_b)
        ret.append(are_equal)

    return ret
