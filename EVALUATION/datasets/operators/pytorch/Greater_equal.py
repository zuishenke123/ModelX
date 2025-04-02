# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.greater_equal using NumPy
def generate_test_cases():
    # Generate a list of test cases with varying characteristics
    test_cases = [
        (np.array([1, 2, 3], dtype=np.float32), np.array([3, 2, 1], dtype=np.float32)),  # Simple comparison
        (np.random.randint(0, 10, size=(10,)), np.full(10, 5, dtype=np.float32)),  # Array against constant value
        (np.linspace(-5, 5, num=10, dtype=np.float32), np.zeros(10, dtype=np.float32)),  # Range of values against zero
        (np.array([0, 0, 0], dtype=np.float32), np.array([1, 2, 3], dtype=np.float32)),  # Zero less than positive numbers
        (np.array([-1, -2, -3], dtype=np.float32), np.array([-3, -2, -1], dtype=np.float32)),  # Negative values
        (np.random.rand(10, 10), np.random.rand(10, 10)),  # 2D arrays of random floats
        (np.array([np.inf, -np.inf, np.nan], dtype=np.float32), np.array([np.inf, -np.inf, np.nan], dtype=np.float32)),  # Special values
        (np.array([], dtype=np.float32), np.array([], dtype=np.float32)),  # Empty arrays
        (np.random.randint(-100, 100, size=(10, 10)), np.random.randint(-100, 100, size=(10, 10))),  # Random integers
        (np.array([1.5, 2.5, 3.5], dtype=np.float32), np.array([1.5, 2.0, 4.0], dtype=np.float32))  # Decimal comparisons
    ]
    return test_cases


# Function to test torch.greater_equal with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.greater_equal on multiple test cases:")
    ret = []
    for i, (a, b) in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor_a = torch.from_numpy(a)
        tensor_b = torch.from_numpy(b)

        # Apply torch.greater_equal
        result = torch.greater_equal(tensor_a, tensor_b)
        ret.append(result.numpy())

    return ret
