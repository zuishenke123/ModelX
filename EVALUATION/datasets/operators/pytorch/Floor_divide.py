# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.floor_divide using NumPy
def generate_test_cases():
    # Generate a list of test cases with varying characteristics
    test_cases = [
        (np.array([10, 20, 30], dtype=np.float32), np.array([3, 4, 5], dtype=np.float32)),  # Simple division
        (np.random.randint(1, 100, size=(10,)), np.full(10, 2, dtype=np.float32)),  # Array divided by scalar 2
        (np.linspace(-10, 10, num=10, dtype=np.float32), np.full(10, -3, dtype=np.float32)),  # Negative divisor
        (np.array([0, 0, 0], dtype=np.float32), np.array([1, 2, 3], dtype=np.float32)),  # Zero numerators
        (np.array([-1, -2, -3], dtype=np.float32), np.array([-1, -1, -1], dtype=np.float32)),  # Negative values
        (np.random.rand(10, 10) * 100, np.full((10, 10), 7, dtype=np.float32)),  # 2D random floats divided by 7
        (np.array([np.inf, -np.inf, np.nan, 1], dtype=np.float32), np.array([1, 1, 1, 0], dtype=np.float32)),  # Include division by zero
        (np.array([], dtype=np.float32), np.array([], dtype=np.float32)),  # Empty arrays
        (np.random.randint(-100, 100, size=(10, 10)), np.random.randint(1, 10, size=(10, 10))),  # Avoid division by zero with random positive divisors
        (np.array([1.5, 2.5, 3.5], dtype=np.float32), np.array([1, 1, 1], dtype=np.float32))  # Fractional values
    ]
    return test_cases


# Function to test torch.floor_divide with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.floor_divide on multiple test cases:")
    ret = []
    for i, (numerator, denominator) in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor_numerator = torch.from_numpy(numerator)
        tensor_denominator = torch.from_numpy(denominator)

        # Apply torch.floor_divide
        floor_divide_results = torch.floor_divide(tensor_numerator, tensor_denominator)
        ret.append(floor_divide_results.numpy())

    return ret
