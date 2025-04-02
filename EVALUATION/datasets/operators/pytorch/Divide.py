# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.divide using NumPy
def generate_test_cases():
    # Generate a list of test cases with varying characteristics
    test_cases = [
        (np.array([10, 20, 30], dtype=np.float32), np.array([2, 4, 5], dtype=np.float32)),  # Simple division
        (np.random.randint(1, 100, size=(10,)), 2),  # Array divided by a scalar
        (np.linspace(1, 10, 10, dtype=np.float32), np.full(10, 0.5, dtype=np.float32)),  # Division by a constant array
        (np.array([0, 0, 0], dtype=np.float32), np.array([1, 2, 3], dtype=np.float32)),  # Zero divided by numbers
        (np.random.rand(10, 10), np.random.rand(10, 10) + 0.1),  # 2D random floats, avoiding division by zero
        (np.array([-1, -2, -3], dtype=np.float32), np.array([-1, -2, -3], dtype=np.float32)),
        # Negative divided by negative
        (np.full((10,), 1.0, dtype=np.float32), 0),  # Division by zero (scalar)
        (np.array([np.inf, -np.inf, np.nan], dtype=np.float32), np.array([1, 1, 1], dtype=np.float32)),
        # Special values
        (np.array([], dtype=np.float32), np.array([], dtype=np.float32)),  # Empty arrays
        (np.array([10, 20, 30], dtype=np.float32), np.array([3, 0, 3], dtype=np.float32))  # Include division by zero
    ]
    return test_cases


# Function to test torch.divide with generated test cases
def test_torch_divide(test_cases):
    import torch
    print("Testing torch.divide on multiple test cases:")
    ret = []

    for i, (numerator, denominator) in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor_numerator = torch.from_numpy(numerator)
        tensor_denominator = torch.from_numpy(denominator if isinstance(denominator, np.ndarray) else np.array(denominator))

        # Apply torch.divide with rounding mode
        divide_results = torch.divide(tensor_numerator, tensor_denominator, rounding_mode='trunc')
        ret.append(divide_results.numpy())

    return ret



