# -*- coding: utf-8 -*-
import numpy as np

# Function to generate test cases for torch.acosh using NumPy
def generate_test_cases():
    # Generate a list of numpy arrays suitable for acosh (x >= 1)
    test_cases = [
        np.array([1.0, 1.5, 2.0, 3.0, 10.0], dtype=np.float32),  # Basic range including 1
        np.full((5,), 1.0, dtype=np.float32),                     # Edge case with the lowest valid value
        np.linspace(1.0, 10.0, num=10, dtype=np.float32),         # Linearly spaced values
        np.random.uniform(1.0, 5.0, size=10).astype(np.float32),  # Random values in a controlled range
        np.random.uniform(1.0, 100.0, size=(2, 5)).astype(np.float32),  # 2D array of random values
        np.full((3, 3), 10.0, dtype=np.float32),                  # 3x3 constant array
        np.array([1.00001, 2.0, 3.14159, 10.00001], dtype=np.float32),  # Near the edge values
        np.array([1.0] + list(np.random.uniform(1.0, 5.0, size=4)), dtype=np.float32),  # Mixed with edge value
        np.exp(np.linspace(0, 3, num=10, dtype=np.float32)),      # Exponential values
        np.geomspace(1.0, 1000.0, num=10, dtype=np.float32)       # Geometric progression
    ]
    return test_cases


# Function to test torch.acosh with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.acosh on multiple test cases:")
    ret = []

    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors just before the operation
        tensor = torch.from_numpy(case)

        # Apply torch.acosh
        result = torch.acosh(tensor)
        ret.append(result.numpy())

    return ret
