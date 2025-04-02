# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.floor using NumPy
def generate_test_cases():
    # Generate a list of test cases with varying characteristics
    test_cases = [
        np.array([1.7, 2.5, 3.3, -1.2, -2.8, 0.0], dtype=np.float32),  # Mixed positive and negative floats
        np.linspace(-5.5, 5.5, num=10, dtype=np.float32),  # Linearly spaced values
        np.random.uniform(-10, 10, size=(10,)),  # Random floating-point numbers
        np.zeros(10, dtype=np.float32),  # All zeros
        np.full((10,), 3.999, dtype=np.float32),  # Values near an integer
        np.random.uniform(-5, 5, size=(5, 5)),  # 2D array of random floats
        np.array([np.inf, -np.inf, np.nan], dtype=np.float32),  # Special values: Infinity and NaN
        np.array([], dtype=np.float32),  # Empty array
        np.full((10, 10), -0.1, dtype=np.float32),  # 2D array of negative non-integers
        np.linspace(0.01, 1.0, 10, dtype=np.float32)  # Small positive numbers close to zero
    ]
    return test_cases


# Function to test torch.floor with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.floor on multiple test cases:")
    ret = []
    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(case)

        # Apply torch.floor
        floor_results = torch.floor(tensor)
        ret.append(floor_results.numpy())

    return ret
