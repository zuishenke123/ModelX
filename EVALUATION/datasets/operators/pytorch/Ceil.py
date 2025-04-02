# -*- coding: utf-8 -*-
import numpy as np

# Function to generate test cases for torch.ceil using NumPy
def generate_test_cases():
    # Generate a list of test cases with various characteristics
    test_cases = [
        np.array([1.2, 2.5, 3.7, 4.1, -1.1, -2.2, -3.5], dtype=np.float32),  # Mixed positive and negative floats
        np.random.randn(10),  # Random floating-point numbers
        np.linspace(-3, 3, num=10, dtype=np.float32),  # Linearly spaced values
        np.zeros(10, dtype=np.float32),  # Zero values
        np.full((10,), 1.999, dtype=np.float32),  # Constant near-two values
        np.full((10,), -0.999, dtype=np.float32),  # Constant near-zero negative values
        np.random.uniform(-5, 5, size=(5, 5)),  # 2D array of random floats
        np.array([np.finfo(np.float32).max, -np.finfo(np.float32).max, 1e-10], dtype=np.float32),  # Infinity and NaN
        np.array([], dtype=np.float32),  # Empty array
        np.full((10, 10), 3.14, dtype=np.float32)  # 2D array with constant pi values
    ]
    return test_cases


# Function to test torch.ceil with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.ceil on multiple test cases:")
    ret = []
    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(case)
        # Apply torch.ceil
        ceil_results = torch.ceil(tensor)
        ret.append(ceil_results.numpy())

    return ret
