# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.cumprod using NumPy
def generate_test_cases():
    # Generate a list of test cases with varying characteristics
    test_cases = [
        np.array([1, 2, 3, 4], dtype=np.float32),  # Simple sequential numbers
        np.random.randint(1, 5, size=(10,)),  # 1D array of small integers
        np.random.rand(10, 5),  # 2D array of random floats
        np.full((10,), 0.5, dtype=np.float32),  # Constant values less than 1
        np.array([[1, 2], [3, 4]], dtype=np.float32),  # Small 2D array
        np.linspace(1, 12, 12).reshape(3, 4),  # 2D array with linearly increasing values
        np.ones((10, 10), dtype=np.float32),  # 2D array of ones
        np.array([], dtype=np.float32),  # Empty array
        np.array([np.inf, -np.inf, np.nan, 1], dtype=np.float32),  # Array with special values
        np.random.rand(5, 5, 5)  # 3D array of random floats
    ]
    return test_cases


# Function to test torch.cumprod with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.cumprod on multiple test cases:")
    ret = []
    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(case)

        # Apply torch.cumprod along the last dimension
        cumprod_results = torch.cumprod(tensor, dim=-1)
        ret.append(cumprod_results.numpy())

    return ret