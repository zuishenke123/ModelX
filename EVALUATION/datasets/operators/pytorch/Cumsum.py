# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.cumsum using NumPy
def generate_test_cases():
    # Generate a list of test cases with various tensor configurations
    test_cases = [
        np.array([1, 2, 3, 4], dtype=np.float32),  # Simple 1D array
        np.random.randint(1, 10, size=(10,)),  # 1D array of random integers
        np.random.randn(10, 10),  # 2D array of random floats
        np.full((10,), 5, dtype=np.float32),  # Constant array
        np.array([[1, 2], [3, 4]], dtype=np.float32),  # Small 2D array
        np.linspace(1, 100, 100).reshape(10, 10),  # 2D array with linearly increasing values
        np.zeros((10, 10), dtype=np.float32),  # 2D array of zeros
        np.array([], dtype=np.float32),  # Empty array
        np.array([np.inf, -np.inf, np.nan, 1], dtype=np.float32),  # Array with special values
        np.random.rand(5, 5, 5)  # 3D array of random floats
    ]
    return test_cases


# Function to test torch.cumsum with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.cumsum on multiple test cases:")
    ret = []

    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(case)

        # Apply torch.cumsum along the last dimension
        cumsum_results = torch.cumsum(tensor, dim=-1)
        ret.append(cumsum_results.numpy())

    return ret
