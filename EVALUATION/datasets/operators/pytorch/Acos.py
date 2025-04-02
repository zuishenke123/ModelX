# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.acos using NumPy
def generate_test_cases():
    # Generate a list of numpy arrays with values within the domain of acos
    test_cases = [
        np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32),  # Simple test case
        np.array([[-1.0, 0.0], [0.5, 1.0]], dtype=np.float32),    # 2D array
        np.linspace(-1, 1, num=10, dtype=np.float32),             # Range of values within domain
        np.random.uniform(-1, 1, size=10).astype(np.float32),     # Random values within domain
        np.random.uniform(-1, 1, size=(5, 2)).astype(np.float32), # 2D random values within domain
        np.array([1.0, -0.3, 0.2, -0.7, 0.5], dtype=np.float32),  # Mixed positive and negative values
        np.full((10,), -1.0, dtype=np.float32),                   # All at lower boundary
        np.full((10,), 1.0, dtype=np.float32),                    # All at upper boundary
        np.random.uniform(-1, 1, size=(3, 3)).astype(np.float32), # 3x3 matrix of random values
        np.array([-0.1, -0.2, 0.1, 0.2, -0.3, 0.3], dtype=np.float32) # Smaller range within domain
    ]
    return test_cases


# Function to test torch.acos with generated test cases
def test_operator(test_cases):
    import torch

    print("Testing torch.acos on multiple test cases:")
    ret = []

    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(case)

        # Apply torch.acos
        result = torch.acos(tensor)
        ret.append(result.numpy())

    return ret
