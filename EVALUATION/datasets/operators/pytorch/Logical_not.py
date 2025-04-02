# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.logical_not using NumPy
def generate_test_cases():
    # Generate a list of test cases with varying characteristics
    test_cases = [
        np.array([True, False, True, False], dtype=bool),  # Simple alternating pattern
        np.zeros((10,), dtype=bool),  # All False values
        np.ones((10,), dtype=bool),  # All True values
        np.full((5, 5), True, dtype=bool),  # 2D array with all True
        np.full((5, 5), False, dtype=bool),  # 2D array with all False
        np.random.choice([True, False], size=(10, 10)),  # Random Boolean 2D array
        np.array([], dtype=bool),  # Empty array
        np.tile([True, False], 10),  # Repeated pattern
        np.eye(5, dtype=bool),  # Identity matrix pattern
        np.tri(5, dtype=bool)  # Lower triangular matrix pattern
    ]
    return test_cases


# Function to test torch.logical_not with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.logical_not on multiple test cases:")
    ret = []
    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(case)

        # Apply torch.logical_not
        logical_not_results = torch.logical_not(tensor)
        ret.append(logical_not_results.numpy())

    return ret
