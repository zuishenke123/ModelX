# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.any using NumPy
def generate_test_cases():
    # Generate a list of test cases with different tensor configurations
    test_cases = [
        np.array([False, False, False], dtype=bool),    # All False
        np.array([True, False, False], dtype=bool),     # One True
        np.zeros((5, 5), dtype=bool),                   # 2D array all False
        np.ones((5, 5), dtype=bool),                    # 2D array all True
        np.random.choice([True, False], size=(10, 10)), # 2D array random booleans
        np.zeros((1, 100), dtype=bool),                 # 2D array with one row, all False
        np.ones((100, 1), dtype=bool),                  # 2D array with one column, all True
        np.full((10, 10), False),                       # 2D array with constant False
        np.random.choice([True, False], size=(5, 5, 5)), # 3D array with random booleans
        np.array([], dtype=bool)                        # Empty array
    ]
    return test_cases


# Function to test torch.any with generated test cases
def test_operator(test_cases):
    import torch

    print("Testing torch.any on multiple test cases:")
    ret = []

    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(case)

        # Apply torch.any
        any_result = torch.any(tensor)
        ret.append(any_result.item())
    return ret
