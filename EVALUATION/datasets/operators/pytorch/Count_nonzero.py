# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.count_nonzero using NumPy
def generate_test_cases():
    # Generate a list of test cases with varying tensor configurations
    test_cases = [
        np.array([0, 1, 0, 2, 0, 3, 0], dtype=np.int32),   # Simple mixed zeros and non-zeros
        np.zeros(10, dtype=np.int32),                      # All zeros
        np.ones(10, dtype=np.int32),                       # No zeros
        np.random.randint(-10, 10, size=(10,)),            # Random integers including zeros
        np.full((10,), np.nan, dtype=np.float32),          # All NaNs (considered as non-zero)
        np.random.randn(5, 5),                             # 2D array of random floats
        np.array([], dtype=np.float32),                    # Empty array
        np.full((10, 10), 0.0, dtype=np.float32),          # 2D array all zeros
        np.array([np.inf, -np.inf, 0, np.nan], dtype=np.float32),  # Infinities, zero, NaN
        np.random.randint(0, 2, size=(5, 5, 5))            # 3D array with binary values
    ]
    return test_cases


# Function to test torch.count_nonzero with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.count_nonzero on multiple test cases:")
    ret = []
    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(case)

        # Apply torch.count_nonzero
        nonzero_count = torch.count_nonzero(tensor)
        ret.append(nonzero_count.item())

    return ret
