# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.bitwise_not using NumPy
def generate_test_cases():
    # Generate a list of test cases with varying characteristics
    test_cases = [
        np.array([1, 2, 3, 4], dtype=np.int32),  # Simple integers
        np.random.randint(0, 256, size=(10,), dtype=np.int32),  # Random bytes
        np.array([-1, -2, -4, -8], dtype=np.int32),  # Negative numbers
        np.array([255, 256, 512], dtype=np.int32),  # Boundary conditions
        np.array([], dtype=np.int32),  # Empty array
        np.tile([170, 85], 10),  # Repeated patterns
        np.full((5, 5), 1023, dtype=np.int32),  # 2D arrays
        np.eye(5, dtype=np.int32),  # Identity matrix
        np.array([1024, 2048, 4096], dtype=np.int32),  # Powers of two
    ]
    return test_cases


# Function to test torch.bitwise_not with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.bitwise_not on multiple test cases:")
    ret = []
    for i, array in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(array)

        # Apply torch.bitwise_not
        bitwise_not_results = torch.bitwise_not(tensor)
        ret.append(bitwise_not_results.numpy())

    return ret
