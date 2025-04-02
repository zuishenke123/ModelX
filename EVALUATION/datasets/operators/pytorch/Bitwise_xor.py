# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.bitwise_xor using NumPy
def generate_test_cases():
    # Generate a list of test cases with varying characteristics
    test_cases = [
        (np.array([1, 2, 3, 4], dtype=np.int32), np.array([4, 3, 2, 1], dtype=np.int32)),  # Simple integers
        (np.random.randint(0, 256, size=(10,), dtype=np.int32), np.full(10, 128, dtype=np.int32)),  # Random bytes and mask
        (np.array([-1, -2, -4, -8], dtype=np.int32), np.array([-1, -1, -1, -1], dtype=np.int32)),  # Negative numbers
        (np.array([255, 256, 512], dtype=np.int32), np.array([128, 128, 128], dtype=np.int32)),  # Boundary conditions
        (np.array([], dtype=np.int32), np.array([], dtype=np.int32)),  # Empty arrays
        (np.tile([170, 85], 10), np.tile([255, 0], 10)),  # Repeated pattern arrays
        (np.full((5, 5), 1023, dtype=np.int32), np.full((5, 5), 511, dtype=np.int32)),  # 2D arrays
        (np.eye(5, dtype=np.int32), np.full((5, 5), 1, dtype=np.int32)),  # Identity matrix
        (np.array([1024, 2048, 4096], dtype=np.int32), np.full(3, 2047, dtype=np.int32)),  # Power of two
    ]
    return test_cases


# Function to test torch.bitwise_xor with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.bitwise_xor on multiple test cases:")
    ret = []
    for i, (a, b) in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor_a = torch.from_numpy(a)
        tensor_b = torch.from_numpy(b)

        # Apply torch.bitwise_xor
        bitwise_xor_results = torch.bitwise_xor(tensor_a, tensor_b)
        ret.append(bitwise_xor_results.numpy())

    return ret
